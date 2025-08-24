import tensordict
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
import torch
from torchrl.envs.utils import check_env_specs
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torch import nn
import torchrl
from torchrl.modules import ProbabilisticActor, MaskedCategorical

from torchrl.objectives import ClipPPOLoss, ValueEstimators
import tqdm

from pettingzoo_env import GameEnv





def create_rollout(policy=None):
    base_env = GameEnv(store_states=True)
    env = PettingZooWrapper(base_env, use_mask=True)
    env.rollout(max_steps=100, policy=policy)
    base_env.close()


def create_policy(player, env, config):
    n_obs_dim = config['n_obs_dim']
    n_actions = config['n_actions']
    c_h = config['c_h']

    policy_net = nn.Sequential(
        nn.Linear(n_obs_dim, c_h),
        nn.ReLU(),
        nn.Linear(c_h, c_h),
        nn.ReLU(),
        nn.Linear(c_h, n_actions),
    )

    policy_module = tensordict.nn.TensorDictModule(
        policy_net,
        in_keys=[(player, 'observation', 'observation')],
        out_keys=[(player, 'logits')]
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec[player, 'action'],
        in_keys={'logits': (player, 'logits'),
                 'mask': (player, 'action_mask')},
        out_keys=[(player, 'action')],
        distribution_class=MaskedCategorical,
        return_log_prob=True
    )

    return policy

def create_random_policy(player, env, config):
    n_actions = config['n_actions']

    uniform_module = tensordict.nn.TensorDictModule(
        lambda s: torch.ones((1, n_actions)),
        in_keys=[(player, 'observation', 'observation')], out_keys=[(player, 'logits')]
    )
    policy = ProbabilisticActor(
        module=uniform_module,
        spec=env.action_spec[player, 'action'],
        in_keys={'logits': (player, 'logits'), 'mask': (player, 'action_mask')},
        out_keys=[(player, 'action')],
        distribution_class=MaskedCategorical,
        default_interaction_type=tensordict.nn.InteractionType.RANDOM
    )
    return policy


def create_critic(player, config):
    n_obs_dim = config['n_obs_dim']
    c_h = config['c_h']

    critic_net = nn.Sequential(
        nn.Linear(n_obs_dim, c_h),
        nn.ReLU(),
        nn.Linear(c_h, c_h),
        nn.ReLU(),
        nn.Linear(c_h, 1),
    )
    critic = tensordict.nn.TensorDictModule(
        critic_net,
        in_keys=[(player, 'observation', 'observation')],
        out_keys=[(player, 'state_value')]
    )

    return critic


def create_loss_module(player, policy, critic):

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        entropy_coeff=1e-4,
        normalize_advantage=False
        )
    loss_module.set_keys(
        reward=(player, 'reward'),
        action=(player, 'action'),
        value=(player, 'state_value'),
        done=(player, 'done'),
        terminated=(player, 'terminated')
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=0.99, lmbda=0.9
    )


    return loss_module




def training(players, policies, collector, replay_buffers, losses, optimizers, config):
    n_iters = config['n_iters']
    num_epochs = config['num_epochs']
    frames_per_batch = config['frames_per_batch']
    minibatch_size = config['minibatch_size']
    last_bench_desc = ''

    pbar_desc = ', '.join(
        f'episode_reward_mean_{player} = 0' for player in players)
    pbar = tqdm.tqdm(total=n_iters, desc=pbar_desc)

    for iteration, batch in enumerate(collector):
        for player in players:
            with torch.no_grad():
                losses[player].value_estimator(batch, params=losses[player].critic_network_params, target_params=losses[player].target_critic_network_params)

            other_player = 'player_0' if player == 'player_1' else 'player_1'
            player_batch = batch.exclude(other_player)
            replay_buffers[player].extend(player_batch)

        for _ in range(num_epochs):
            for _ in range(frames_per_batch//minibatch_size):
                for player in players:
                    subdata = replay_buffers[player].sample()
                    loss_vals = losses[player](subdata)
                    loss_value = (
                        loss_vals['loss_objective'] + loss_vals['loss_critic'] + loss_vals['loss_entropy']
                    )
                    loss_value.backward()
                    optimizers[player].step()
                    optimizers[player].zero_grad()

        collector.update_policy_weights_()

        if (iteration+1) % 5==0:
            last_bench_desc = bench_against_random(policies['player_0'], 'player_0', config)

        done = batch.get(('next', 'player_0', 'done'))
        episode_reward_means = { player:
            batch.get(('next', player, 'reward'))[done].mean().item()
        for player in players}

        pbar_desc = 'Mean Reward: ' + ' | '.join( 
            f'{episode_reward_means[player]:.2f}' for player in players)
        pbar_desc += f' ; {last_bench_desc}'
        pbar.set_description(pbar_desc)
        pbar.update()


def bench_against_random(policy, policy_player_name, config): 
    env = GameEnv()
    env = PettingZooWrapper(env, use_mask=True)

    other_player = 'player_0' if policy_player_name == 'player_1' else 'player_1'
    random_policy = create_random_policy(other_player, env, config)

    full_policy = tensordict.nn.TensorDictSequential(policy, random_policy)

    win_counter = 0
    loss_counter = 0
    draw_counter = 0
    for _ in range(100):
        env.reset()
        rollout = env.rollout(max_steps=100, policy=full_policy)
        outcome = rollout['next', policy_player_name, 'reward'].squeeze()[-1].item()
        if outcome==1:
            win_counter+=1
        elif outcome==0:
            draw_counter += 1
        else:
            loss_counter += 1


    bench_desc = f'vs Random: {win_counter} W | {loss_counter} L | {draw_counter} D'
    return bench_desc
        



def main():
    frames_per_batch = 6000
    n_iters = 60
    total_frames = frames_per_batch * n_iters

    num_epochs = 30
    minibatch_size = 400
    lr = 1e-4

    n_obs_dim = 1966
    n_actions = 386

    config = {
        'n_iters': n_iters,
        'num_epochs': num_epochs,
        'frames_per_batch': frames_per_batch,
        'minibatch_size': minibatch_size,
        'n_obs_dim': n_obs_dim,
        'n_actions': n_actions,
        'c_h': 256,
        'lr': lr,
    }

    env = GameEnv()
    env = PettingZooWrapper(env, use_mask=True)


    check_env_specs(env)

    players = ['player_0', 'player_1']
    policies = {player: create_policy(player, env, config) for player in players}
    critics = {player: create_critic(player, config) for player in players}
    losses = {player: create_loss_module(
        player, policies[player], critics[player]) for player in players}
    optimizers = {player: torch.optim.Adam(
        losses[player].parameters(), config['lr']) for player in players}

    full_policy = tensordict.nn.TensorDictSequential(*policies.values())

    collector = SyncDataCollector(env, full_policy, frames_per_batch=frames_per_batch,
                    total_frames=total_frames)

    replay_buffers = {
        player: ReplayBuffer(
            storage=LazyTensorStorage(frames_per_batch),
            sampler=SamplerWithoutReplacement(), 
            batch_size=minibatch_size) 
            for player in players 
    }


    training(players, policies, collector, replay_buffers, losses, optimizers, config)

    torch.save(policies['player_0'].module[0].module.state_dict(), 'checkpoints/actor.pt')
    torch.save(critics['player_0'].module.state_dict(), 'checkpoints/critic.pt')

if __name__=='__main__':
    main()