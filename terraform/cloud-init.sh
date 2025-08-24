#!/bin/bash

cd /root

curl -LsSf https://astral.sh/uv/install.sh | sh
source /root/.local/bin/env

git clone https://github.com/kilianmandon/seven-wonders-rl.git
cd seven-wonders-rl

uv venv --python 3.10
source .venv/bin/activate

uv pip install -r requirements.txt