import cv2
import numpy as np

def hex_to_bgr(h):
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return rgb[::-1]
    
def main():
    mask = cv2.imread('img_proc/card_transform_params_mask.png', cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape
    inner_mask = cv2.resize(mask, (w-20, h-20))
    inner_mask = np.pad(inner_mask, 10)
    colors_hex = [
        'A25531',
        '0086B7',
        'AE8BB2',
        '70669D',
        '000000',
    ]

    colors = [hex_to_bgr(h) for h in colors_hex]
    names = ['hidden_age_1', 'hidden_age_2', 'hidden_age_3', 'hidden_guilds', 'missing']

    for color, name in zip(colors, names):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[inner_mask>100] = color

        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img[:, :, 3] = mask
        cv2.imwrite(f'images/processed_cards/dummies/{name}_processed.png', img)




    

if __name__=='__main__':
    main()