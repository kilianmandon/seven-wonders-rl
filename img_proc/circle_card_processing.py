from pathlib import Path

import cv2
import numpy as np


if __name__ == "__main__":
    target_size = 200
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 100, 255, -1)

    for img_path in Path('images/progress_tokens_cropped').glob('*.jpg'):
        img = cv2.imread(img_path)
        resized = cv2.resize(img, (200, 200))
        resized_rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)
            
        resized_rgba[:, :, 3] = mask
            
        cv2.imwrite(f'images/processed_cards/progress_tokens/{img_path.stem}.png', resized_rgba)