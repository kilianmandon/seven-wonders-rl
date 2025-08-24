import cv2
import numpy as np
import os
import glob
from pathlib import Path
import random
import json
import pickle

class CardSegmentationPipeline:
    def __init__(self, input_folder, output_folder, target_width=400, target_height=560, 
                 transform_params_file=None):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.target_width = target_width
        self.target_height = target_height
        self.target_ratio = target_width / target_height
        self.transform_params_file = transform_params_file
        
        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # FloodFill settings
        self.lo_diff = 4
        self.up_diff = 4
        
        # Storage for all card contours and masks
        self.all_card_contours = []
        self.unified_mask = None
        
        # Transformation parameters
        self.transform_params = None
    
    def auto_floodfill_background(self, img):
        """Automatically segment background using floodfill at center-95% height"""
        if img is None:
            raise FileNotFoundError("Image not found")
        
        h, w = img.shape[:2]
        
        # Auto point at center horizontally, 95% height
        x = w // 2
        y = int(h * 0.95)
        
        # Mask for floodFill: must be 2 pixels larger
        mask = np.zeros((h+2, w+2), np.uint8)
        
        # Flood fill from the auto point
        temp_img = img.copy()
        retval, _, mask, _ = cv2.floodFill(temp_img, mask, (x, y),
                                         newVal=(0, 0, 0),
                                         loDiff=(self.lo_diff,)*3,
                                         upDiff=(self.up_diff,)*3,
                                         flags=4 | (255 << 8))
        
        # Remove padding from mask and invert (we want card, not background)
        background_mask = mask[1:-1, 1:-1]
        card_mask = cv2.bitwise_not(background_mask)
        
        return card_mask
    
    def find_card_contour(self, mask):
        """Find the best four-point contour for the card"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the largest contour (should be the card)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate to 4 points
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we don't get exactly 4 points, use bounding rectangle
        if len(approx) != 4:
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            approx = np.int0(box)
        
        return approx.reshape(4, 2)
    
    def order_points(self, pts):
        """Order points as: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum and difference to find corners
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left
        
        return rect
    
    def get_card_dimensions(self, ordered_pts):
        """Calculate width and height from ordered points"""
        (tl, tr, br, bl) = ordered_pts
        
        # Calculate width and height
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        
        max_width = max(int(width_a), int(width_b))
        max_height = max(int(height_a), int(height_b))
        
        return max_width, max_height
    
    def create_rounded_corner_mask(self, width, height, corner_radius=20):
        """Create a mask with rounded corners"""
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Create rounded corners by drawing circles at corners
        # Top-left
        cv2.circle(mask, (corner_radius, corner_radius), corner_radius, 0, -1)
        cv2.rectangle(mask, (0, 0), (corner_radius, corner_radius), 0, -1)
        
        # Top-right  
        cv2.circle(mask, (width - corner_radius, corner_radius), corner_radius, 0, -1)
        cv2.rectangle(mask, (width - corner_radius, 0), (width, corner_radius), 0, -1)
        
        # Bottom-right
        cv2.circle(mask, (width - corner_radius, height - corner_radius), corner_radius, 0, -1)
        cv2.rectangle(mask, (width - corner_radius, height - corner_radius), (width, height), 0, -1)
        
        # Bottom-left
        cv2.circle(mask, (corner_radius, height - corner_radius), corner_radius, 0, -1)
        cv2.rectangle(mask, (0, height - corner_radius), (corner_radius, height), 0, -1)
        
        # Now draw filled circles to create the rounded effect
        cv2.circle(mask, (corner_radius, corner_radius), corner_radius, 255, -1)
        cv2.circle(mask, (width - corner_radius, corner_radius), corner_radius, 255, -1)
        cv2.circle(mask, (width - corner_radius, height - corner_radius), corner_radius, 255, -1)
        cv2.circle(mask, (corner_radius, height - corner_radius), corner_radius, 255, -1)
        
        return mask
    
    def process_single_image(self, image_path):
        """Process a single image and return card info"""
        print(f"Processing: {os.path.basename(image_path)}")
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return None
        
        # Auto segment background
        card_mask = self.auto_floodfill_background(img)
        
        # Find card contour
        card_contour = self.find_card_contour(card_mask)
        if card_contour is None:
            print(f"Could not find card contour in: {image_path}")
            return None
        
        # Order points and get dimensions
        ordered_pts = self.order_points(card_contour)
        width, height = self.get_card_dimensions(ordered_pts)
        
        return {
            'path': image_path,
            'image': img,
            'mask': card_mask,
            'contour': card_contour,
            'ordered_pts': ordered_pts,
            'width': width,
            'height': height,
            'ratio': width / height if height > 0 else 1.0
        }
    
    def create_unified_mask(self):
        """Create a unified rounded corner mask based on target dimensions"""
        corner_radius = min(self.target_width, self.target_height) // 20  # 5% of smaller dimension
        self.unified_mask = self.create_rounded_corner_mask(
            self.target_width, self.target_height, corner_radius
        )
    
    def save_transform_params(self, params_file):
        """Save transformation parameters to file"""
        self.transform_params = {
            'target_width': self.target_width,
            'target_height': self.target_height,
            'target_ratio': self.target_ratio,
            'corner_radius': min(self.target_width, self.target_height) // 20,
            'lo_diff': self.lo_diff,
            'up_diff': self.up_diff
        }
        
        # Save parameters as JSON
        with open(params_file, 'w') as f:
            json.dump(self.transform_params, f, indent=2)
        
        # Save the unified mask
        mask_file = params_file.replace('.json', '_mask.png')
        if self.unified_mask is not None:
            cv2.imwrite(mask_file, self.unified_mask)
        
        print(f"Transform parameters saved to: {params_file}")
        print(f"Unified mask saved to: {mask_file}")
    
    def load_transform_params(self, params_file):
        """Load transformation parameters from file"""
        if not os.path.exists(params_file):
            print(f"Parameters file not found: {params_file}")
            return False
        
        # Load parameters
        with open(params_file, 'r') as f:
            self.transform_params = json.load(f)
        
        # Apply loaded parameters
        self.target_width = self.transform_params['target_width']
        self.target_height = self.transform_params['target_height']
        self.target_ratio = self.transform_params['target_ratio']
        self.lo_diff = self.transform_params.get('lo_diff', 4)
        self.up_diff = self.transform_params.get('up_diff', 4)
        
        # Load the unified mask
        mask_file = params_file.replace('.json', '_mask.png')
        if os.path.exists(mask_file):
            self.unified_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            print(f"Loaded unified mask from: {mask_file}")
        
        print(f"Transform parameters loaded from: {params_file}")
        print(f"Target dimensions: {self.target_width}x{self.target_height}")
        return True
    
    def warp_and_crop_card(self, card_info):
        """Warp and crop card to target dimensions with rounded corners"""
        img = card_info['image']
        ordered_pts = card_info['ordered_pts']
        
        # Destination points for perspective transform
        dst_pts = np.array([
            [0, 0],
            [self.target_width - 1, 0],
            [self.target_width - 1, self.target_height - 1],
            [0, self.target_height - 1]
        ], dtype="float32")
        
        # Get perspective transform matrix and warp
        M = cv2.getPerspectiveTransform(ordered_pts.astype("float32"), dst_pts)
        warped = cv2.warpPerspective(img, M, (self.target_width, self.target_height))
        
        # Apply unified rounded corner mask
        if self.unified_mask is not None:
            # Convert to BGRA (with alpha channel)
            warped_rgba = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
            
            # Apply mask to alpha channel
            warped_rgba[:, :, 3] = self.unified_mask
            
            return warped_rgba
        
        return warped
    
    def process_folder(self, save_params_to=None):
        """Process all images in the input folder"""
        # If transform_params_file is provided, try to load existing parameters
        if self.transform_params_file and os.path.exists(self.transform_params_file):
            if self.load_transform_params(self.transform_params_file):
                return self._process_with_existing_params()
        
        # Otherwise, process normally and optionally save parameters
        return self._process_and_calculate_params(save_params_to)
    
    def _process_with_existing_params(self):
        """Process folder using pre-existing transformation parameters"""
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(self.input_folder, ext.upper())))
        
        if not image_files:
            print(f"No images found in {self.input_folder}")
            return
        
        print(f"Found {len(image_files)} images")
        print(f"Using existing transformation parameters:")
        print(f"  Dimensions: {self.target_width}x{self.target_height}")
        print(f"  Ratio: {self.target_ratio:.3f}")
        
        # Process all images with existing parameters
        processed_count = 0
        for image_path in image_files:
            card_info = self.process_single_image(image_path)
            if card_info:
                try:
                    # Warp and crop card using existing parameters
                    processed_card = self.warp_and_crop_card(card_info)
                    
                    # Generate output filename
                    input_name = Path(card_info['path']).stem
                    output_path = os.path.join(self.output_folder, f"{input_name}_processed.png")
                    
                    # Save processed card
                    cv2.imwrite(output_path, processed_card)
                    print(f"Saved: {output_path}")
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing {card_info['path']}: {e}")
        
        print(f"\nProcessing complete! Processed {processed_count} cards using existing parameters")
        print(f"Output saved to: {self.output_folder}")
    
    def _process_and_calculate_params(self, save_params_to=None):
        """Process folder and calculate new transformation parameters"""
        """Process all images in the input folder"""
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(self.input_folder, ext.upper())))
        
        if not image_files:
            print(f"No images found in {self.input_folder}")
            return
        
        print(f"Found {len(image_files)} images")
        
        # Process all images to collect card info
        card_infos = []
        ratios = []
        
        for image_path in image_files:
            card_info = self.process_single_image(image_path)
            if card_info:
                card_infos.append(card_info)
                ratios.append(card_info['ratio'])
        
        if not card_infos:
            print("No valid cards found!")
            return
        
        # Calculate average ratio and adjust target dimensions if needed
        avg_ratio = np.mean(ratios)
        print(f"Average card ratio: {avg_ratio:.3f}")
        
        # Optionally adjust target dimensions to match average ratio
        if abs(avg_ratio - self.target_ratio) > 0.1:
            if avg_ratio > self.target_ratio:
                # Cards are wider than target
                self.target_width = int(self.target_height * avg_ratio)
            else:
                # Cards are taller than target
                self.target_height = int(self.target_width / avg_ratio)
            print(f"Adjusted target dimensions to: {self.target_width}x{self.target_height}")
            self.target_ratio = self.target_width / self.target_height
        
        # Create unified mask
        self.create_unified_mask()
        
        # Save transformation parameters if requested
        if save_params_to:
            self.save_transform_params(save_params_to)
        
        # Process and save all cards
        processed_count = 0
        for i, card_info in enumerate(card_infos):
            try:
                # Warp and crop card
                processed_card = self.warp_and_crop_card(card_info)
                
                # Generate output filename
                input_name = Path(card_info['path']).stem
                output_path = os.path.join(self.output_folder, f"{input_name}_processed.png")
                
                # Save processed card
                cv2.imwrite(output_path, processed_card)
                print(f"Saved: {output_path}")
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {card_info['path']}: {e}")
        
        print(f"\nProcessing complete! Processed {processed_count} cards")
        print(f"Output saved to: {self.output_folder}")


# Usage examples
def card_processing():
    # STEP 1: Process the first folder (age_1) and save transformation parameters
    print("=== Processing age_1 folder (master reference) ===")
    pipeline_age1 = CardSegmentationPipeline(
        input_folder="images/age_1",
        output_folder="images/processed_cards/age_1",
        target_width=400,
        target_height=560
    )
    
    # Process and save the transformation parameters
    params_file = "card_transform_params.json"
    pipeline_age1.process_folder(save_params_to=params_file)
    
    # STEP 2: Process the other folders using the same transformation parameters
    other_folders = ["age_2", "age_3", "guilds"]
    
    for folder in other_folders:
        print(f"\n=== Processing {folder} folder using age_1 parameters ===")
        pipeline = CardSegmentationPipeline(
            input_folder=f'images/{folder}',
            output_folder=f"images/processed_cards/{folder}",
            transform_params_file=params_file  # Use the saved parameters
        )
        pipeline.process_folder()
    
    print("\n=== All folders processed! ===")
    print("All cards now have identical dimensions and masking based on age_1 reference.")

def wonder_processing():
    print("=== Processing wonder folder (master reference) ===")
    pipeline_wonders = CardSegmentationPipeline(
        input_folder="images/wonders",
        output_folder="images/processed_cards/wonders",
        target_width=560,
        target_height=400
    )
    pipeline_wonders.process_folder()

if __name__=='__main__':
    wonder_processing()