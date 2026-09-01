import os
import cv2
import numpy as np
import slideio

FILE_PATH = os.path.join('0109_GBM_019_04_IHC_CD34/0109_GBM_019_04_IHC_CD34/0109.vsi')
OUTPUT_DIR = os.path.join('BGM3')
PATCH_SIZE = 256
OVERLAP = 0

def create_patches():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    # 1. Open the slide using the 'VSI' driver
    print(f"--- Opening {os.path.basename(FILE_PATH)} with SlideIO ---")
    try:
        slide = slideio.open_slide(FILE_PATH, "VSI")
    except Exception as e:
        print(f"Error opening slide: {e}")
        return

    # 2. Find the High-Resolution Scene
    num_scenes = slide.num_scenes
    print(f"Found {num_scenes} scenes.")

    best_scene = None
    max_pixels = 0
    best_index = -1

    for i in range(num_scenes):
        scene = slide.get_scene(i)
        rect = scene.rect # (x, y, width, height)
        width, height = rect[2], rect[3]
        pixels = width * height
        
        print(f"  Scene {i}: {width} x {height} pixels - Name: {scene.name}")
        
        if pixels > max_pixels:
            max_pixels = pixels
            best_scene = scene
            best_index = i

    if not best_scene:
        print("Could not find a valid scene.")
        return

    print(f"\n>>> Selected Scene {best_index} (Resolution: {best_scene.rect[2]}x{best_scene.rect[3]}) <<<")
    
    # 3. Generate Patches
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    scene_rect = best_scene.rect
    img_w = scene_rect[2]
    img_h = scene_rect[3]
    
    print(f"Starting patch generation ({PATCH_SIZE}x{PATCH_SIZE})...")
    
    count = 0
    step = PATCH_SIZE - OVERLAP
    
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            w = min(PATCH_SIZE, img_w - x)
            h = min(PATCH_SIZE, img_h - y)
            
            image_block = best_scene.read_block(rect=(x, y, w, h), size=(w, h))

            if w != PATCH_SIZE or h != PATCH_SIZE:
                continue
            if np.mean(image_block) > 230:
                continue

            patch_name = f"patch_y{y}_x{x}.png"
            save_path = os.path.join(OUTPUT_DIR, patch_name)
            
            image_block = cv2.cvtColor(image_block, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(save_path, image_block)
            count += 1
            
            if count % 50 == 0:
                print(f"Generated {count} patches...", end='\r')

    print(f"\nDone! Saved {count} patches to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    create_patches()