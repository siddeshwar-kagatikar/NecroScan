import os
import json
import numpy as np
import cv2
import pandas as pd
from shapely.geometry import shape, box
from shapely.ops import unary_union
from tqdm import tqdm

PATCHES_DIR = os.path.join('BGM2') 
OUTPUT_CSV = 'GBM_labels.csv'
PATCH_SIZE = 256

GEOJSON_FILES = [
    '0108_GBM_019_03_H&E/0108_GBM_019_03_H&E/0108.vsi - 20x_BF_01.geojson'
]

TARGET_CLASS_NAME = "Necrosis" 

TISSUE_THRESHOLD = 0.50
NECROSIS_POS_THRESH = 0.30
NECROSIS_NEG_THRESH = 0.05

def estimate_tissue_percentage(image_path):
    img = cv2.imread(image_path) 
    if img is None: return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    tissue_mask = saturation > 25
    return np.sum(tissue_mask) / (img.shape[0] * img.shape[1])

def generate_ground_truth():
    # 1. Load and Merge GeoJSONs
    all_polygons = []
    print(f"--- Loading {len(GEOJSON_FILES)} GeoJSON files ---")
    
    for json_path in GEOJSON_FILES:
        if not os.path.exists(json_path):
            print(f"Warning: File {json_path} not found. Skipping.")
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        for feature in data['features']:
            try:
                if feature['properties']['classification']['name'] == TARGET_CLASS_NAME:
                    poly = shape(feature['geometry'])
                    if not poly.is_valid: poly = poly.buffer(0)
                    all_polygons.append(poly)
            except KeyError:
                continue
    
    if not all_polygons:
        print("Error: No polygons found.")
        return

    print("Merging polygons...")
    master_necrosis_shape = unary_union(all_polygons)

    # 2. Iterate Patches
    patch_data = []
    if not os.path.exists(PATCHES_DIR):
        print(f"Error: Directory {PATCHES_DIR} not found.")
        return
        
    files = sorted([f for f in os.listdir(PATCHES_DIR) if f.endswith(('.png', '.jpg'))])
    print(f"Scanning {len(files)} patches...")

    count_background = 0
    count_ambiguous = 0
    count_normal = 0
    count_necrosis = 0

    for fname in tqdm(files, desc="Labeling"):
        try:
            name_no_ext = os.path.splitext(fname)[0]
            parts = name_no_ext.split('_')
            y_start = int(next(p for p in parts if p.startswith('y'))[1:])
            x_start = int(next(p for p in parts if p.startswith('x'))[1:])
        except:
            continue

        img_path = os.path.join(PATCHES_DIR, fname)
        tissue_ratio = estimate_tissue_percentage(img_path)
        
        if tissue_ratio < TISSUE_THRESHOLD:
            count_background += 1
            continue 

        # Check Intersection
        patch_box = box(x_start, y_start, x_start + PATCH_SIZE, y_start + PATCH_SIZE)
        intersection_area = 0.0
        
        if patch_box.intersects(master_necrosis_shape):
            intersection_area = patch_box.intersection(master_necrosis_shape).area

        necrosis_ratio = intersection_area / (PATCH_SIZE * PATCH_SIZE)
        
        label = -2 # Default Ambiguous
        
        if necrosis_ratio >= NECROSIS_POS_THRESH:
            label = 1
            count_necrosis += 1
        elif necrosis_ratio <= NECROSIS_NEG_THRESH:
            label = 0
            count_normal += 1
        else:
            # Between 5% and 30%
            count_ambiguous += 1

        # Only save valid training data (0 and 1) to the CSV
        if label in [0, 1]:
            patch_data.append({
                'filename': fname,
                'label': label,
                'tissue_ratio': tissue_ratio,
                'necrosis_ratio': necrosis_ratio
            })

    # 3. Print Final Report
    print("\n" + "="*40)
    print(f"PROCESSING REPORT for {PATCHES_DIR}")
    print("="*40)
    print(f"Total Patches Scanned: {len(files)}")
    print("-" * 30)
    print(f"Background (<50% tissue): {count_background} (Discarded)")
    print(f"Ambiguous (5-30% overlap): {count_ambiguous} (Discarded)")
    print("-" * 30)
    print(f"NORMAL (Label 0):          {count_normal} (Saved)")
    print(f"NECROSIS (Label 1):        {count_necrosis} (Saved)")
    print("="*40)

    # 4. Save
    if len(patch_data) > 0:
        df = pd.DataFrame(patch_data)
        file_exists = os.path.isfile(OUTPUT_CSV)
        df.to_csv(OUTPUT_CSV, mode='a', header=not file_exists, index=False)
        print(f"\nSuccessfully appended {len(df)} rows to {OUTPUT_CSV}")
    else:
        print("\nNo valid patches found to save.")

if __name__ == "__main__":
    generate_ground_truth()