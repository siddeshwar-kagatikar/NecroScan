import os
import cv2
import numpy as np
import torch
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from cellpose import models
from tqdm import tqdm

INPUT_ROOT_DIR = os.path.join('BGM3')
OUTPUT_DIR = os.path.join('BGM3_masks')

GPU_BATCH_SIZE = 512 
MACRO_BATCH_SIZE = 2048 
READ_THREADS = 16 

MODEL_TYPE = 'nuclei'
CHANNELS = [0, 0] 
DIAMETER = None 


write_queue = queue.Queue()
stop_writer_event = threading.Event()

def get_output_filename(root_dir, file_path):
    """Generates a unique filename: ParentFolder_ImageName_mask.png"""
    parent_folder = os.path.basename(os.path.dirname(file_path))
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    safe_parent = parent_folder.replace(" ", "_")
    return f"{safe_parent}_{base_name}_mask.png"

def writer_thread_func():
    """Background thread that constantly saves images so the GPU doesn't have to wait."""
    while not stop_writer_event.is_set() or not write_queue.empty():
        try:
            save_path, binary_mask = write_queue.get(timeout=1)
            cv2.imwrite(save_path, binary_mask)
            write_queue.task_done()
        except queue.Empty:
            continue

def read_image(path):
    """Helper for parallel reading."""
    img = cv2.imread(path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None

def run_segmentation():
    # 1. Setup
    if not os.path.exists(INPUT_ROOT_DIR):
        print(f"Error: Input directory '{INPUT_ROOT_DIR}' not found.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check GPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"--- GPU Detected: {gpu_name} (Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB) ---")
        print(f"--- Batch settings: GPU Batch={GPU_BATCH_SIZE}, RAM Buffer={MACRO_BATCH_SIZE} ---")
    else:
        print("--- WARNING: No GPU detected. This will be slow. ---")

    # Initialize Model
    print(f"--- Initializing Cellpose ({MODEL_TYPE}) ---")
    try:
        model = models.Cellpose(gpu=use_gpu, model_type=MODEL_TYPE)
    except AttributeError:
        model = models.CellposeModel(gpu=use_gpu, model_type=MODEL_TYPE)

    # 2. Start Writer Thread
    writer_t = threading.Thread(target=writer_thread_func, daemon=True)
    writer_t.start()

    # 3. Recursively Scan Files
    print(f"Scanning '{INPUT_ROOT_DIR}' for files...")
    files_to_process = []
    
    for root, dirs, files in os.walk(INPUT_ROOT_DIR):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                full_path = os.path.join(root, f)
                out_name = get_output_filename(INPUT_ROOT_DIR, full_path)
                out_path = os.path.join(OUTPUT_DIR, out_name)
                
                if not os.path.exists(out_path):
                    files_to_process.append((full_path, out_path))

    total_files = len(files_to_process)
    if total_files == 0:
        print("All images already processed!")
        stop_writer_event.set()
        return

    print(f"Found {total_files} pending images.")
    print("Starting Optimized Pipeline...")

    # 4. Process in Macro Batches
    for i in tqdm(range(0, total_files, MACRO_BATCH_SIZE), desc="Overall Progress"):
        chunk = files_to_process[i : i + MACRO_BATCH_SIZE]
        chunk_paths = [c[0] for c in chunk]
        chunk_out_paths = [c[1] for c in chunk]

        images = []
        valid_indices = []
        
        with ThreadPoolExecutor(max_workers=READ_THREADS) as executor:
            results = list(executor.map(read_image, chunk_paths))
            
        for idx, img in enumerate(results):
            if img is not None:
                images.append(img)
                valid_indices.append(idx)
        
        if not images:
            continue

        try:
            masks, flows, styles = model.eval(
                images, 
                diameter=DIAMETER, 
                channels=CHANNELS,
                flow_threshold=0.4,
                cellprob_threshold=0.0,
                batch_size=GPU_BATCH_SIZE, 
                progress=False 
            )[:3]
        except RuntimeError as e:
            print(f"\nCRITICAL ERROR: GPU Out of Memory? Try lowering GPU_BATCH_SIZE. Error: {e}")
            break

       
        for k, mask in enumerate(masks):
            original_idx = valid_indices[k]
            save_path = chunk_out_paths[original_idx]

            binary_mask = np.zeros_like(mask, dtype=np.uint8)
            binary_mask[mask > 0] = 255
            
            write_queue.put((save_path, binary_mask))

    print("\nWaiting for writer thread to finish remaining files...")
    write_queue.join() # Wait for queue to empty
    stop_writer_event.set()
    writer_t.join()
    
    print(f"Processing complete! Check: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    run_segmentation()