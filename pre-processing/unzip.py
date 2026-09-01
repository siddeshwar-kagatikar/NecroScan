import zipfile
import os

# Define file paths
zip_path = "0109_GBM_019_04_IHC_CD34.zip"      
output_folder = "0109_GBM_019_04_IHC_CD34"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
        print(f"Successfully extracted to: {output_folder}")
except FileNotFoundError:
    print(f"Error: The file {zip_path} was not found.")