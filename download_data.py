# download_data.py
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import config

def download_and_extract_kaggle_dataset():
    """
    Downloads the specified Kaggle dataset and extracts it.
    Requires kaggle.json to be set up.
    """
    print("Initializing Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    dataset_id = config.KAGGLE_DATASET_ID
    download_path = config.KAGGLE_DOWNLOAD_PATH

    # Create download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)

    # Check if dataset already seems to be downloaded and extracted
    # This check is basic; more robust checks might be needed depending on dataset structure
    if os.path.exists(config.RAW_DATA_SOURCE_DIR) and \
       os.path.exists(os.path.join(config.RAW_DATA_SOURCE_DIR, "Normal")) and \
       os.path.exists(os.path.join(config.RAW_DATA_SOURCE_DIR, "Tuberculosis")):
        print(f"Dataset already appears to be downloaded and extracted at '{config.RAW_DATA_SOURCE_DIR}'. Skipping download.")
        return

    print(f"Downloading dataset '{dataset_id}' to '{download_path}'...")
    try:
        api.dataset_download_files(dataset_id, path=download_path, unzip=False, quiet=False)
        print("Download complete.")
    except Exception as e:
        print(f"Error during Kaggle API download: {e}")
        print("Please ensure your Kaggle API token (kaggle.json) is correctly set up.")
        print("See: https://www.kaggle.com/docs/api#authentication")
        return

    # Find the downloaded zip file (Kaggle dataset names can be unpredictable)
    downloaded_files = os.listdir(download_path)
    zip_file_name = None
    for f_name in downloaded_files:
        if f_name.endswith('.zip'):
            zip_file_name = f_name
            break
    
    if not zip_file_name:
        print(f"No zip file found in {download_path} after download. Please check the Kaggle dataset page.")
        return

    zip_file_path = os.path.join(download_path, zip_file_name)

    print(f"Extracting '{zip_file_path}'...")
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        print(f"Extraction complete. Dataset should be in '{download_path}'.")
        # Optionally, remove the zip file after extraction
        # os.remove(zip_file_path)
        # print(f"Removed zip file: {zip_file_path}")
    except zipfile.BadZipFile:
        print(f"Error: Downloaded file '{zip_file_path}' is not a valid zip file or is corrupted.")
    except Exception as e:
        print(f"Error during extraction: {e}")

if __name__ == "__main__":
    download_and_extract_kaggle_dataset()