import os
import zipfile
import urllib.request
import shutil

ZIP_URL = "https://github.com/mwt5345/DeepLenseSim/archive/refs/heads/main.zip"
ZIP_NAME = "DeepLenseSim.zip"
EXTRACTED_DIR = "DeepLenseSim-main"
TARGET_DIR = "Model_I"

def setup_data():
    if os.path.exists(TARGET_DIR):
        print("[INFO] Model_I dataset already exists.")
        return

    print("[INFO] Downloading DeepLenseSim dataset...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_NAME)

    print("[INFO] Extracting archive...")
    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall()

    src = os.path.join(EXTRACTED_DIR, "Model_I")
    if not os.path.exists(src):
        raise RuntimeError("Model_I not found in DeepLenseSim archive")

    shutil.move(src, TARGET_DIR)

    # Cleanup
    shutil.rmtree(EXTRACTED_DIR)
    os.remove(ZIP_NAME)

    print("[SUCCESS] Model_I dataset setup complete.")

if __name__ == "__main__":
    setup_data()
