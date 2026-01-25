import pandas as pd
import os
import shutil

# dowloaded files from Kaggle
SOURCE_CSV = "temp_download/images.csv"
SOURCE_IMAGES_DIR = "temp_download/images_compressed"

# train_data path
TARGET_DIR = "data/train"

# ignores categories : "Other", "Not sure", "Skip"
CATEGORIES = [
    "t-shirt",
    "pants", 
    "shoes",
    "shirt",
    "dress",
    "outwear",
    "shorts",
    "skirt",
    "hat",
    "hoodie",
    "longsleeve",
    "blazer",
]

def prepare_dataset():
    print("Start Data filter")

    if not os.path.exists(SOURCE_CSV):
        print(f"Error : File {SOURCE_CSV} inexistant.")
        return

    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    # Create subfile for each category we want
    for eng_category in CATEGORIES:
        os.makedirs(os.path.join(TARGET_DIR, eng_category), exist_ok=True)

    # Open CSV Data
    df = pd.read_csv(SOURCE_CSV)
    print(f" Number of lines in csv : {len(df)}")

    count_success = 0
    count_ignored = 0
    count_missing = 0

    # Loop on each image
    for index, row in df.iterrows():
        image_id = row['image']  # filepath
        label_eng = row['label'].lower() # label
        kids = row['kids']
        
        # Only take interesting labels
        if label_eng in CATEGORIES and not kids:
            
            src_path = os.path.join(SOURCE_IMAGES_DIR, image_id + ".jpg")
            dst_path = os.path.join(TARGET_DIR, label_eng, image_id + ".jpg")
            
            # Copy image in distant path from source path
            if os.path.exists(src_path):
                try:
                    shutil.copy(src_path, dst_path)
                    count_success += 1
                except Exception as e:
                    print(f"Copy error : {e}")
            else:
                # missing image from the source
                count_missing += 1
        else:
            count_ignored += 1

    print("\n--- Summary ---")
    print(f"Images treated: {count_success}")
    print(f"Images ignored : {count_ignored}")
    print(f"Missing images (in CSV but not in folder) : {count_missing}")
    print(f"Ready images in : {TARGET_DIR}")

if __name__ == "__main__":
    prepare_dataset()