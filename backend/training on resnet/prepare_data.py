import pandas as pd
import os
import shutil

# dowload files from Kaggle
SOURCE_CSV = "temp_download/images.csv"
SOURCE_IMAGES_DIR = "temp_download/images_original"

# train_data path
TARGET_DIR = "data/train"

# Translate categories from Kaggle to our Categories
# and ignores categories : "Other", "Not sure", "Skip"
CATEGORY_MAPPING = {
    "t-shirt": "T-shirt",
    "pants": "Pantalon", 
    "shoes": "Chaussures",
    "shirt": "Chemise",
    "dress": "Robe",
    "outwear": "Manteau",
    "shorts": "Short",
    "skirt": "Jupe",
    "hat": "Chapeau",
    "hoodie": "Pull",  
    "longsleeve": "Pull",
    "blazer": "Veste"
}

def prepare_dataset():
    print("Start Data filter")

    if not os.path.exists(SOURCE_CSV):
        print(f"Error : File {SOURCE_CSV} inexistant.")
        return

    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    # Create subfile for each category we want
    for french_category in set(CATEGORY_MAPPING.values()):
        os.makedirs(os.path.join(TARGET_DIR, french_category), exist_ok=True)

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
        
        # Only take interesting labels
        if label_eng in CATEGORY_MAPPING:
            french_label = CATEGORY_MAPPING[label_eng]
            
            src_path = os.path.join(SOURCE_IMAGES_DIR, image_id + ".jpg")
            dst_path = os.path.join(TARGET_DIR, french_label, image_id + ".jpg")
            
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

        # Charging bar
        if index % 500 == 0:
            print(f"Traitement... {index} images vues")

    print("\n--- Bilan ---")
    print(f"Images triées et copiées : {count_success}")
    print(f"Images ignorées (catégories inutiles) : {count_ignored}")
    print(f"Images manquantes (dans CSV mais pas dossier) : {count_missing}")
    print(f"Données prêtes dans : {TARGET_DIR}")

if __name__ == "__main__":
    prepare_dataset()