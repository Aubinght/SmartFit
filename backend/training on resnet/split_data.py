import os
import shutil
import random

SOURCE_DIR = "data/base"
OUTPUT_DIR = "data_split"

TRAIN_RATIO = 0.80 
VAL_RATIO   = 0.10 
TEST_RATIO  = 0.10

def split_dataset():
    print(f"Data spliting")

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(OUTPUT_DIR, split)
        if os.path.exists(split_path):
            shutil.rmtree(split_path)
        os.makedirs(split_path)

    categories = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    total_images = 0
    for category in categories:
        src_cat_path = os.path.join(SOURCE_DIR, category)
        images = [f for f in os.listdir(src_cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        n = len(images)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        
        train_imgs = images[:train_end]
        val_imgs   = images[train_end:val_end]
        test_imgs  = images[val_end:]
        
        splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }

        print(f"Traitement de '{category}': {n} images -> Train:{len(train_imgs)}, Val:{len(val_imgs)}, Test:{len(test_imgs)}")

        # Files copy
        for split_name, img_list in splits.items():
            dest_cat_path = os.path.join(OUTPUT_DIR, split_name, category)
            os.makedirs(dest_cat_path, exist_ok=True)
            
            for img in img_list:
                shutil.copy2(
                    os.path.join(src_cat_path, img),
                    os.path.join(dest_cat_path, img)
                )
                total_images += 1

    print(f"Done ! {total_images} images in '{OUTPUT_DIR}/'.")
    
if __name__ == "__main__":
    split_dataset()