from datetime import datetime
from backend.database import Clothing
CATEGORIES = [
    "T-shirt",
    "Chemise",
    "Pull",
    "Pantalon",
    "Jean",
    "Jupe",
    "Robe",
    "Veste",
    "Manteau",
    "Chaussures"
]

import random

def detect_clothing(image_path):
    """
    function that takes an image as an input and returns the predicted type of cloth
    """
    # Chose random category
    category = random.choice(CATEGORIES)
    return category

# ==========================
# wardrobe storage
# ==========================
wardrobe = []
wardrobe_dict = {}
json_file = "static/wardrobe_results.json"
import os
import json
import random
UPLOAD_FOLDER = "static/uploads/"

def generate_unique_filename(base_name="cloth", ext=".jpg"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{base_name}_{timestamp}{ext}"
def save_image(img, ext=".jpg"):
    filename = generate_unique_filename(ext=ext)
    path = os.path.join(UPLOAD_FOLDER, filename)
    img.save(path)
    return path

def add_to_wardrobe_json(cloth):
    """Add a new image to wardrobe and update the JSON file"""
    wardrobe.append(cloth)
    wardrobe_dict[cloth.image_path] = cloth.category
    # Load existing JSON if present
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            try:
                wardrobe_data = json.load(f)
            except json.JSONDecodeError:
                wardrobe_data = []
    else:
        wardrobe_data = []
    #add new cloth to local memory
    wardrobe_data.append({
        "image": cloth.image_path,
        "category": cloth.category
    })
    # Save updated JSON
    with open(json_file, "w") as f:
        json.dump(wardrobe_data, f, indent=4)
    print(f"Added to : {cloth.image_path} -> {cloth.category}")
    print(f"JSON updated : {json_file}")

