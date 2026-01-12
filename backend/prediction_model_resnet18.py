from datetime import datetime
from backend.predict_resnet18 import predict_single_image
import random

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

COLORS = [
    "Blue",
    "Red",
    "Orange",
    "Yellow",
    "Green",
    "Purple",
    "Pink"
]

class Clothing:
    def __init__(self, image_path, category = None, color=None):
        self.category = category
        self.image_path = image_path
        self.color = color
    def __repr__(self):
        return f"{self.category}"

def detect_clothing(image_path):
    """
    function that takes an image as an input and returns the predicted type of cloth
    """
    category = predict_single_image(image_path)
    return category

def detect_color(image_path):
    """
    function that takes an image as an input and returns the predicted color
    """
    # Chose random color
    category = random.choice(COLORS)
    return category
# ==========================
# wardrobe storage
# ==========================
wardrobe = []
wardrobe_dict = {}
json_file = "static/wardrobe_results.json"
import os
import json

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

if __name__ == "__main__":
    # Exemple d'utilisation
    image_path = "static/uploads/cloth_20251110_171257_713120.jpg"
    category = detect_clothing(image_path)
    cloth = Clothing(image_path, category)
    add_to_wardrobe_json(cloth)

    print(wardrobe_dict)