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


import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans

COLOR_MAP = {
    "Blue": (0, 70, 170),
    "Red": (200, 0, 0),
    "Orange": (255, 140, 0),
    "Yellow": (255, 230, 0),
    "Green": (0, 100, 0),
    "Purple": (100, 0, 100),
    "Pink": (255, 150, 180),
    "White": (245, 245, 245),
    "Black": (20, 20, 20),
    "Grey": (120, 120, 120),
    "Beige": (225, 200, 170),
    "Navy": (0, 0, 80)
}

def detect_color_k_means(image_path):
    img = Image.open(image_path).convert('RGB')
    #width, height = img.size
    #img_cropped = img.crop((width*0.15, height*0.15, width*0.85, height*0.85))
    img_small = img.resize((60, 60))
    pixels = np.array(img_small).reshape(-1, 3)

    # Division in 3 clusters, possibly 3 parts of the image : back, cloth, shadow
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    # identify the cluster with the most pixels in the 20 center pixels
    center_mask = np.zeros((60, 60), dtype=bool)
    center_mask[20:40, 20:40] = True
    center_labels = labels.reshape(60, 60)[center_mask]
    
    # winning cluster is the closest to the center pixels
    counts = np.bincount(center_labels, minlength=3)
    dominant_cluster_idx = np.argmax(counts)
    
    dominant_rgb = centers[dominant_cluster_idx]

    # simplified LAB conversion
    def transform(rgb):
        r, g, b = rgb / 255.0
        return np.array([0.299*r + 0.587*g + 0.114*b, r - g, g - b])

    target_lab = transform(dominant_rgb)
    names = list(COLOR_MAP.keys())
    ref_labs = np.array([transform(np.array(COLOR_MAP[n])) for n in names])

    distances = np.linalg.norm(ref_labs - target_lab, axis=1)
    return names[np.argmin(distances)]

def detect_color(image_path):
    """
    function that takes an image as an input and returns the predicted color
    """
    # Chose random color
    #category = random.choice(COLORS)
    category = detect_color_k_means(image_path)
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