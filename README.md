# SmartFit - AI-Powered Wardrobe Management System

## Overview

SmartFit is the start to an intelligent wardrobe management application that leverages deep learning and computer vision to help users organize, catalog, and discover clothing combinations. The system uses a fine-tuned ResNet18 neural network to automatically classify clothing items and employs K-means clustering for color detection, enabling smart recommendations based on color matching.

## Key Features

### 1. **User Authentication**
   - Secure user registration and login system
   - Password hashing using Werkzeug security utilities
   - Session management with Flask-Login for persistent authentication

### 2. **Clothing Detection & Classification**
   - Automatic clothing type recognition using a fine-tuned ResNet18 deep learning model
   - Supports 12 clothing categories: T-shirt, Blazer, Dress, Hat, Hoodie, Longsleeve, Outwear, Pants, Shirt, Shoes, Shorts, Skirt
   suggestions when confidence is below 30%
   - Model trained on English dataset with compression for efficient re-training
   - Put in pause : Confidence-based predictions with fallback to top-3 

### 3. **Color Detection**
   - K-means clustering algorithm for dominant color extraction
   - Intelligent color mapping to 13 predefined colors: Blue, Red, Orange, Yellow, Green, Purple, Pink, White, Black, Grey, Beige, Navy, and Multicolor
   - Crop and center-focused analysis to identify primary clothing colors while ignoring background

### 4. **Smart Wardrobe Management**
   - Upload and catalog clothing items with automatic metadata (color and category) extraction
   - View full wardrobe or filter by clothing category
   - Persistent storage of clothing items linked to individual user accounts
   - Delete items from collection with automatic image cleanup

### 5. **Outfit Recommendations**
   - Color-based outfit matching engine
   - Recommends all different clothing categories that match the color of selected item

## Technology Stack

- **Backend Framework**: Flask (Python web framework)
- **Database**: SQLAlchemy ORM with SQLite (`macollection.db`)
- **Deep Learning**: PyTorch with ResNet18 architecture
- **Image Processing**: PIL (Python Imaging Library), scikit-learn (K-means clustering)
- **Authentication**: Flask-Login with password hashing
- **Frontend**: HTML/Jinja2 templates

## Project Structure

```
SmartFit/
├── app.py                           # Main Flask application
├── README.md                        # Project documentation
├── backend/
│   ├── database.py                  # SQLAlchemy models (User, Clothing)
│   ├── detect_eng.py                # Clothing & color detection pipeline
│   ├── predict_eng_model.py         # ResNet18 inference with model loading
│   ├── recommender.py               # Clothing recommendation engine
│   ├── random_model.py              # Baseline random prediction model
│   ├── classes_eng.txt              # Clothing class labels
│   ├── finetuned_model_eng_compressed_images_freed_layer_4.pth  # Trained model weights
│   └── training on resnet/          # Training scripts and utilities
│       ├── train_model.py           # Model training pipeline
│       ├── evaluate.py              # Model evaluation metrics
│       ├── split_data.py            # Dataset splitting utilities
│       ├── data_cleaning.py         # Data preprocessing (English)
│       └── data_cleaning_french.py  # Data preprocessing (French)
│   └── abandonned_training/         # Previous model versions and experiments
├── static/
│   ├── uploads/                     # User uploaded clothing images
│   ├── images/                      # Static assets
│   └── wardrobe_results.json        # (1st Experiment) Wardrobe file
├── templates/                       # HTML templates
│   ├── homepage.html                # Landing page
│   ├── login.html                   # User login form
│   ├── signup.html                  # User registration form
│   ├── upload.html                  # Clothing upload & confirmation
│   ├── wardrobe.html                # Wardrobe display & management
│   ├── recommendation.html          # Outfit recommendations
│   ├── logout.html                  # Logout confirmation
│   ├── template.html                # Base template
│   └── index.html                   # index page
└── instance/                        # Database folder

```

## Core Modules

### `app.py` - Main Application
Handles all Flask routes and business logic.

### `backend/database.py` - Data Models
Defines two main SQLAlchemy models:
- **User**: Stores user credentials (username, hashed password) and has a relationship to clothing items
- **Clothing**: Represents wardrobe items with fields for image path, category, color, and user association

### `backend/detect_eng.py` - Detection Pipeline
Two-stage detection system:
1. **Clothing Classification**: `detect_clothing(image_path)` → Uses ResNet18 model
2. **Color Detection**: `detect_color(image_path)` → Uses K-means clustering with LAB color space transformation

### `backend/predict_eng_model.py` - Model Inference
ResNet18 inference engine that:
- Loads pre-trained weights from `.pth` file
- Applies image preprocessing (resize to 224×224, normalization)
- Returns single prediction or top-3 predictions based on confidence threshold (30%)

### `backend/recommender.py` - Recommendation Engine
Implements color-based matching:
- Queries wardrobe for items matching a reference item's color
- Filters out duplicate categories
- Supports random shuffle

## Workflow

1. **User Registration** → Creates account in SQLite database
2. **Image Upload** → User selects clothing photo
3. **AI Detection** → ResNet18 classifies type, K-means extracts color
4. **User Confirmation** → User reviews/corrects AI predictions
5. **Storage** → Item saved to database with metadata
6. **Recommendation** → User can browse recommendations based on color matches

## Key Algorithm Details

### Clothing Classification
- **Model**: ResNet18 with custom head (Dropout + Linear layer)
- **Input**: Images resized to 224×224 pixels, normalized with ImageNet stats
- **Output**: Probability distribution over 12 clothing categories
- **Confidence Mechanism**: Returns single prediction if >30% confidence, otherwise top-3 alternatives

### Color Detection
- **Algorithm**: K-means clustering (k=3) on downsampled image
- **Region Focus**: Crops image to central 70% to avoid background
- **Center Weight**: Prioritizes pixel distribution in image center (20-40 pixel zone)
- **Multicolor Detection**: If >2 significant clusters (>30% of center pixels), classified as "Multicolor"
- **Color Mapping**: Uses approximate LAB color space transformation for accurate perceptual color matching

## Installation & Setup

```bash
# Clone repository
git clone <repo_url>
cd SmartFit

# Install dependencies
pip install flask flask-sqlalchemy flask-login werkzeug pillow torch torchvision scikit-learn

# Initialize database
python app.py

# Run application
python app.py
```

The application runs on `http://localhost:5000`

## Possible Future Enhancements

- Advanced outfit recommandation (beyond color matching)
- Seasonal recommendations
- Wardrobe analytics and statistics
- Mobile application
- Real-time camera integration for on-the-fly classification