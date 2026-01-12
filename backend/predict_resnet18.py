import torch
from torchvision import models, transforms
from PIL import Image
import os

# Parameters
IMG_PATH = "static/uploads/cloth_20251110_171257_713120.jpg"
MODEL_PATH = "backend/finetuned_model_80_24112025.pth"
CLASSES_PATH = "backend/classes_80_24112025.txt"

def predict_single_image(img_path,model_path = MODEL_PATH, classes_path = CLASSES_PATH):
    # load classes
    with open(classes_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Load model
    device = torch.device("cpu") # enough for one image
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # Add dimension batch (1, 3, 224, 224)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        #predict
        conf, predicted_idx = torch.max(probabilities, 0)
        print(f"Image : {IMG_PATH}")
        predicted_label = class_names[predicted_idx]

        # Get more detailed prediction in case it is less thant 50% confidence
        if conf.item() < 1:
            # Low confidence : Get 3 best predictions
            top_probs, top_idxs = torch.topk(probabilities, 3)
            
            top_3_results = []
            for i in range(3):
                idx = top_idxs[i].item()
                prob = top_probs[i].item()
                label = class_names[idx]
                top_3_results.append((label, prob))
                
            return(top_3_results) 
        else:
            predicted_label = class_names[predicted_idx]
            print(f"Prédiction : {predicted_label}")
            print(f"Confiance : {conf.item() * 100:.2f}%")
            return(predicted_label)

if __name__ == "__main__":
    predict_single_image(IMG_PATH,MODEL_PATH,CLASSES_PATH)