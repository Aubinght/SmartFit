import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import numpy as np

DATA_DIR = "data_split/val"
MODEL_PATH = "backend/finetuned_model.pth"
CLASSES_PATH = "backend/classes.txt"
BATCH_SIZE = 32


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Test on : {device}")

    # Transform, as in training
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(DATA_DIR, data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class_names = test_dataset.classes
    print(f"Classes : {class_names}")

    # Upload original model
    model = models.resnet18(weights=None) #we will put our own weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    # Upload retrained model
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Model not found")
        return

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    correct_predictions = np.sum(np.array(all_preds) == np.array(all_labels))
    accuracy = correct_predictions / len(all_labels)
    print(f"\n Accuracy : {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate()