import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time


# train data set
DATA_DIR = "data_split/train" 

# Path to save trained model
SAVE_DIR = "backend"
MODEL_FILENAME = "finetuned_model.pth"
CLASSES_FILENAME = "classes.txt"

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10     
LEARNING_RATE = 0.001

def train():
    print(f"Start training")

    # detect device depending on processor
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device used : {device}")

    # Data transormation for Resnet compatibility
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),      #Resnet input
        #transforms.RandomHorizontalFlip(),  # Data augmentation : useless, only  76.91% on val for Training lasted 19m 54s, against 75.42%
        transforms.ToTensor(),              
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet Standardisation
    ])

    # Upload (homemade) dataset
    image_dataset = datasets.ImageFolder(DATA_DIR, data_transforms)
    
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    class_names = image_dataset.classes
    num_classes = len(class_names)
    dataset_size = len(image_dataset)

    print(f"Dataset size : {dataset_size} images")
    print(f"Classes detected ({num_classes}) : {class_names}")

    print("Uploading Pretrained Resnet18")    
    # upload model with weights from ImageNet
    model = models.resnet18(weights='IMAGENET1K_V1')

    # avoid new learning
    for param in model.parameters():
        param.requires_grad = False

    # replace only fully connected layer with our categories
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimize and only update last layer
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    since = time.time()
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # Forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # Backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.float() / dataset_size

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - since
    print(f'Training lasted {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Final accuracy: {epoch_acc:.4f}')

    #Save model
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    save_path = os.path.join(SAVE_DIR, MODEL_FILENAME)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved in : {save_path}")

    # Save IDs with Classes name
    classes_path = os.path.join(SAVE_DIR, CLASSES_FILENAME)
    with open(classes_path, "w") as f:
        for c in class_names:
            f.write(f"{c}\n")
    print(f"Classes saved in : {classes_path}")

if __name__ == "__main__":
    train()