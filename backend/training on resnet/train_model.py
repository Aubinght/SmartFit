import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time


# train dataset
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
        #transforms.RandomHorizontalFlip(),  # Data augmentation : not much improvement, takes longer
        #transforms.RandomRotation(10),
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
    print(f"Classes ({num_classes}) : {class_names}")

    print("Uploading Pretrained Resnet18")    
    # upload model with weights from ImageNet
    model = models.resnet18(weights='IMAGENET1K_V1')

    # avoid new learning
    for param in model.parameters():
        param.requires_grad = False

    # replace only last fully connected layer with our categories
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
    nn.Dropout(0.3), #added dropout
    nn.Linear(num_ftrs, num_classes)
)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimize and only update last layer
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    start = time.time()
    for epoch in range(NUM_EPOCHS):
        if epoch == 5:
            for param in model.layer4.parameters():
                param.requires_grad = True
            optimizer = optim.Adam([
                {'params': model.fc.parameters(), 'lr': LEARNING_RATE},
                {'params': model.layer4.parameters(), 'lr': LEARNING_RATE * 0.1}
            ], weight_decay=1e-4)

        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)
        model.train()
        batch_loss = 0.0
        batch_corrects = 0
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

            batch_loss += loss.item() * inputs.size(0)
            batch_corrects += torch.sum(preds == labels.data)

        epoch_loss = batch_loss / dataset_size
        epoch_acc = batch_corrects.float() / dataset_size

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - start
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

#---- We get the following results for the last english model:
'''
data augmentation and full_size pictures
Device used : mps
Dataset size : 3773 images
Classes (12) : ['blazer', 'dress', 'hat', 'hoodie', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']
Uploading Pretrained Resnet18
Epoch 1/10
----------
Loss: 1.6115 Acc: 0.5030
Epoch 2/10
----------
Loss: 1.0124 Acc: 0.7021
Epoch 3/10
----------
Loss: 0.8433 Acc: 0.7472
Epoch 4/10
----------
Loss: 0.7567 Acc: 0.7702
Epoch 5/10
----------
Loss: 0.6784 Acc: 0.7927
Epoch 6/10
----------
Loss: 0.6602 Acc: 0.7949
Epoch 7/10
----------
Loss: 0.6169 Acc: 0.8015
Epoch 8/10
----------
Loss: 0.6049 Acc: 0.8084
Epoch 9/10
----------
Loss: 0.5767 Acc: 0.8214
Epoch 10/10
----------
Loss: 0.5637 Acc: 0.8187
Training lasted 19m 29s
Final accuracy: 0.8187




Without kids :
without data augmentation :
Start training
Device used : mps
Dataset size : 3495 images
Classes (12) : ['blazer', 'dress', 'hat', 'hoodie', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']
Uploading Pretrained Resnet18
Epoch 1/10
----------
Loss: 1.6118 Acc: 0.4961
Epoch 2/10
----------
Loss: 1.0201 Acc: 0.6961
Epoch 3/10
----------
Loss: 0.8288 Acc: 0.7508
Epoch 4/10
----------
Loss: 0.7364 Acc: 0.7768
Epoch 5/10
----------
Loss: 0.6687 Acc: 0.7963
Epoch 6/10
----------
Loss: 0.6367 Acc: 0.8049
Epoch 7/10
----------
Loss: 0.5943 Acc: 0.8106
Epoch 8/10
----------
Loss: 0.5643 Acc: 0.8292
Epoch 9/10
----------
Loss: 0.5445 Acc: 0.8295
Epoch 10/10
----------
Loss: 0.5043 Acc: 0.8378
Training lasted 2m 23s
Final accuracy: 0.8378
val :  Accuracy : 76.89%

with data augmentation: 
Training lasted 2m 7s
Final accuracy: 0.8306
val  Accuracy : 76.66%



--------
No kids, reduced size :
layer_4 freed
With Dropout 0.3
val  Accuracy on reduced size : 86.04%
val Accuracy on full size : 93.86%

Dataset size : 3495 images
Classes (12) : ['blazer', 'dress', 'hat', 'hoodie', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']
Uploading Pretrained Resnet18
Epoch 1/10
----------
Loss: 1.7474 Acc: 0.4429
Epoch 2/10
----------
Loss: 1.1631 Acc: 0.6403
Epoch 3/10
----------
Loss: 0.9994 Acc: 0.6870
Epoch 4/10
----------
Loss: 0.8944 Acc: 0.7082
Epoch 5/10
----------
Loss: 0.8549 Acc: 0.7219
Epoch 6/10
----------
Loss: 0.6056 Acc: 0.7991
Epoch 7/10
----------
Loss: 0.2012 Acc: 0.9416
Epoch 8/10
----------
Loss: 0.0851 Acc: 0.9765
Epoch 9/10
----------
Loss: 0.0434 Acc: 0.9911
Epoch 10/10
----------
Loss: 0.0232 Acc: 0.9948
Training lasted 2m 28s
Final accuracy: 0.9948
'''