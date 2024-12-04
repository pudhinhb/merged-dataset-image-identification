import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os

# Load the pre-trained ResNet model
model = models.resnet50(weights='IMAGENET1K_V1')
model.eval()  # Set the model to evaluation mode

# Define the ImageNet class names (from a file or predefined list)
imagenet_class_names = []
with open('imagenet1000_clsidx_to_labels.txt', 'r') as f:
    imagenet_class_names = [line.strip() for line in f.readlines()]

# Define your custom class names (e.g., ant and bee)
custom_class_names = ['ant', 'bee']  # Update based on your dataset

# Combine ImageNet and custom class names
combined_class_names = imagenet_class_names + custom_class_names

# Load Custom Dataset Using ImageFolder
train_dir = 'hymenoptera_data/hymenoptera_data/train'  # Update with the correct path to your custom dataset
val_dir = 'hymenoptera_data/hymenoptera_data/val'      # Update with the correct path to your validation dataset

# Define the image transformation pipeline for inference and training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the training and validation datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# Create DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Print the custom dataset class names
print(f"Custom Class Names: {train_dataset.classes}")

# Modify the ResNet model's final fully connected layer to match the new class count
num_classes = len(imagenet_class_names) + len(custom_class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer

# Unfreeze layers for fine-tuning
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Set up loss function and optimizer for fine-tuning
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Train the model on both the custom dataset and ImageNet classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()

    accuracy = correct_preds / total_preds
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy*100:.2f}%")

# Save the fine-tuned model (optional)
torch.save(model.state_dict(), 'fine_tuned_resnet50.pth')

# Function to predict an image with combined classes (ImageNet + Custom dataset)
def predict_image(image_path, model, transform, class_names, top_k=5):
    model.eval()  # Set the model to evaluation mode
    image = Image.open(image_path).convert("RGB")  # Open the image
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension

    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)  # Forward pass
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Get probabilities
        top_probs, top_indices = torch.topk(probabilities, top_k)  # Get top K predictions

    # Map indices to class names
    top_probs = top_probs.cpu().numpy().flatten()
    top_indices = top_indices.cpu().numpy().flatten()

    top_classes = [class_names[idx] for idx in top_indices]
    return list(zip(top_classes, top_probs))

# Test the prediction with a sample image (Update with your image path)
test_image_path = 'dog.jpeg'  # Replace with the correct image path
predictions = predict_image(test_image_path, model, transform, combined_class_names)

# Output top predictions
print("Top Predictions:")
for i, (cls, prob) in enumerate(predictions):
    print(f"Rank {i+1}: {cls}, Probability: {prob:.2f}")
