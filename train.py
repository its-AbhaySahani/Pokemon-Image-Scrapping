# Install dependencies if not already installed
# !pip install torch torchvision transformers

import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up data directories
data_dir = 'E:/DataSet'  # updated to the actual path to your training folder

# Image transformations for ViT input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load dataset
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Number of classes (unique Pokémon)
num_classes = len(train_dataset.classes)

# Load ViT model and specify the number of output classes
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print epoch details
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Save the model
torch.save(model.state_dict(), 'vit_pokemon_detector.pth')
print("Model training complete and saved!")

# # Example for making predictions
# def predict(image_path):
#     model.eval()
#     image = Image.open(image_path)
#     image = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = model(image).logits
#     _, predicted = torch.max(outputs, 1)
#     label = train_dataset.classes[predicted]
#     return label

# # Predict on a sample image
# sample_image_path = 'path/to/sample_image.jpg'  # replace with the path to a test image
# predicted_label = predict(sample_image_path)
# print(f'Predicted Pokémon: {predicted_label}')
