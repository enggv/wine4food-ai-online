import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm

# --- 1. Трансформации ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 2. Зареждане на пълния Food-101 dataset чрез ImageFolder ---
from torchvision.datasets import ImageFolder

dataset_root = './data/food-101/images'
train_dataset = ImageFolder(root=dataset_root, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# --- 3. Визуализация без блокиране ---
def show_sample():
    raw_transform = transforms.Compose([transforms.Resize((224, 224))])
    sample_dataset = ImageFolder(root=dataset_root, transform=raw_transform)
    img, label = sample_dataset[0]
    plt.imshow(img)
    plt.title(f"Label: {train_dataset.classes[label]}")
    plt.axis('off')
    plt.savefig("sample_preview.png")
    print("✔️ Преглед на първото изображение записан като sample_preview.png")

show_sample()

# --- 4. Модел + GPU настройка ---
num_classes = len(train_dataset.classes)
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 5. Обучение ---
epochs = 10  # можеш да го увеличиш или намалиш по преценка

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"✅ Epoch {epoch+1}: Loss = {running_loss:.4f}, Accuracy = {acc:.2f}%")

# --- 6. Запис на модела ---
os.makedirs("../models", exist_ok=True)
model_path = "../models/food_model_101classes.pt"
torch.save(model.state_dict(), model_path)
print(f"💾 Моделът е записан успешно като {model_path}")
