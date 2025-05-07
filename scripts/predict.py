import os
import torch
import timm
from PIL import Image
from torchvision import transforms
from torchvision.datasets import Food101

# --- 1. Зареждане на класовете ---
from torchvision.datasets import Food101
dummy_dataset = Food101(root='./data', split='train', download=False)
class_names = dummy_dataset.classes

# --- 2. Модел и устройство ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(class_names)
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load("../models/food_model_101classes.pt", map_location=device))
model.to(device)
model.eval()

# --- 3. Трансформации ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 4. Функция за предсказание ---
def predict_image(image_path):
    assert os.path.exists(image_path), f"❌ Файлът {image_path} не съществува."
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return class_names[predicted_class]
