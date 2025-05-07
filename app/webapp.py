from flask import Flask, render_template, request, url_for
import os
from PIL import Image
import torch
from torchvision import transforms
import timm
from pairings import pairings  # Импортираме речника с винените съчетания
from huggingface_hub import hf_hub_download

# Flask app
app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Позволени формати
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Предикт функция
def predict_image(image_path, confidence_threshold=0.5):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Класовете са ключовете на речника
    class_names = list(pairings.keys())

    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(class_names))

    model_path = hf_hub_download(repo_id="enggv/food101-effnet-model", filename="food_model_101classes.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        probs = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, predicted_class = torch.max(probs, dim=1)
        confidence = max_prob.item()

    if confidence < confidence_threshold:
        return None

    return class_names[predicted_class.item()]

# Рут
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    wine = None
    image_url = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Изтрий предишни файлове
            for existing_file in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], existing_file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"⚠️ Неуспех при изтриване на {file_path}: {e}")

            # Провери типа на файла
            if not allowed_file(file.filename):
                prediction = "Файлът не е изображение. Моля качете .jpg, .png или .webp файл."
                return render_template('index.html', prediction=prediction)

            # Запиши и предскажи
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                prediction = predict_image(filepath)
                if prediction is None:
                    prediction = "❗ Не успяхме да разпознаем храна на изображението."
                    wine = None
                else:
                    wine = pairings.get(prediction, "Няма налична препоръка")
            except Exception as e:
                print(f"❌ Грешка при предсказание: {e}")
                prediction = "⚠️ Възникна неочаквана грешка при разпознаване."
                wine = None

            image_url = url_for('static', filename='uploads/' + file.filename)
            print(f"🖼️ Път към изображението: {image_url}")

    return render_template('index.html', prediction=prediction, image_url=image_url, wine=wine)

# Точка за стартиране
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
