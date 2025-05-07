# 🧠🍷 Wine4Food AI

> Качи снимка на ястие – получи предложение за вино!  
> AI асистент за хранителни изображения и винени препоръки, обучен върху Food-101 и ползващ EfficientNet-B0.

---

## 🔍 Какво прави този проект?

**Wine4Food AI** е уеб приложение, което:
- приема снимка на храна от потребителя
- използва обучен deep learning модел (EfficientNet-B0), за да разпознае ястието
- дава винена препоръка според вида на храната (на база подбран речник)

---

## 🛠️ Как работи?

- Използва **PyTorch** и **timm** за зареждане и inference с обучен модел
- Моделът е хостнат в **Hugging Face Hub**  
- Логиката за винен pairing е в `pairings.py`
- Уеб интерфейсът е създаден с **Flask**

---

## 🚀 Как да го стартираш локално

```bash
# 1. Клонирай проекта
git clone https://github.com/enggv/wine4food-ai-online.git
cd wine4food-ai-online

# 2. Създай виртуална среда (по избор)
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate      # Windows

# 3. Инсталирай зависимостите
pip install -r requirements.txt

# 4. Стартирай приложението
python app/webapp.py
След това отвори браузър и зареди: http://localhost:5000
🤖 Модел

Моделът е обучен с Food-101 и fine-tune-нат за 101 класа.
📦 Hugging Face модел

🔮 Планирано развитие

    🎯 Hugging Face Spaces или Render Deployment

    📊 Визуализации на резултати

    🌍 Многоезичен интерфейс

    🧠 Подобрения в препоръчващата система

📄 Лиценз

MIT License

By @enggv – AI for wine lovers 🍷🤖