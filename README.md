# 🌿 AI Plant Doctor

AI Plant Doctor is a full-stack AI-powered application that diagnoses plant diseases from leaf images and recommends treatments. Built with FastAPI, PyTorch, Bootstrap, and Chart.js — it also includes a training dashboard, multilingual support, prediction history logging, and optional treatment reminders.

---
-
## 🚀 Features

- 🧠 Upload a plant leaf image → get diagnosis and confidence score
- 🌱 Treatment recommendations in English, Sinhala, Tamil
- 📊 Dashboard with training loss, accuracy charts, and prediction history
- ✉️ Optional email reminders for treatment
- 🗂 Prediction logs stored in CSV
- 🔤 Auto-detects browser language
- 🖼 Image preview after upload

---

## 🧰 Stack

- **Backend**: FastAPI, Python, PyTorch
- **Frontend**: HTML, Bootstrap 5, Chart.js
- **Data**: PlantVillage dataset (Kaggle)
- **Model**: ResNet-based classifier
- **Logging**: CSV + Chart
- **Training**: Customizable via CLI

---

## 📦 Installation

```bash
git clone https://github.com/yourname/ai-plant-doctor.git
cd ai-plant-doctor
python -m venv venv
venv\Scripts\activate    # Or source venv/bin/activate (Linux/Mac)
pip install -r requirements.txt
```

---

## 🌱 Dataset Setup

To download and split the PlantVillage dataset, follow [`kaggle_guide.md`](kaggle_guide.md)
or train your own one dataset, follow [`train-your-own-data.md`](train-your-own-data.md)

---

## 🧠 Train the Model

```bash
python -m training.train --data_dir=data --epochs=10
```

After training, you will get:

- `app/models/plant_model.pt` ✅
- `training_log.csv` ✅
- `training_loss.png` ✅

---

## 🔁 Run the App

```bash
uvicorn app.main:app --reload
```

### Access:
- 🌿 Web UI: [http://localhost:8000](http://localhost:8000)
- 📊 Dashboard: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)

---

## 🧪 API Endpoints

| Endpoint                 | Method | Description                          |
|--------------------------|--------|--------------------------------------|
| `/api/predict`           | POST   | Upload image → diagnosis + treatment |
| `/api/training-log`      | GET    | Get training history (CSV)           |
| `/api/prediction-history` | GET   | Past predictions from logs           |

---

## 🗃 Logs

- **Predictions** → `logs/prediction_history.csv`
- **Training log** → `training_log.csv`

---

## 🌍 Language Support

- English (`en`)
- Sinhala (`si`)
- Tamil (`ta`)

Auto-detects language via browser header or dropdown.

---

## 📨 Email Reminders

Enable by modifying `send_email_reminder()` in `inference.py` with your SMTP credentials.

---

## 📁 Project Structure

```
├── app/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   ├── utils/                 # treatment_guide.json
│   └── models/                # plant_model.pt
├── public/                    # index.html + dashboard.html
├── training/                  # train.py, dataset.py, evaluate.py
├── logs/                      # prediction_history.csv
├── data/                      # train/, val/, raw/
├── split_dataset.py
├── requirements.txt
```

---

## 📄 License

MIT License

---

## ✨ Credits

- Dataset: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Icon: Bootstrap Icons + Emoji
```