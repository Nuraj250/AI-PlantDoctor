# 🌿 AI Plant Doctor

AI Plant Doctor is a deep learning web app that diagnoses plant leaf diseases and recommends effective treatments — with multilingual support and farmer-friendly advice. Includes a training dashboard, CSV logs, prediction history, and optional email reminders.

---

## 📸 Features

- 🧠 Upload plant leaf image → Get disease diagnosis
- 🌱 Treatment recommendations (English, Sinhala, Tamil)
- 🌐 Web dashboard for training stats + prediction history
- 📬 Email treatment reminders (optional)
- 🧪 Train your own model using Kaggle's PlantVillage dataset
- 🔁 Auto-detect user language from browser

---

## 🛠 Project Stack

- FastAPI + Python for backend
- PyTorch for model inference
- Bootstrap + Chart.js for frontend
- CSV + JSON-based logging
- Optional: Gmail SMTP for reminders

---

## 📦 Requirements

- Python 3.8+
- pip / virtualenv
- `kaggle.json` for dataset (see [`kaggle_guide.md`](kaggle_guide.md))
- Optional: Gmail app password for email reminders

---

## 🚀 Installation

```bash
git clone https://github.com/Nuraj250/ai-plant-doctor.git
cd ai-plant-doctor
python -m venv venv
source venv/bin/activate     # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🌱 Get the Dataset

1. Download from Kaggle (see [`kaggle_guide.md`](kaggle_guide.md))
2. Extract to: `data/raw/PlantVillage/`
3. Run this to split data:

```bash
python split_dataset.py
```

---

## 🧠 Train the Model

```bash
python -m training.train --data_dir=data --epochs=10
```

Outputs:
- `app/models/plant_model.pt`
- `training_log.csv`
- `training_loss.png`

---

## 🌐 Run the App

```bash
./run.sh
# or
uvicorn app.main:app --reload
```

Visit:

- 🌿 App: [http://localhost:8000](http://localhost:8000)
- 📊 Dashboard: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)

---

## 🧪 API Endpoints

| Endpoint                 | Method | Description                          |
|--------------------------|--------|--------------------------------------|
| `/api/predict`           | POST   | Upload image + language + email      |
| `/api/training-log`      | GET    | Returns `training_log.csv` as JSON   |
| `/api/prediction-history` | GET  | Returns past predictions as JSON     |

---

## 🗃 Prediction Log

Predictions are saved to:

```text
logs/prediction_history.csv
```

With:
- Timestamp
- Label
- Confidence
- Language
- Treatment given

---

## ✉️ Email Reminders (Optional)

Enable email reminders by:

1. Creating a Gmail App Password
2. Updating `.env` or `send_email_reminder()` with credentials
3. Submitting an email address with your image

🛑 Email is never stored — it’s used to send a 1-time treatment reminder.

---

## 🌍 Multilingual Support

- UI dropdown for: English, සිංහල (Sinhala), தமிழ் (Tamil)
- Auto-detects browser language (via `Accept-Language` header)
- Treatment guidance translated inside `treatment_guide.json`

---

## 📊 Dashboard

- Training Loss & Accuracy (Chart.js)
- Export CSV and PNG
- Table of latest predictions

---

## 📂 Project Structure

```
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── routes/              # API routes
│   ├── services/            # Model & inference
│   ├── utils/               # treatment_guide.json
│   ├── models/              # Saved .pt file
├── public/                  # index.html, dashboard.html
├── training/                # train.py, dataset.py
├── logs/                    # prediction_history.csv
├── data/                    # train/ and val/
├── split_dataset.py
├── requirements.txt
```

---

## 🤝 Contributing

- PRs welcome!
- Help add more languages, crop diseases, or regional advice
- Farmers, researchers, and devs — let's collaborate

---

## 📄 License

MIT License

---

## 🙏 Credits

- Dataset: [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Inspiration: Community gardeners & farmers
