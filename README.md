# 🌿 AI Plant Doctor

AI Plant Doctor is an intelligent web application to diagnose plant diseases using deep learning. Upload a photo of a plant leaf, and the app will detect possible diseases with confidence scores. Includes a training dashboard for model monitoring and prediction history.

---

## 📸 Features

- Upload plant leaf image → get disease diagnosis
- Train your own model using PlantVillage dataset
- Dashboard with:
  - Training loss/accuracy chart
  - Prediction history table
  - CSV & PNG export
- Powered by FastAPI + PyTorch + Bootstrap

---

## 📦 Requirements

- Python 3.8+
- pip / virtualenv
- `kaggle.json` for dataset download (see `kaggle_guide.md`)

---

## 🚀 Installation

```bash
git clone https://github.com/yourname/ai-plant-doctor.git
cd ai-plant-doctor
python -m venv venv
source venv/bin/activate     # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🌱 Download the Dataset (PlantVillage)

Follow the instructions in [`kaggle_guide.md`](kaggle_guide.md) to download and prepare the dataset.

---

## 🧠 Train the Model

```bash
python -m training.train --data_dir=data --epochs=10
```

- Outputs: `app/models/plant_model.pt`
- Also generates: `training_log.csv`, `training_loss.png`

---

## 🌐 Run the App

```bash
./run.sh
# or
uvicorn app.main:app --reload
```

Visit:
- Web UI: [http://localhost:8000](http://localhost:8000)
- Dashboard: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)

---

## 🛠 Project Structure

```
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── services/            # Inference logic
│   ├── routes/              # API routes
│   ├── models/              # Saved PyTorch model
│   └── utils/               # (Optional helpers)
├── training/                # Train, evaluate, split
├── public/                  # index.html, dashboard.html
├── logs/                    # prediction_history.csv
├── data/                    # train/ and val/ folders
```

---

## 📊 Dashboard Features

- Chart.js live graphs
- Export training logs to CSV
- See recent predictions in a table

---

## 📤 API Endpoints

| Endpoint             | Method | Description                        |
|----------------------|--------|------------------------------------|
| `/api/predict`       | POST   | Upload image → get prediction      |
| `/api/training-log`  | GET    | Get training loss/acc log          |
| `/api/prediction-history` | GET | List of predictions with confidence |

---

## 📄 License

MIT License

---

## 🤝 Contributing

PRs welcome. Please open an issue for any bugs or feature requests.
```

