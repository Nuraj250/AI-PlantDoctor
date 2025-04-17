# 🌱 How to Train AI Plant Doctor on Your Own Dataset

This guide walks you through how to prepare your **own plant leaf dataset**, train the model using it, and integrate it into the AI Plant Doctor app without needing the Kaggle PlantVillage dataset.

---

## ✅ 1. Organize Your Dataset

Structure your folder like this:

```
your_dataset/
├── Tomato___Healthy/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Tomato___Leaf_Spot/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Tomato___Blight/
│   └── ...
```

Each subfolder = **1 class**, and the images should be:
- `.jpg`, `.jpeg`, or `.png`
- Clear, centered leaf images

---

## ✅ 2. Place Raw Data

Move your dataset into the project folder:

```
AI-PlantDoctor/
└── data/
    └── raw/
        └── your_dataset/
            └── [class folders...]
```

---

## ✅ 3. Split Into `train/` and `val/`

Run this script:

```bash
python split_dataset.py
```

This generates:

```
data/
├── train/
│   └── Tomato___Healthy/
├── val/
│   └── Tomato___Healthy/
```

You can customize the split ratio in the script (`split_ratio=0.8` by default).

---

## ✅ 4. Train Your Model

Run:

```bash
python -m training.train --data_dir=data --epochs=10
```

You will get:
- `app/models/plant_model.pt` (the model)
- `training_log.csv` (loss/acc log)
- `training_loss.png` (graph)

---

## ✅ 5. Update Class Names

In `app/services/inference.py`, make sure the `CLASS_NAMES` list matches your folders:

```python
CLASS_NAMES = [
  "Tomato___Healthy",
  "Tomato___Leaf_Spot",
  "Tomato___Blight"
]
```

You can also load this dynamically later (optional enhancement).

---

## ✅ 6. Add Treatments

Edit `app/utils/treatment_guide.json`:

```json
{
  "Tomato___Leaf_Spot": {
    "treatment": "Use copper fungicide spray.",
    "video": "https://youtube.com/leafspot",
    "product": "https://amazon.com/copper-spray",
    "lang": {
      "si": "කෝපර් වායුමය ස්ප්‍රේ භාවිතා කරන්න.",
      "ta": "காப்பர் பீதி ஸ்ப்ரே பயன்படுத்தவும்."
    }
  }
}
```

Do this for each class name.

---

## ✅ 7. Run the App

```bash
uvicorn app.main:app --reload
```
## 🔄 Retrain Anytime

- Modify or expand your data
- Rerun `split_dataset.py`
- Retrain using `training.train`

---

## 🛠 Tips

- Ideal image size: ~224x224px
- Use at least 50+ images per class
- Keep class names consistent (no typos)
