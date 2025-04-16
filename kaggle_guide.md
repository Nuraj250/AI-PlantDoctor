# 🌱 How to Download the PlantVillage Dataset

This guide walks you through downloading the PlantVillage dataset from Kaggle to use with AI Plant Doctor.

---

## ✅ Step 1: Get Kaggle API Key

1. Go to [https://www.kaggle.com/account](https://www.kaggle.com/account)
2. Scroll to **API** section → Click **"Create New API Token"**
3. This will download a file named `kaggle.json`

---

## ✅ Step 2: Set Up Kaggle CLI

Install the Kaggle CLI (if not already):

```bash
pip install kaggle
```

Move the `kaggle.json` to your local config folder:

```bash
mkdir ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Windows users: place `kaggle.json` in `C:\Users\<YourUsername>\.kaggle\`

---

## ✅ Step 3: Download the Dataset

Run:

```bash
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d data/raw
```

---

## ✅ Step 4: Split into Train/Val

Use the included script:

```bash
python split_dataset.py
```

This will generate:

```
data/
├── train/
│   ├── Tomato___Leaf_Mold/
│   └── ...
├── val/
```

You're now ready to train!