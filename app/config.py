import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "app/models/plant_model.pt")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
