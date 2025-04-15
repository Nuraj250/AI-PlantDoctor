from PIL import Image

def is_valid_image(file: bytes) -> bool:
    try:
        Image.open(io.BytesIO(file)).verify()
        return True
    except Exception:
        return False
