import base64
import re
from io import BytesIO
from PIL import Image


def decode_base64_image(data_url: str) -> Image.Image:
    """
    Converts a base64 data URL (e.g. "data:image/jpeg;base64,/9j/...")
    into a PIL Image in RGB mode.
    """
    # Strip the "data:<mime>;base64," prefix if present
    match = re.match(r"data:[^;]+;base64,(.+)", data_url, re.DOTALL)
    if match:
        b64_data = match.group(1)
    else:
        b64_data = data_url  # assume raw base64 if no prefix

    image_bytes = base64.b64decode(b64_data)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.save("debug_decoded_image.jpg")  # Save for debugging
    return image