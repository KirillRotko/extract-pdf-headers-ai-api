from pdf2image import convert_from_bytes
from typing import List
from PIL import Image

def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    return convert_from_bytes(pdf_bytes, dpi=300)