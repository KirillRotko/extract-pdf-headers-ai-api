from fastapi import FastAPI, UploadFile, File
from donut_infer import extract_text_from_image, load_model
from utils import pdf_to_images
from typing import List
from PIL import Image
import io

app = FastAPI()

@app.on_event("startup")
def preload_model():
    print("📦 Предзагрузка модели Donut...")
    load_model()
    print("✅ Модель загружена в память.")

@app.post("/extract-headings")
async def extract_headings(file: UploadFile = File(...)):
    print("Получен файл:", file.filename)
    content = await file.read()
    print("Файл прочитан, размер:", len(content))
    images: List[Image.Image] = []

    if file.filename.lower().endswith(".pdf"):
        print("Обработка PDF...")
        images = pdf_to_images(content)
    else:
        print("Обработка картинки...")
        image = Image.open(io.BytesIO(content))
        images = [image]

    results = []
    for i, img in enumerate(images):
        print(f"Обработка страницы {i+1}")
        text = extract_text_from_image(img)
        results.append({
            "page": i + 1,
            "headings_raw": text
        })
        print(f"Страница {i+1} обработана")

    print("Обработка завершена")

    print("Результат (сырое):", repr(results))

    return {"results": results}