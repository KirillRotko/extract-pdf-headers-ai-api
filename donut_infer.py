from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
from functools import lru_cache

@lru_cache()
def load_model():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa", use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    return processor, model

def extract_text_from_image(image: Image.Image) -> str:
    processor, model = load_model()
    image = image.convert("RGB")
    task_prompt = "<s_docvqa><s_question>List all headings or section titles visible in the image. Include text that looks like a title based on font size or style. Provide a numbered list.</s_question><s_answer>"
    inputs = processor(image, task_prompt, return_tensors="pt")

    outputs = model.generate(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values,
        max_length=1024,
        use_cache=True
    )

    result = processor.decode(outputs[0], skip_special_tokens=True)

    return clean_result(result, task_prompt)

def clean_result(result: str, prompt: str) -> str:
    return result.replace(prompt, "").strip()