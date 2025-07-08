import os
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from PIL import Image
from threading import Thread
import time
from io import BytesIO
import uvicorn

# Constants
MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

# Setup device
device = torch.device("cuda")

# Load RolmOCR model
MODEL_ID = "reducto/RolmOCR"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device).eval()

# Create FastAPI app
app = FastAPI(title="RolmOCR API")

def extract_text_from_image(image: Image.Image, query: str):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query},
        ]
    }]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)

    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
    return buffer.strip()

@app.post("/ocr/")
async def ocr_image(file: UploadFile = File(...), query: str = "Extract the text from this image"):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        result = extract_text_from_image(image, query)
        return JSONResponse({"result": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# For local running (comment out in Colab)
#if __name__ == "__main__":
#  uvicorn.run(app, host="0.0.0.0", port=7860)
