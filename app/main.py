import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from pydantic import BaseModel
import easyocr
from typing import List, Tuple, Dict

app = FastAPI(title="EasyOCR API", version="1.0.0")

# Simple reader cache to avoid re-downloading models and speed up repeated calls
_reader_cache: Dict[Tuple[Tuple[str, ...], bool], easyocr.Reader] = {}

def get_reader(langs: List[str], gpu: bool) -> easyocr.Reader:
    key = (tuple(langs), gpu)
    if key not in _reader_cache:
        _reader_cache[key] = easyocr.Reader(langs, gpu=gpu)
    return _reader_cache[key]

@app.get("/health")
def health():
    return {"ok": True}

class UrlPayload(BaseModel):
    image_url: str
    langs: List[str] = ["es", "en"]
    gpu: bool = False
    detail: int = 1  # 0 = only text, 1 = with boxes + confidence

@app.post("/ocr/url")
def ocr_url(body: UrlPayload):
    reader = get_reader(body.langs, body.gpu)
    result = reader.readtext(body.image_url, detail=body.detail)
    if body.detail == 0:
        return {"texts": result}
    # detail == 1
    return [{"box": box, "text": text, "conf": float(conf)} for (box, text, conf) in result]

@app.post("/ocr/file")
async def ocr_file(file: UploadFile = File(...),
                   langs: str = Query("es,en"),
                   gpu: bool = Query(False),
                   detail: int = Query(1)):
    data = await file.read()
    img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    reader = get_reader(langs.split(","), gpu)
    result = reader.readtext(img, detail=detail)
    if detail == 0:
        return {"texts": result}
    return [{"box": box, "text": text, "conf": float(conf)} for (box, text, conf) in result]
