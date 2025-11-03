import io
import logging
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
import easyocr
import requests
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="EasyOCR API", version="1.0.1")

# cache del lector
_reader_cache: Dict[Tuple[Tuple[str, ...], bool], easyocr.Reader] = {}

def get_reader(langs: List[str], gpu: bool) -> easyocr.Reader:
    key = (tuple(langs), gpu)
    if key not in _reader_cache:
        _reader_cache[key] = easyocr.Reader(langs, gpu=gpu)
    return _reader_cache[key]

@app.get("/health")
def health():
    return {"ok": True}

# opcional, para que "/" no dé 404
@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "health": "/health"}

class UrlPayload(BaseModel):
    image_url: str
    langs: List[str] = ["es", "en"]
    gpu: bool = False
    detail: int = 1  # 0 solo texto, 1 cajas+confianza

# *** NUEVO: descargamos la imagen nosotros ***
@app.post("/ocr/url")
def ocr_url(body: UrlPayload):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/*,*/*;q=0.8",
        }
        r = requests.get(body.image_url, headers=headers, timeout=30)
        if r.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="El origen devuelve 429 (rate limit/bloqueo). Usa /ocr/file o cambia de URL."
            )
        if r.status_code != 200 or not r.content:
            raise HTTPException(status_code=400, detail=f"No se pudo descargar la imagen: HTTP {r.status_code}")

        img = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
        reader = get_reader(body.langs, body.gpu)
        result = reader.readtext(img, detail=body.detail)
        if body.detail == 0:
            return {"texts": result}
        return [{"box": box, "text": text, "conf": float(conf)} for (box, text, conf) in result]

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="El contenido descargado no es una imagen válida.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR error: {e}")

@app.post("/ocr/file")
async def ocr_file(
    file: UploadFile = File(...),
    langs: str = Query("es,en"),
    gpu: bool = Query(False),
    detail: int = Query(1)
):
    try:
        # Leer el archivo
        data = await file.read()
        
        # Validar que no esté vacío
        if not data:
            raise HTTPException(status_code=400, detail="El archivo está vacío")
        
        # Validar que sea una imagen válida
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="El archivo no es una imagen válida")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al abrir la imagen: {str(e)}")
        
        # Convertir a numpy array
        img_array = np.array(img)
        
        # Parsear idiomas
        langs_list = [lang.strip() for lang in langs.split(",")]
        
        # Obtener reader y procesar
        reader = get_reader(langs_list, gpu)
        result = reader.readtext(img_array, detail=detail)
        
        # Devolver resultado
        if detail == 0:
            return {"texts": result}
        
        return [
            {"box": box, "text": text, "conf": float(conf)} 
            for (box, text, conf) in result
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        # Log del error para debugging
        logging.error(f"Error en OCR: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")
