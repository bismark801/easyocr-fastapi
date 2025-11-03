
# EasyOCR API (FastAPI) listo para EasyPanel

Servicio HTTP para OCR con [EasyOCR](https://github.com/JaidedAI/EasyOCR), empaquetado con FastAPI y Docker.

## Endpoints
- `GET /health` → healthcheck
- `POST /ocr/file` → sube una imagen; query: `langs=es,en`, `gpu=false`, `detail=1`
- `POST /ocr/url` (JSON) → `{"image_url":"https://...","langs":["es","en"],"gpu":false,"detail":1}`

> `detail=0` devuelve solo textos; `detail=1` devuelve cajas, texto y confianza.

## Despliegue en EasyPanel
1. Crea **New App** → **From Git Repository** (sube este proyecto a tu GitHub) **o** **From Dockerfile**.
2. En **Domains & Proxy**, pon **Proxy port = 8000**.
3. (Opcional) En **Volumes**, monta un volumen en `/root/.EasyOCR` para cachear modelos.
4. **Deploy**.

## cURL de prueba
Archivo:
```bash
curl -F "file=@/ruta/imagen.jpg" "https://TU_DOMINIO/ocr/file?langs=es,en&gpu=false&detail=1"
```
URL:
```bash
curl -X POST "https://TU_DOMINIO/ocr/url"   -H "Content-Type: application/json"   -d '{"image_url":"https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png","langs":["en"],"gpu":false,"detail":1}'
```

## Desarrollo local
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Abre `http://localhost:8000/docs` para ver Swagger.

## Notas
- La **primera llamada** descarga modelos; puede tardar.
- Si tu servidor no tiene GPU, usa `gpu=false` (por defecto).
