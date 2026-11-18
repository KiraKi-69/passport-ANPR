from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import time
import uuid
import os

app = FastAPI()

DEEPSEEK_URL = "http://deepseek-ocr-api.ocr-service.svc:7860"
UPLOAD_DIR = "/tmp/ocr_uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- 1. Загрузка файла (Mistral Files API) ---
@app.post("/v1/files")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    return {
        "id": file_id,
        "object": "file",
        "bytes": os.path.getsize(file_path),
        "created_at": int(time.time()),
        "filename": file.filename,
        "purpose": "ocr"
    }

# --- 2. Получение URL для скачивания (Mistral Files API) ---
@app.get("/v1/files/{file_id}/url")
async def get_file_url(file_id: str):
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    
    base_url = "http://mistral-ocr-mock.ocr-service.svc:8000"
    return {
        "url": f"{base_url}/v1/files/{file_id}/content",
        "expires_at": int(time.time()) + 86400
    }

# --- 3. Скачивание файла (внутренний эндпоинт) ---
@app.get("/v1/files/{file_id}/content")
async def download_file(file_id: str):
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    return FileResponse(file_path, media_type="application/pdf", filename=f"{file_id}.pdf")

# --- 4. Удаление файла (Mistral Files API) ---
@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str):
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    if os.path.exists(file_path):
        os.remove(file_path)
    # Возвращаем успех, даже если файла не было (идемпотентность)
    return {
        "id": file_id,
        "object": "file",
        "deleted": True
    }

# --- 5. OCR (Mistral OCR API) ---
class OCRRequest(BaseModel):
    model: str = "mistral-ocr-latest"
    document: dict

@app.post("/v1/ocr")
def fake_mistral_ocr(req: OCRRequest):
    doc_url = req.document.get("document_url")
    
    if not doc_url:
        raise HTTPException(400, "Only document_url supported")

    with httpx.Client(timeout=60) as client:
        # Скачиваем файл по URL, который вернул наш же эндпоинт /url
        file_resp = client.get(doc_url)
        file_resp.raise_for_status()
        file_content = file_resp.content

        # Отправляем в DeepSeek-OCR
        files = {"file": ("doc.pdf", file_content, "application/pdf")}
        ocr_resp = client.post(f"{DEEPSEEK_URL}/ocr", files=files)
        ocr_resp.raise_for_status()
        job_id = ocr_resp.json()["job_id"]

        # Ждем результат
        for _ in range(60):
            time.sleep(1)
            status = client.get(f"{DEEPSEEK_URL}/ocr/{job_id}").json()
            if status["status"] == "completed":
                result = client.get(f"{DEEPSEEK_URL}/ocr/{job_id}/result").json()
                pages = [
                    {"index": p["page_num"] - 1, "markdown": p["markdown"]}
                    for p in result.get("pages", [])
                ]
                return {"pages": pages, "model": "mistral-ocr-latest"}
            elif status["status"] == "failed":
                raise HTTPException(500, "OCR failed")

    raise HTTPException(504, "OCR timeout")