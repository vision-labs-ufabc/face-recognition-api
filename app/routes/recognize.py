from fastapi import APIRouter, UploadFile, File, Body, HTTPException
from fastapi.responses import JSONResponse
import base64

from app.services.recognizer import recognizer

router = APIRouter()

@router.post("/recognize")
async def recognize(file: UploadFile = File(None), payload: dict = Body(None)):
    """
    Recebe:
     - multipart/form-data com campo 'file'
     - ou JSON {"image": "data:image/jpeg;base64,...."} (ou apenas o base64)
    Retorna o dicionário com o resultado.
    """
    image_bytes = None

    if file is not None:
        image_bytes = await file.read()
    elif payload is not None and isinstance(payload, dict) and "image" in payload:
        b64 = payload["image"]
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Imagem em base64 inválida")
    else:
        raise HTTPException(status_code=400, detail="Envie 'file' (multipart) ou JSON { 'image': 'data:image/...;base64,...' }")

    try:
        resultado = recognizer.recognize_image_bytes(image_bytes)
        return JSONResponse(resultado)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
