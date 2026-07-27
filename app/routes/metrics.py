from fastapi import APIRouter, HTTPException
import json
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/metricas.json")
def get_metricas():
    try:
        with open("app/Trabalho_Final_Offline_FaceNet/final/metricas.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return JSONResponse(data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Arquivo de métricas não encontrado")
