import cv2
import numpy as np
import time
import json
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import euclidean

app = FastAPI()

print("Carregando modelos biométricos e banco de dados...")
detector = MTCNN()
embedder = FaceNet()

# Carrega a base de rostos cadastrados
banco_matriz = np.load("final/embeddings.npy")
with open("final/users.json", "r") as f:
    banco_nomes = json.load(f)

# Limiar de segurança da fechadura (Threshold)
# No FaceNet, L2 distances menores que 1.0 geralmente indicam a mesma pessoa
# Para fechaduras severas, 0.8 ou 0.9 é um bom número.
LIMIAR_ACEITACAO = 0.85 

@app.get("/")
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/metricas.json")
async def get_metricas():
    # Permite que o frontend baixe o arquivo de métricas
    import json
    with open("final/metricas.json", "r", encoding="utf-8") as f:
        return json.load(f)
    
@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            encoded_data = data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            start_time = time.time()
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. Tenta achar rostos na imagem
            deteccoes = detector.detect_faces(img_rgb)
            
            resultado = {
                "nome": "Desconhecido",
                "reconhecido": False,
                "distancia": 0.0,
                "tempo_inferencia": 0,
                "rostos_encontrados": len(deteccoes)
            }
            
            if len(deteccoes) > 0:
                # Pega a caixa delimitadora do rosto mais visível
                x, y, w, h = deteccoes[0]['box']
                # Garante que as coordenadas não vazem da imagem
                x, y = max(0, x), max(0, y)
                
                rosto = img_rgb[y:y+h, x:x+w]
                
                # Previne erros caso a detecção seja inválida/vazia
                if rosto.size > 0:
                    rosto = cv2.resize(rosto, (160, 160))
                    rosto_batch = np.expand_dims(rosto, axis=0)
                    
                    # 2. Calcula a assinatura do rosto na câmera
                    embedding_atual = embedder.embeddings(rosto_batch)[0]
                    
                    # 3. Compara com todo mundo do banco de dados (Distância Euclidiana)
                    menor_distancia = float('inf')
                    nome_identificado = "Desconhecido"
                    
                    for i, embedding_salvo in enumerate(banco_matriz):
                        dist = euclidean(embedding_atual, embedding_salvo)
                        if dist < menor_distancia:
                            menor_distancia = dist
                            nome_candidato = banco_nomes[i]
                            
                    # 4. A Regra de Negócio da Fechadura
                    if menor_distancia < LIMIAR_ACEITACAO:
                        resultado["nome"] = nome_candidato
                        resultado["reconhecido"] = True
                    
                    resultado["distancia"] = round(float(menor_distancia), 3)
                    
            resultado["tempo_inferencia"] = round((time.time() - start_time) * 1000, 1)
            
           # Adaptação para o frontend (convertendo a distância em uma "confiança" % visual)
            if resultado["reconhecido"]:
                # Mapeia: Distância 0.0 -> 100% | Distância = Limiar -> 70%
                confianca_perc = 100.0 - ((resultado["distancia"] / LIMIAR_ACEITACAO) * 30.0)
                # Garante que fique travado entre 70% e 100%
                resultado["confianca"] = round(max(70.0, min(100.0, confianca_perc)), 1)
            else:
                resultado["confianca"] = 0.0

            await websocket.send_json(resultado)
            
    except WebSocketDisconnect:
        pass