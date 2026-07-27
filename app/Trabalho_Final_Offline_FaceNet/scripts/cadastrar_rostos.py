import os
import cv2
import json
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

print("Carregando modelos MTCNN e FaceNet...")
detector = MTCNN()
embedder = FaceNet()

pasta_fotos = "dados/fotos_cadastro"
banco_embeddings = {}

print("Iniciando extração de embeddings...")
for arquivo in os.listdir(pasta_fotos):
    if arquivo.endswith(('.jpg', '.jpeg', '.png')):
        nome = os.path.splitext(arquivo)[0].title().replace("_", " ")
        caminho_img = os.path.join(pasta_fotos, arquivo)
        
        # Lê a imagem e converte para RGB
        img = cv2.imread(caminho_img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Detecta o rosto na foto
        deteccoes = detector.detect_faces(img_rgb)
        if len(deteccoes) == 0:
            print(f"Aviso: Nenhum rosto detectado em {arquivo}.")
            continue
            
        # Pega a primeira detecção (a mais confiável)
        x, y, w, h = deteccoes[0]['box']
        # Recorta o rosto da imagem
        rosto_recortado = img_rgb[y:y+h, x:x+w]
        rosto_recortado = cv2.resize(rosto_recortado, (160, 160))
        
        # 2. Transforma o rosto em um vetor de 512 números (Embedding)
        rosto_batch = np.expand_dims(rosto_recortado, axis=0)
        embedding = embedder.embeddings(rosto_batch)[0]
        
        banco_embeddings[nome] = embedding.tolist()
        print(f"✅ Rosto de {nome} cadastrado com sucesso!")

# 3. Salva os dados na estrutura do seu repositório
os.makedirs("final", exist_ok=True)

# Salva as matrizes (.npy) e o mapa de nomes (.json)
nomes_lista = list(banco_embeddings.keys())
matriz_embeddings = np.array(list(banco_embeddings.values()))

np.save("final/embeddings.npy", matriz_embeddings)
with open("final/users.json", "w") as f:
    json.dump(nomes_lista, f)

print("Banco de dados biométrico gerado em final/embeddings.npy e final/users.json!")