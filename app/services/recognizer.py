import io
import json
import time
import base64

import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import euclidean

# Parâmetros (ajuste se necessário)
LIMIAR_ACEITACAO = 0.85
EMBEDDINGS_PATH = "app/Trabalho_Final_Offline_FaceNet/final/embeddings.npy"
USERS_PATH = "app/Trabalho_Final_Offline_FaceNet/final/users.json"

class Recognizer:
    def __init__(self):
        # Carrega detectores e embedder uma vez
        print("Carregando modelos biométricos (MTCNN, FaceNet) e banco de dados...")
        self.detector = MTCNN()
        self.embedder = FaceNet()
        # Carrega banco (pode levantar FileNotFoundError se não existir)
        self.banco_matriz = np.load(EMBEDDINGS_PATH)
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            self.banco_nomes = json.load(f)
        self.limiar = LIMIAR_ACEITACAO

    def _bytes_to_bgr(self, image_bytes: bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def recognize_image_bytes(self, image_bytes: bytes) -> dict:
        """
        Recebe bytes de imagem (jpg/png) e retorna dicionário:
        { nome, reconhecido, distancia, confianca, tempo_inferencia, rostos_encontrados }
        """
        start = time.time()

        # decode
        img_bgr = self._bytes_to_bgr(image_bytes)
        if img_bgr is None:
            raise ValueError("Imagem inválida ou corrompida")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        deteccoes = self.detector.detect_faces(img_rgb)

        resultado = {
            "nome": "Desconhecido",
            "reconhecido": False,
            "distancia": 0.0,
            "tempo_inferencia": 0,
            "rostos_encontrados": len(deteccoes),
            "confianca": 0.0,
        }

        if len(deteccoes) > 0:
            x, y, w, h = deteccoes[0]["box"]
            x, y = max(0, x), max(0, y)
            rosto = img_rgb[y:y + h, x:x + w]

            if rosto.size > 0:
                rosto = cv2.resize(rosto, (160, 160))
                rosto_batch = np.expand_dims(rosto, axis=0)
                embedding_atual = self.embedder.embeddings(rosto_batch)[0]

                menor_distancia = float("inf")
                nome_candidato = "Desconhecido"

                for i, embedding_salvo in enumerate(self.banco_matriz):
                    dist = euclidean(embedding_atual, embedding_salvo)
                    if dist < menor_distancia:
                        menor_distancia = dist
                        nome_candidato = self.banco_nomes[i]

                if menor_distancia < self.limiar:
                    resultado["nome"] = nome_candidato
                    resultado["reconhecido"] = True

                resultado["distancia"] = round(float(menor_distancia), 3)

        resultado["tempo_inferencia"] = round((time.time() - start) * 1000, 1)

        # calcula "confianca" similar ao frontend original
        if resultado["reconhecido"]:
            confianca_perc = 100.0 - ((resultado["distancia"] / self.limiar) * 30.0)
            resultado["confianca"] = round(max(70.0, min(100.0, confianca_perc)), 1)
        else:
            resultado["confianca"] = 0.0

        return resultado

# Instância pronta para importação em routes
recognizer = Recognizer()
