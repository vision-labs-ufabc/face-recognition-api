import os
import json
import numpy as np
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("Inicializando pipeline de avaliação...")
detector = MTCNN()
embedder = FaceNet()

# Carrega o banco de dados da fechadura
try:
    banco_matriz = np.load("final/embeddings.npy")
    with open("final/users.json", "r") as f:
        banco_nomes = json.load(f)
except FileNotFoundError:
    print("Erro: Rode o cadastrar_rostos.py primeiro para gerar o banco de dados.")
    exit()

LIMIAR_ACEITACAO = 0.85
PASTA_TESTE = "dados/fotos_teste"

# Listas para a Matriz de Confusão
y_true = [] # O que a pessoa REALMENTE é (1 = Autorizado, 0 = Desconhecido)
y_pred = [] # O que o MODELO previu (1 = Liberou a porta, 0 = Trancou a porta)

print(f"\nIniciando varredura na pasta: {PASTA_TESTE}")

for pasta_pessoa in os.listdir(PASTA_TESTE):
    caminho_pasta = os.path.join(PASTA_TESTE, pasta_pessoa)
    if not os.path.isdir(caminho_pasta): 
        continue

    # Regra Lógica: Se o nome da pasta de teste existe no banco de dados cadastrado, 
    # essa pessoa tem passe livre (1). Se não (ex: pasta "Desconhecidos"), passe negado (0).
    eh_autorizado_real = 1 if pasta_pessoa.title() in banco_nomes else 0

    fotos = [f for f in os.listdir(caminho_pasta) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Avaliando {len(fotos)} fotos da classe verdadeira: {pasta_pessoa} (Label: {eh_autorizado_real})")

    for arquivo in fotos:
        caminho_img = os.path.join(caminho_pasta, arquivo)
        img = cv2.imread(caminho_img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Detecção
        deteccoes = detector.detect_faces(img_rgb)
        if len(deteccoes) == 0:
            print(f"  [!] Rosto não detectado pelo MTCNN em: {arquivo}. Ignorando.")
            continue
            
        x, y, w, h = deteccoes[0]['box']
        x, y = max(0, x), max(0, y)
        rosto = img_rgb[y:y+h, x:x+w]
        
        if rosto.size == 0: 
            continue
            
        rosto = cv2.resize(rosto, (160, 160))
        rosto_batch = np.expand_dims(rosto, axis=0)
        
        # 2. Extração de Features (Embedding)
        embedding_atual = embedder.embeddings(rosto_batch)[0]
        
        # 3. Cálculo de Distância (O Rosto bate com alguém do banco?)
        menor_distancia = float('inf')
        for embedding_salvo in banco_matriz:
            dist = euclidean(embedding_atual, embedding_salvo)
            if dist < menor_distancia:
                menor_distancia = dist
        
        # 4. Decisão do Modelo
        eh_autorizado_pred = 1 if menor_distancia < LIMIAR_ACEITACAO else 0
        
        y_true.append(eh_autorizado_real)
        y_pred.append(eh_autorizado_pred)

# ==========================================
# CÁLCULO E EXPORTAÇÃO DAS MÉTRICAS
# ==========================================
print("\nGerando Matriz de Confusão e consolidação de métricas...")

# O scikit-learn faz a matemática pesada comparando as duas listas
acuracia = accuracy_score(y_true, y_pred)
precisao = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

metricas_finais = {
    "acuracia": round(acuracia * 100, 1),
    "precisao": round(precisao * 100, 1),
    "recall": round(recall * 100, 1),
    "f1_score": round(f1 * 100, 1)
}

# Salva o arquivo JSON que será consumido pelo Frontend
with open("final/metricas.json", "w", encoding="utf-8") as f:
    json.dump(metricas_finais, f, indent=4)

print("\n" + "="*40)
print("RESULTADOS DA AVALIAÇÃO OFFLINE")
print("="*40)
print(f"Total de imagens validadas: {len(y_true)}")
print(f"Verdadeiros Positivos (Abriu certo): {tp}")
print(f"Verdadeiros Negativos (Barrou certo): {tn}")
print(f"Falsos Positivos (Abriu pro invasor): {fp}  <-- Risco de Segurança")
print(f"Falsos Negativos (Barrou o dono): {fn}      <-- Risco de Experiência")
print("-" * 40)
print(f"Acurácia: {metricas_finais['acuracia']}%")
print(f"Precisão: {metricas_finais['precisao']}%")
print(f"Recall:   {metricas_finais['recall']}%")
print(f"F1-Score: {metricas_finais['f1_score']}%")
print("="*40)
print("✅ Arquivo metricas.json gerado com sucesso para o dashboard HTML!")