====================================================
GUIA DE EXECUÇÃO COMPLETA (BACKEND E GERAÇÃO DE DADOS)
====================================================

Siga esta ordem passo a passo para preparar o ambiente, gerar os dados biométricos e de métricas, e por fim subir o servidor.

----------------------------------------------------
1. ORGANIZAÇÃO DAS PASTAS PARA OS TESTES
----------------------------------------------------
Antes de rodar os scripts, certifique-se de que a estrutura de pastas na raiz do projeto contenha as seguintes pastas de insumos (caso ainda não existam):
- dados/fotos_cadastro/ (Para colocar as fotos das pessoas que deseja realizar o reconhecimento com o FaceNet, coloque o nome da pessoa no arquivo).
- dados/fotos_teste/ (Com subpastas separadas por nome, ex: Lucas/, Samira/, Desconhecidos/, para o script de avaliação).

----------------------------------------------------
2. ORDEM DE EXECUÇÃO DOS CÓDIGOS (SCRIPTS OFFLINE)
----------------------------------------------------
Abra o terminal na raiz do projeto, baixe os requirements.txt (crie um .venv se desejar) e execute os scripts na seguinte sequência:

PASSO A: Cadastrar os Rostos (Gera os Embeddings)
Coloque 1 foto de boa qualidade de cada pessoa em dados/fotos_cadastro/ e execute o script de cadastro para extrair as assinaturas matemáticas:
> python cadastrar_rostos.py
(O que ele faz: Cria os arquivos final/embeddings.npy e final/users.json).

PASSO B: Avaliar o Modelo e Gerar as Métricas
Organize as imagens de teste nas subpastas dentro de dados/fotos_teste/ e execute o script de validação:
> python avaliar_modelo.py
(O que ele faz: Simula a varredura, calcula a Matriz de Confusão e gera o arquivo final/metricas.json com a acurácia, precisão, recall e f1-score reais do modelo).

----------------------------------------------------
3. COMO EXECUTAR O SERVIDOR 
----------------------------------------------------
Com os dados biométricos e as métricas gerados, suba o servidor backend utilizando o Uvicorn:
> uvicorn app.server:app --reload

----------------------------------------------------
4. COMO EXECUTAR A PÁGINA 
----------------------------------------------------
1. Certifique-se de que o backend está rodando no terminal (passo anterior).
2. Abra o arquivo index.html no seu navegador de preferência (ou utilize uma extensão como o Live Server do VS Code).
3. Na interface do dashboard, clique no botão "Iniciar Câmera", permita o acesso ao dispositivo e o sistema se conectará via WebSocket (ws://localhost:8000/ws/video) para iniciar o reconhecimento facial em tempo real com as métricas e o histórico atualizados.