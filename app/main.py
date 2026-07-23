# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# from app.routes.recognize import router as recognize_router
# from app.routes.metrics import router as metrics_router

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.include_router(recognize_router)
# app.include_router(metrics_router)

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {
        "message": "Face Recognition API"
    }


@app.get("/health")
def health():
    return {
        "status": "online"
    }