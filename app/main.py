from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Plant Doctor",
        description="Diagnose plant diseases using AI",
        version="1.0.0"
    )

    # CORS for local development or frontend calls
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(predict.router, prefix="/api")

    return app

app = create_app()
