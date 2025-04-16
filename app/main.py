from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routes import predict

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Plant Doctor",
        description="Diagnose plant diseases using AI",
        version="1.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # You can restrict to specific frontend domains
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes
    app.include_router(predict.router, prefix="/api")

    # Mount static UI (Bootstrap form)
    app.mount("/public", StaticFiles(directory="public"), name="public")

    # Serve index.html at root
    @app.get("/", response_class=FileResponse)
    async def serve_index():
        return FileResponse("public/index.html")

    return app

app = create_app()
