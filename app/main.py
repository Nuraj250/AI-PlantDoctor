from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.responses import Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.routes import predict
import os

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Plant Doctor",
        description="Diagnose plant diseases using AI",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount public UI files
    app.mount("/public", StaticFiles(directory="public"), name="public")

    # Routes
    app.include_router(predict.router, prefix="/api")

    # Serve root index
    @app.get("/", response_class=FileResponse)
    async def serve_index():
        return FileResponse("public/index.html")

    # Serve dashboard
    @app.get("/dashboard", response_class=FileResponse)
    async def serve_dashboard():
        return FileResponse("public/dashboard.html")

    # 404 fallback to frontend
    @app.exception_handler(StarletteHTTPException)
    async def custom_404_handler(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404 and not request.url.path.startswith("/api"):
            index_path = "public/index.html"
            if os.path.exists(index_path):
                return FileResponse(index_path)
        return Response(status_code=exc.status_code, content=exc.detail)

    return app

app = create_app()
