from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from api.student import router as student_router
from api.admin import router as admin_router
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Recognition Attendance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175"],  # Vite dev server ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes under /api prefix
api_router = APIRouter(prefix="/api")
api_router.include_router(student_router, prefix="/student")
api_router.include_router(admin_router, prefix="/admin")

app.include_router(api_router)


@app.get("/")
def root():
    return {
        "message": "Face Recognition Attendance API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {"status": "ok"}