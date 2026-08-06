from fastapi import FastAPI
import logging

from app.routers import translation


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


app = FastAPI(
    title="Offline Translation Service"
)


app.include_router(
    translation.router,
    prefix="/ai"
)


@app.get("/")
def home():
    return {
        "status": "translation running"
    }