from fastapi import FastAPI
import logging

from app.routers import speech, translation


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


app = FastAPI(
    title="Google Cloud Translation Service"
)


app.include_router(
    translation.router,
    prefix="/ai"
)

app.include_router(speech.router, prefix="/ai")


@app.get("/")
def home():
    return {
        "status": "translation running",
        "provider": "Google Cloud Translation NMT",
    }
