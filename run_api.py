# run_api.py – launcher for the FastAPI app

import uvicorn

from backend.api import app  # imports the FastAPI app from backend/api.py


if __name__ == "__main__":
    print("🚀 Starting F1 API on http://127.0.0.1:8000 ...")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
