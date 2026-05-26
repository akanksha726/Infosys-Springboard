import os
import json
import asyncio
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime

# Import the existing engine main function
from run_market_engine import main as run_engine

app = FastAPI(title="AI Ecommerce Trend Forecaster API")

# Enable CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "output", "market_dashboard_data.json")

# Status tracking
engine_status = {
    "is_running": False,
    "last_run": None,
    "error": None
}

@app.get("/")
async def health_check():
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "engine_running": engine_status["is_running"]
    }

@app.get("/dashboard-data")
async def get_dashboard_data():
    """Serves the latest market dashboard data."""
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Dashboard data not found. Please run the engine first.")
    
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

def run_engine_task():
    """Background task to execute the data pipeline."""
    global engine_status
    engine_status["is_running"] = True
    try:
        print(f"[{datetime.now()}] Starting Market Engine...")
        run_engine()
        engine_status["last_run"] = datetime.now().isoformat()
        engine_status["error"] = None
        print(f"[{datetime.now()}] Market Engine completed successfully.")
    except Exception as e:
        engine_status["error"] = str(e)
        print(f"[{datetime.now()}] Market Engine failed: {str(e)}")
    finally:
        engine_status["is_running"] = False

@app.post("/run-engine")
async def trigger_engine(background_tasks: BackgroundTasks):
    """Triggers the full processing pipeline in the background."""
    if engine_status["is_running"]:
        return {"message": "Engine is already running", "status": engine_status}
    
    background_tasks.add_task(run_engine_task)
    return {"message": "Engine started in background", "status": engine_status}

if __name__ == "__main__":
    import uvicorn
    # Use environment variable PORT or default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
