"""
MODUL 4 - LAB SESI 2 - Langkah 4: Packaging - FastAPI
"""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Cek folder model lokal/docker
if Path("model/model.pkl").exists():
    MODEL_FILE = Path("model/model.pkl")
else:
    # Fallback pencarian otomatis jika dijalankan di luar docker
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = CURRENT_DIR.parent
    model_files = [f for f in PROJECT_DIR.rglob("model.pkl") if "m-83e0307e6d0941828942f96bc21d5d3e" in str(f)]
    MODEL_FILE = model_files[0] if model_files else list(PROJECT_DIR.rglob("model.pkl"))[0]

print(f"--> MEMUAT MODEL DARI: {MODEL_FILE}")

app = FastAPI(title="MLOps Training - Model Serving")
model = joblib.load(MODEL_FILE)


class PredictRequest(BaseModel):
    age: int
    income: float
    lama_bekerja_tahun: float = 0
    skor_kredit_internal: int = 650
    jumlah_pinjaman_aktif: int = 0
    jumlah_tanggungan: int = 0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    data = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    X = pd.DataFrame([data])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    return {"prediction": int(pred), "probability": round(float(proba), 4)}