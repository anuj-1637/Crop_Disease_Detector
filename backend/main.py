"""
main.py
-------
FastAPI app entry point.

Run locally with:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000/docs for the interactive API docs.

Make sure 'crop_disease_model.pth' (produced by the training script)
is placed in this same 'backend' folder before starting the server.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model import DiseaseClassifier
from schemas import PredictionResponse, HealthResponse

app = FastAPI(
    title="Crop Disease Detection API",
    description="Upload a leaf image and get a disease prediction.",
    version="1.0.0",
)

# Allow the frontend (running on a different origin/port) to call this API.
# For production, replace "*" with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once when the server starts (not on every request - that would be slow)
classifier = None


@app.on_event("startup")
def load_model_on_startup():
    global classifier
    try:
        classifier = DiseaseClassifier()
        print(f"Model loaded successfully with {len(classifier.class_names)} classes.")
    except FileNotFoundError:
        print("WARNING: Model file not found. /predict will fail until the model is added.")


@app.get("/", response_model=HealthResponse)
def health_check():
    """Simple health check endpoint - useful to verify deployment worked."""
    return HealthResponse(
        status="ok",
        classes_loaded=len(classifier.class_names) if classifier else 0,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(file: UploadFile = File(...)):
    """
    Accepts an uploaded leaf image and returns the predicted disease class,
    confidence score, top-3 alternatives, and a basic treatment tip.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    try:
        result = classifier.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionResponse(**result)
