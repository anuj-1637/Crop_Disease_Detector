# Crop Disease Detector

## Project Structure
```
crop-disease-app/
├── backend/
│   ├── main.py              # FastAPI app (API endpoints)
│   ├── model.py             # Model loading + prediction logic
│   ├── schemas.py           # Pydantic response models
│   ├── requirements.txt     # Backend dependencies
│   └── crop_disease_model.pth  # <-- YOU add this (from training script)
├── frontend/
│   └── index.html           # Standalone frontend (HTML/CSS/JS, no framework needed)
└── README.md
```

## Step 1: Train the model first
Use the earlier `crop_disease_classifier.py` script to train and save
`crop_disease_model.pth`. Copy that file into `backend/`.

## Step 2: Run the backend locally
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` — FastAPI auto-generates interactive
API docs where you can test the `/predict` endpoint directly.

## Step 3: Run the frontend
Just open `frontend/index.html` directly in a browser (double-click it),
or serve it with any static server:
```bash
cd frontend
python -m http.server 5500
```
Then visit `http://localhost:5500`.

The frontend calls `http://localhost:8000` by default (see the hidden
`apiUrl` input in index.html) — change that value once your backend
is deployed to a real URL.

## Step 4: Deploy

### Backend options (pick one)
- **Render.com** (easiest, free tier): Create a new "Web Service",
  point it at your GitHub repo's `backend/` folder, set the start
  command to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- **Railway.app**: Similar to Render, auto-detects FastAPI apps.
- **Hugging Face Spaces** (Docker SDK): Good if you want it tied to
  your ML portfolio directly.

Important: the `crop_disease_model.pth` file needs to be included in
your deployment (either committed to the repo if small enough, or
uploaded to cloud storage and downloaded on startup for larger models).

### Frontend options
- **Netlify** or **Vercel**: Drag-and-drop the `frontend/` folder,
  it's just static HTML/CSS/JS — deploys in seconds.
- After deploying, update the `apiUrl` value in `index.html` to your
  deployed backend's URL (e.g. `https://your-app.onrender.com`).

## API Reference

### `GET /`
Health check. Returns `{"status": "ok", "classes_loaded": N}`.

### `POST /predict`
Accepts a multipart form upload with key `file` (an image).
Returns:
```json
{
  "predicted_class": "Tomato___Late_blight",
  "confidence": 96.42,
  "top3": [
    {"class_name": "Tomato___Late_blight", "confidence": 96.42},
    {"class_name": "Tomato___Early_blight", "confidence": 2.1},
    {"class_name": "Tomato___healthy", "confidence": 1.48}
  ],
  "treatment_tip": "Remove and destroy infected leaves..."
}
```
"# Crop_Disease_Detector" 
