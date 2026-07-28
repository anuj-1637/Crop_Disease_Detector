

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

MODEL_PATH = "crop_disease_model.pth"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

TREATMENT_TIPS = {
    "healthy": "Plant looks healthy. No action needed.",
    "blight": "Remove and destroy infected leaves. Apply a copper-based fungicide. Avoid overhead watering.",
    "rust": "Apply sulfur or fungicide spray. Improve air circulation between plants.",
    "spot": "Remove affected leaves. Use fungicide and avoid wetting leaves while watering.",
    "mold": "Improve ventilation, reduce humidity, apply appropriate fungicide.",
    "virus": "Remove infected plants to prevent spread. Control insect vectors (aphids/whiteflies).",
}


class DiseaseClassifier:
    """Wraps the trained model so main.py just calls .predict()"""

    def __init__(self, model_path=MODEL_PATH):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.class_names = checkpoint["class_names"]

        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, len(self.class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()

        self.model = model

    def _get_treatment_tip(self, class_name: str) -> str:
        lower_name = class_name.lower()
        for keyword, tip in TREATMENT_TIPS.items():
            if keyword in lower_name:
                return tip
        return "No specific tip available. Consult a local agricultural expert."

    def predict(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            confidence, predicted_idx = torch.max(probs, 0)

        predicted_class = self.class_names[predicted_idx.item()]
        confidence_pct = round(confidence.item() * 100, 2)

        top3_prob, top3_idx = torch.topk(probs, k=min(3, len(self.class_names)))
        top3 = [
            {"class_name": self.class_names[i], "confidence": round(top3_prob[j].item() * 100, 2)}
            for j, i in enumerate(top3_idx)
        ]

        return {
            "predicted_class": predicted_class,
            "confidence": confidence_pct,
            "top3": top3,
            "treatment_tip": self._get_treatment_tip(predicted_class),
        }
