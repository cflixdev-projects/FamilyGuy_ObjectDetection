import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import time
import mss

# ========================================================
# 1. SETUP & MODELL LADEN
# ========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 128
CLASSES = ['Brain Griffin', 'Lois Griffin', 'Peter Griffin', 'Stewie Griffin']

# Exakte CNN-Architektur (inklusive aller BatchNorm-Layer)
class FamilyGuyCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 1, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Modell initialisieren und lokale Gewichtsdatei laden
model = FamilyGuyCNN(num_classes=len(CLASSES)).to(DEVICE)
MODEL_PATH = 'family_guy_best.pth'

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✓ Modell erfolgreich geladen und bereit für den Live-Test!")

# Transformations-Pipeline (ohne Augmentation)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ========================================================
# 2. LIVE SCREEN-CAPTURE LOOP (Fester Ausschnitt 800x600)
# ========================================================
with mss.mss() as sct:
    # Fester Ausschnitt: 800x600 Pixel, startend bei top=100, left=200
    monitor = {"top": 100, "left": 200, "width": 800, "height": 600}
    
    print("\n🔴 Live-Erkennung läuft! Drücke 'q' im OpenCV-Fenster zum Beenden.")
    
    try:
        while True:
            start_time = time.time()
            
            # 1. Screenshot vom definierten Bereich machen
            img_sct = sct.grab(monitor)
            frame = np.array(img_sct)
            
            # 2. Farben konvertieren (BGRA zu RGB für PIL/Torch)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # 3. Vorverarbeitung für das CNN
            input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
            
            # 4. Vorhersage berechnen
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
            predicted_class = CLASSES[predicted_idx.item()]
            conf_score = confidence.item() * 100
            
            # 5. Text auf das Bild zeichnen
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            text = f"{predicted_class} ({conf_score:.1f}%)"
            
            cv2.putText(display_frame, text, (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # 6. Live-Fenster anzeigen
            cv2.imshow("Family Guy Live Detector", display_frame)
            
            # Abbruch bei Druck auf 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"⚠️ Hinweis oder Abbruch: {e}")
    finally:
        cv2.destroyAllWindows()
        print("✓ Live-Erkennung beendet.")