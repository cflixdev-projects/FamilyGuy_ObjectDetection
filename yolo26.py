import cv2
import numpy as np
import time
import mss
from ultralytics import YOLO
import webbrowser as web

# ========================================================
# 1. YOLO MODELL LADEN
# ========================================================
MODEL_PATH = 'best.pt'  # Pfad zu deiner YOLO-Gewichtsdatei

print("⏳ Lade YOLO-Modell...")
model = YOLO(MODEL_PATH)
print("✓ YOLO-Modell erfolgreich geladen und bereit!")
print(model.names)
print(model.device)

count = 0

# ========================================================
# 2. LIVE SCREEN-CAPTURE LOOP (Fester Ausschnitt 800x600)
# ========================================================
with mss.mss() as sct:
    monitor = {"top": 200, "left": 200, "width": 800, "height": 600}
    
    print("\n🔴 Live-Erkennung mit YOLO läuft! Drücke 'q' im OpenCV-Fenster zum Beenden.")
    
    stewie_was_visible = False
    
    try:
        while True:
            start_time = time.time()
            
            # 1. Screenshot vom definierten Bereich machen
            img_sct = sct.grab(monitor)
            frame = np.array(img_sct)
            
            # 2. Farben konvertieren von BGRA (mss) zu BGR (OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # 3. YOLO Inferenz auf dem Frame ausführen
            results = model(frame_bgr, conf=0.4, verbose=False, device='cuda')
            results[0].names[0] = "Brian Griffin"
            
            # 4. Ergebnisse direkt in das Bild einzeichnen lassen
            annotated_frame = results[0].plot()
            
            # FPS einblenden
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 5. Live-Fenster anzeigen
            cv2.imshow("Family Guy YOLO Live Detector", annotated_frame)
            
            # Abbruch bei Druck auf 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"⚠️ Hinweis oder Abbruch: {e}")
    finally:
        cv2.destroyAllWindows()
        print("✓ Live-Erkennung beendet.")