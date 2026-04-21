import cv2
import serial
import time
from ultralytics import YOLO

# --- CONFIGURACIÓN ---
try:
    arduino = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2) 
    print(">>> Conexión con Arduino establecida.")
except:
    print(">>> Error: Arduino no detectado.")
    arduino = None

model = YOLO('runs/classify/train/weights/best.pt')
cap = cv2.VideoCapture(0)
UMBRAL_CONFIANZA = 0.7 

print(">>> Camara en funcionamiento")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        print(">>> Leyendo imagen") 

        results = model(frame, verbose=False)
        
        if results[0].probs is not None:
            confianza = results[0].probs.top1conf.item()
            
            if confianza >= UMBRAL_CONFIANZA:
                class_id = results[0].probs.top1
                nombre_clase = results[0].names[class_id].lower()
                
                print(f">>> El resultado es: {nombre_clase}")
                
                if arduino:
                    if nombre_clase == "aluminio": arduino.write(b'A')
                    elif nombre_clase == "botella": arduino.write(b'B')
                    elif nombre_clase == "carton": arduino.write(b'C')
                    elif nombre_clase == "organico": arduino.write(b'O')

                cv2.putText(frame, f"DETECTADO: {nombre_clase.upper()}", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Buscando...", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Clasificador 4 Clases", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release()
    cv2.destroyAllWindows()
    if arduino: arduino.close()