import cv2
import serial
import time
import numpy as np
from ultralytics import YOLO

#---Conexion_Arduino---
try:
    arduino = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2)
    print(">>> Conexión con Arduino establecida.")
except:
    print(">>> Arduino no detectado, continuando solo con software.")
    arduino = None

model = YOLO('runs/classify/train/weights/best.pt')

#---Conexion_camara---
print(">>> Iniciamos conexion a camara indice 0.");
id_camara = 1

cap = cv2.VideoCapture(id_camara)

if not cap.isOpened():
    print(">>> Error: No se puede abrir la camara indice 1.");
    print("Iniciamos conexion a camara indice 0.");
    id_camara = 0
    cap = cv2.VideoCapture(id_camara)
    if not cap.isOpened():
        print(">>> Error: No se puede abrir la camara indice 0.");
        exit();
    else:
        print("Conexion establecida con camara indice 0.");
else:
    print("Conexion establecida con camara indice 1.");


#---Ventanas---
UMBRAL_CONFIANZA = 0.70

cv2.namedWindow("1. Vision de Camara", cv2.WINDOW_NORMAL) 
cv2.namedWindow("2. Informacion de Clasificacion", cv2.WINDOW_NORMAL)
cv2.resizeWindow("2. Informacion de Clasificacion", 400, 300)

print(">>> Camara en funcionamiento")

try:
    while True:
        ret, frame = cap.read()
        if not ret: 
            print(">>> Error: No se puede recibir video.")
            break
        
        info_panel = np.zeros((300, 400, 3), dtype=np.uint8)
        
        cv2.putText(info_panel, "Estado: Leyendo imagen...", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        results = model(frame, verbose=False)
        
        if results[0].probs is not None:
            confianza = results[0].probs.top1conf.item()
            
            if confianza >= UMBRAL_CONFIANZA:
                class_id = results[0].probs.top1
                nombre_clase = results[0].names[class_id].upper()
                
                print(f">>> El resultado es: {nombre_clase}")

                # Actualización de Panel de Información
                cv2.putText(info_panel, "RESULTADO:", (20, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(info_panel, nombre_clase, (20, 170), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(info_panel, f"Confianza: {confianza*100:.1f}%", (20, 220), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                if arduino:
                    # Envío de caracteres según la clase detectada
                    comandos = {"ALUMINIO": b'A', "BOTELLA": b'B', "CARTON": b'C', "ORGANICO": b'O'}
                    if nombre_clase in comandos:
                        arduino.write(comandos[nombre_clase])
            else:
                cv2.putText(info_panel, "Buscando objeto...", (20, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

        cv2.imshow("1. Vision de Camara", frame)
        cv2.imshow("2. Informacion de Clasificacion", info_panel)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
    if arduino: arduino.close()
    print(">>> Programa finalizado.")