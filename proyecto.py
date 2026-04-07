import cv2
import serial
import time
from ultralytics import YOLO

# --- CONFIGURACIÓN INICIAL ---
# Intento de conexión con Arduino
try:
    # Asegúrate de que el puerto COM coincida con el de tu Arduino IDE
    arduino = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2) 
    print(">>> Conexión con Arduino establecida.")
except Exception as e:
    print(f">>> Error al conectar Arduino: {e}")
    arduino = None

# Cargar el modelo entrenado (el archivo .pt que generaste)
# Si aún no lo tienes, puedes usar 'yolov8n-cls.pt' para pruebas iniciales

model = YOLO('runs/classify/train/weights/best.pt')

# Inicializar cámara
cap = cv2.VideoCapture(0)

if cap.isOpened():
    print(">>> Camara en funcionamiento") # Mensaje clave 1
else:
    print(">>> Error: No se pudo acceder a la cámara")
    exit()

# --- BUCLE PRINCIPAL ---
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(">>> Leyendo imagen") # Mensaje clave 2

        # Predicción con el modelo
        results = model(frame, verbose=False) # verbose=False para limpiar la terminal
        
        if results[0].probs is not None:
            # Obtener el índice de la clase con mayor confianza
            class_id = results[0].probs.top1
            # Obtener el nombre de la clase (Orgánico, Inorgánico, etc.)
            resultado_texto = results[0].names[class_id]

            print(f">>> El resultado es: {resultado_texto}") # Mensaje clave 3

            # --- LÓGICA DE CONTROL MECÁNICO ---
            if arduino:
                if resultado_texto == "organico":
                    arduino.write(b'O') # Envía señal para abrir
                else:
                    arduino.write(b'I') # Envía señal para cerrar/mantener

        # (Opcional) Mostrar la imagen para referencia visual
        cv2.imshow("Preview - Presiona 'q' para salir", frame)

        # Pequeña pausa para no saturar la terminal y el procesador
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

finally:
    # Limpieza de recursos
    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()
    print(">>> Programa finalizado.")