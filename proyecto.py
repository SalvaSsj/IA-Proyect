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

# Ruta de tu modelo
model = YOLO('runs/classify/train/weights/best.pt')

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Umbral de confianza (0.7 = 70%)
# Esto evita que la IA "adivine" cuando no está segura
UMBRAL_CONFIANZA = 0.7 

if not cap.isOpened():
    print(">>> Error: No se pudo acceder a la cámara")
    exit()

print(">>> Sistema de clasificación iniciado. Presiona 'q' para salir.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predicción
        # results[0].probs contiene las probabilidades de las 5 clases
        results = model(frame, verbose=False)
        
        if results[0].probs is not None:
            # 1. Obtener la confianza de la mejor predicción
            confianza = results[0].probs.top1conf.item()
            
            # 2. Solo procesar si supera nuestro umbral
            if confianza >= UMBRAL_CONFIANZA:
                class_id = results[0].probs.top1
                nombre_clase = results[0].names[class_id]
                
                # Formatear el texto para mostrar
                texto_display = f"Clase: {nombre_clase.upper()} ({confianza*100:.1f}%)"
                
                # Imprimir en consola
                print(f">>> Detectado: {texto_display}")
                
                # 3. Dibujar el resultado en la ventana de video
                cv2.putText(frame, texto_display, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Buscando objetos...", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Mostrar la imagen
        cv2.imshow("Clasificador de Residuos", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(">>> Programa finalizado.")