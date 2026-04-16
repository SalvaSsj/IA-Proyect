from ultralytics import YOLO

# 1. Carga de modelo pre-entrenado siendo Nano para ser rapico
model = YOLO('yolov8n-cls.pt') # 'cls' indica que es para clasificación

# 2. Entrenar el modelo
# data: Ruta donde guardara la informacion.
# epochs: Numero de veces que vera el modelo.
resultado = model.train(data='dataset', epochs=20, imgsz=224)

print("Entrenamiento completado.")