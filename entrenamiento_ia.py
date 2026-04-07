from ultralytics import YOLO

# 1. Cargamos un modelo pre-entrenado pequeño (Nano) para que sea rápido
model = YOLO('yolov8n-cls.pt') # 'cls' indica que es para clasificación

# 2. Entrenar el modelo
# data: ruta a tu carpeta de dataset
# epochs: cuántas veces el modelo verá las fotos (ajusta según necesites)
results_inorganico = model.train(data='dataset', epochs=20, imgsz=224)

print("Entrenamiento completado.")