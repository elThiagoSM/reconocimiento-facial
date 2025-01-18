import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def cargar_datos(data_path):
    labels = []
    faces_data = []
    label = 0
    people_list = os.listdir(data_path)
    print(f"Personas encontradas: {people_list}")
    
    for person_name in people_list:
        person_path = os.path.join(data_path, person_name)
        for file_name in os.listdir(person_path):
            if not file_name.lower().endswith(('.jpg', '.png', '.jpeg')):  # Ignorar archivos no imagen
                continue
            file_path = os.path.join(person_path, file_name)
            image = cv2.imread(file_path, 0)  # Cargar imagen en escala de grises
            if image is None:  # Ignorar imágenes corruptas
                print(f"Advertencia: No se pudo cargar {file_path}. Ignorando.")
                continue
            # Redimensionar la imagen al tamaño esperado (150x150)
            image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)
            faces_data.append(image)
            labels.append(label)
        label += 1
    return np.array(faces_data), np.array(labels), people_list

def entrenar_modelo(data_path, model_path):
    faces_data, labels, people_list = cargar_datos(data_path)
    if len(faces_data) == 0 or len(labels) == 0:
        print("Error: No se encontraron datos válidos para entrenar el modelo.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(faces_data, labels, test_size=0.2, random_state=42)
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Iniciando entrenamiento...")

    # Entrenar el modelo usando todos los datos de entrenamiento
    face_recognizer.train(X_train, y_train)
    
    # Guardar el modelo entrenado
    face_recognizer.write(model_path)
    print(f"Modelo almacenado en {model_path}")
    
    # Evaluación básica
    print("Iniciando evaluación...")
    correctos = 0
    for i in tqdm(range(len(y_test)), desc="Evaluando el modelo"):
        prediccion, _ = face_recognizer.predict(X_test[i])
        if prediccion == y_test[i]:
            correctos += 1
    print(f"Precisión en el conjunto de prueba: {correctos / len(y_test) * 100:.2f}%")

if __name__ == "__main__":
    DATA_PATH = "Data"
    MODEL_PATH = "modeloLBPHFace.xml"
    entrenar_modelo(DATA_PATH, MODEL_PATH)
