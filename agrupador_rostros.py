import os
import cv2
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
import json

def cargar_modelo_y_metadatos(model_path, data_path):
    """Carga el modelo entrenado y los metadatos asociados."""
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model_path)

    people_data = {}
    for person_name in os.listdir(data_path):
        metadata_file = os.path.join(data_path, person_name, "metadata.json")
        if os.path.isfile(metadata_file):
            with open(metadata_file, "r") as file:
                people_data[person_name] = json.load(file)

    return face_recognizer, people_data

def procesar_frame(frame, face_recognizer, people_data, rostro_embeddings, rostro_labels, threshold):
    """Procesa un frame para detectar y agrupar rostros."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detecciones = DeepFace.detectFace(rgb_frame, detector_backend="opencv", enforce_detection=False)

    if detecciones is None:
        return

    for detection in detecciones:
        x, y, w, h = detection["box"]
        rostro = frame[y:y+h, x:x+w]
        rostro_gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
        rostro_resized = cv2.resize(rostro_gray, (150, 150))

        label, confidence = face_recognizer.predict(rostro_resized)
        etiqueta = list(people_data.keys())[label] if confidence < 70 else None

        rostro_embedding = DeepFace.represent(rostro, model_name="Facenet", enforce_detection=False)
        if rostro_embedding is None:
            continue

        matches = np.linalg.norm(np.array(rostro_embeddings) - rostro_embedding, axis=1) < threshold

        if any(matches):
            match_index = np.argmin(np.linalg.norm(np.array(rostro_embeddings) - rostro_embedding, axis=1))
            folder_name = rostro_labels[match_index]
        else:
            folder_name = etiqueta or f"Persona_{len(set(rostro_labels)) + 1}"
            rostro_embeddings.append(rostro_embedding)
            rostro_labels.append(folder_name)

        return folder_name, x, y, w, h

def guardar_rostro(frame, output_folder, folder_name, bbox):
    """Guarda el rostro detectado en la carpeta correspondiente."""
    x, y, w, h = bbox
    output_path = os.path.join(output_folder, folder_name)
    os.makedirs(output_path, exist_ok=True)

    rostro_filename = f"{folder_name}_{len(os.listdir(output_path)) + 1}.jpg"
    rostro_path = os.path.join(output_path, rostro_filename)
    cv2.imwrite(rostro_path, frame[y:y+h, x:x+w])

def procesar_video(video_path, output_folder, face_recognizer, people_data, threshold):
    """Procesa un video para recolectar y agrupar rostros."""
    cap = cv2.VideoCapture(video_path)
    rostro_embeddings = []
    rostro_labels = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue

        resultado = procesar_frame(frame, face_recognizer, people_data, rostro_embeddings, rostro_labels, threshold)
        if resultado:
            folder_name, x, y, w, h = resultado
            guardar_rostro(frame, output_folder, folder_name, (x, y, w, h))

    cap.release()

def agrupar_rostros(input_folder, output_folder, model_path, data_path, threshold=0.6):
    """Agrupa rostros en imágenes y videos del directorio de entrada."""
    os.makedirs(output_folder, exist_ok=True)

    # Cargar modelo entrenado y metadatos
    face_recognizer, people_data = cargar_modelo_y_metadatos(model_path, data_path)

    for item_name in tqdm(os.listdir(input_folder), desc="Procesando archivos"):
        item_path = os.path.join(input_folder, item_name)

        if os.path.isdir(item_path):
            continue

        if item_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Procesar imágenes
            image = cv2.imread(item_path)
            if image is None:
                print(f"Advertencia: no se pudo cargar {item_path}")
                continue

            resultado = procesar_frame(image, face_recognizer, people_data, [], [], threshold)
            if resultado:
                folder_name, x, y, w, h = resultado
                guardar_rostro(image, output_folder, folder_name, (x, y, w, h))

        elif item_name.lower().endswith('.mp4'):
            # Procesar videos
            procesar_video(item_path, output_folder, face_recognizer, people_data, threshold)

if __name__ == "__main__":
    INPUT_FOLDER = "imagenes_entrada"
    OUTPUT_FOLDER = "rostros_agrupados"
    MODEL_PATH = "modeloLBPHFace.xml"
    DATA_PATH = "Data"

    agrupar_rostros(INPUT_FOLDER, OUTPUT_FOLDER, MODEL_PATH, DATA_PATH)
