import os
import cv2
import face_recognition
import numpy as np
from tqdm import tqdm
import json

def cargar_modelo(model_path, data_path):
    """Carga el modelo entrenado y los metadatos asociados."""
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model_path)
    
    people_data = {}
    for person_name in os.listdir(data_path):
        person_path = os.path.join(data_path, person_name)
        metadata_file = os.path.join(person_path, "metadata.json")
        if os.path.isfile(metadata_file):
            with open(metadata_file, "r") as file:
                people_data[person_name] = json.load(file)
    return face_recognizer, people_data

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
        if frame_count % 5 != 0:  # Procesar cada 5 cuadros para optimizar
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            rostro_gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            rostro_resized = cv2.resize(rostro_gray, (150, 150), interpolation=cv2.INTER_CUBIC)

            label, confidence = face_recognizer.predict(rostro_resized)

            if confidence < 70:  # Umbral de confianza para rostros conocidos
                etiqueta = list(people_data.keys())[label]
            else:
                etiqueta = None

            matches = face_recognition.compare_faces(rostro_embeddings, face_encoding, tolerance=threshold)
            distances = face_recognition.face_distance(rostro_embeddings, face_encoding)

            if any(matches):
                match_index = np.argmin(distances)
                folder_name = rostro_labels[match_index]
            else:
                if etiqueta:
                    folder_name = etiqueta
                else:
                    folder_name = f"Persona_{len(set(rostro_labels)) + 1}"

                rostro_embeddings.append(face_encoding)
                rostro_labels.append(folder_name)

            output_path = os.path.join(output_folder, folder_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            rostro_filename = f"{folder_name}_{len(os.listdir(output_path)) + 1}.jpg"
            rostro_path = os.path.join(output_path, rostro_filename)
            cv2.imwrite(rostro_path, frame[top:bottom, left:right])

    cap.release()

def agrupar_rostros_con_modelo(input_folder, output_folder, model_path, data_path, threshold=0.6):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Inicializar las variables para embeddings y etiquetas
    rostro_embeddings = []
    rostro_labels = []

    # Cargar modelo entrenado y metadatos
    face_recognizer, people_data = cargar_modelo(model_path, data_path)

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

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                rostro_gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
                rostro_resized = cv2.resize(rostro_gray, (150, 150), interpolation=cv2.INTER_CUBIC)

                label, confidence = face_recognizer.predict(rostro_resized)

                if confidence < 70:
                    etiqueta = list(people_data.keys())[label]
                else:
                    etiqueta = None

                matches = face_recognition.compare_faces(rostro_embeddings, face_encoding, tolerance=threshold)
                distances = face_recognition.face_distance(rostro_embeddings, face_encoding)

                if any(matches):
                    match_index = np.argmin(distances)
                    folder_name = rostro_labels[match_index]
                else:
                    if etiqueta:
                        folder_name = etiqueta
                    else:
                        folder_name = f"Persona_{len(set(rostro_labels)) + 1}"

                    rostro_embeddings.append(face_encoding)
                    rostro_labels.append(folder_name)

                output_path = os.path.join(output_folder, folder_name)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                rostro_filename = f"{folder_name}_{len(os.listdir(output_path)) + 1}.jpg"
                rostro_path = os.path.join(output_path, rostro_filename)
                cv2.imwrite(rostro_path, image[top:bottom, left:right])

        elif item_name.lower().endswith('.mp4'):
            # Procesar videos
            procesar_video(item_path, output_folder, face_recognizer, people_data, threshold)

if __name__ == "__main__":
    INPUT_FOLDER = "imagenes_entrada"
    OUTPUT_FOLDER = "rostros_agrupados"
    MODEL_PATH = "modeloLBPHFace.xml"
    DATA_PATH = "Data"

    agrupar_rostros_con_modelo(INPUT_FOLDER, OUTPUT_FOLDER, MODEL_PATH, DATA_PATH)
