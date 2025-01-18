import os
import json
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
import cv2
from PIL import Image

def cargar_modelo(model_path, data_path):
    people_data = {}

    # Reconstruir el modelo
    num_classes = len(os.listdir(data_path))  # Número de clases (carpetas en Data)
    classifier = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(256, num_classes)
    )
    classifier.load_state_dict(torch.load(model_path))  # Cargar los pesos
    classifier.eval()  # Poner el modelo en modo evaluación

    # Cargar información de las personas
    for person_name in os.listdir(data_path):
        person_path = os.path.join(data_path, person_name)
        metadata_file = os.path.join(person_path, "metadata.json")

        if os.path.isfile(metadata_file):  # Leer solo si existe el archivo JSON
            with open(metadata_file, "r") as file:
                metadata = json.load(file)
                people_data[person_name] = metadata

    return classifier, people_data

def procesar_frame(frame, classifier, people_data):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    model = InceptionResnetV1(pretrained='vggface2').eval()  # Modelo preentrenado para embeddings

    for (x, y, w, h) in faces:
        rostro = frame[y:y+h, x:x+w]

        try:
            # Preprocesar rostro
            image = Image.fromarray(cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)).resize((160, 160))
            image = np.asarray(image) / 255.0
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

            # Extraer embeddings y predecir
            with torch.no_grad():
                embedding = model(image).squeeze().unsqueeze(0)  # (1, 512)
                outputs = classifier(embedding)
                _, predicted = torch.max(outputs, 1)

            confidence = torch.softmax(outputs, dim=1).max().item()
            person_name = list(people_data.keys())[predicted.item()]
            person_info = people_data.get(person_name, {})

            if confidence > 0.7:
                name_display = person_info.get("nombre_completo", person_name)
                details = [
                    f"Tel: {person_info.get('telefono', 'N/A')}",
                    f"Dirección: {person_info.get('direccion', 'N/A')}",
                    f"Profesión: {person_info.get('profesion', 'N/A')}",
                    f"Sueldo: {person_info.get('sueldo', 'N/A')}",
                    f"Email: {person_info.get('email', 'N/A')}",
                    f"DNI: {person_info.get('dni', 'N/A')}"
                ]
                color = (0, 255, 0)  # Verde para conocidos
            else:
                name_display = "Desconocido"
                details = ["No disponible"]
                color = (0, 0, 255)  # Rojo para desconocidos

            # Mostrar nombre y detalles junto al rostro
            cv2.putText(frame, f"{name_display} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            for i, detail in enumerate(details):
                cv2.putText(frame, detail, (x, y + h + 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        except Exception as e:
            print(f"Error procesando rostro: {e}")

    return frame

def reconocer_rostros(model_path, data_path):
    classifier, people_data = cargar_modelo(model_path, data_path)
    option = input("Selecciona una opción:\n1. Usar cámara en tiempo real\n2. Proporcionar ruta de una imagen\n> ")

    if option == "1":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = procesar_frame(frame, classifier, people_data)
            cv2.imshow("Reconocimiento Facial", frame)

            if cv2.waitKey(1) == 27:  # Tecla ESC
                break

        cap.release()

    elif option == "2":
        image_path = input("Ingresa la ruta de la imagen: ")
        if os.path.isfile(image_path):
            frame = cv2.imread(image_path)
            frame = procesar_frame(frame, classifier, people_data)

            # Mostrar la imagen procesada
            cv2.imshow("Reconocimiento Facial", frame)

            # Guardar la imagen procesada
            output_path = f"./resultado_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, frame)
            print(f"Imagen procesada guardada en: {output_path}")

            cv2.waitKey(0)
        else:
            print("La ruta de la imagen no es válida.")

    else:
        print("Opción no válida.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = "facenet_classifier.pth"
    DATA_PATH = "Data"
    reconocer_rostros(MODEL_PATH, DATA_PATH)
