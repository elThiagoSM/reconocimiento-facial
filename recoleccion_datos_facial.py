import cv2
import os
import json
import imutils
from datetime import datetime


def crear_directorio(data_path, person_name=None):
    if not person_name:
        person_name = input("Introduce el nombre de la persona: ")

    # Contar cuántas carpetas ya existen con el mismo nombre
    existing_dirs = [d for d in os.listdir(data_path) if d.startswith(person_name)]
    count = len(existing_dirs)
    unique_name = f"{person_name}_{count + 1}" if count > 0 else person_name

    person_path = os.path.join(data_path, unique_name)
    if not os.path.exists(person_path):
        os.makedirs(person_path)
        print(f"Carpeta creada: {person_path}")
    else:
        print(f"Carpeta existente: {person_path}")
    return person_path, unique_name


def guardar_metadatos(person_path, person_name):
    metadata_file = os.path.join(person_path, "metadata.json")

    print("Introduce los datos adicionales para esta persona:")
    nombre_completo = input("Nombre completo: ")
    direccion = input("Dirección: ")
    telefono = input("Teléfono: ")
    profesion = input("Profesión: ")
    sueldo = input("Sueldo estimado: ")
    email = input("Correo electrónico: ")
    dni = input("DNI: ")

    # Crear un ID único (basado en el timestamp)
    person_id = f"{person_name}_{int(datetime.now().timestamp())}"

    metadata = {
        "id": person_id,
        "nombre_completo": nombre_completo,
        "direccion": direccion,
        "telefono": telefono,
        "profesion": profesion,
        "sueldo": sueldo,
        "email": email,
        "dni": dni,
        "fecha_registro": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notas": input("Notas adicionales (opcional): ")
    }

    with open(metadata_file, "w") as file:
        json.dump(metadata, file, indent=4)
    print(f"Metadatos guardados en {metadata_file}")


def seleccionar_fuente():
    opcion = input("Selecciona la fuente de video:\n1. Cámara web\n2. Archivo de video\nOpción: ")
    if opcion == "1":
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)
    elif opcion == "2":
        ruta_video = input("Introduce la ruta del archivo de video: ")
        return cv2.VideoCapture(ruta_video)
    else:
        print("Opción no válida. Usando cámara web por defecto.")
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)


def obtener_contador_inicial(person_path):
    existing_files = [f for f in os.listdir(person_path) if f.startswith("rostro_") and f.endswith(".jpg")]
    if existing_files:
        max_count = max(int(f.split("_")[1].split(".")[0]) for f in existing_files)
        return max_count + 1
    return 0


def recolectar_datos(data_path):
    opcion = input("¿Deseas crear un nuevo rostro o agregar a uno existente? (nuevo/existente): ").strip().lower()
    if opcion == "nuevo":
        person_path, person_name = crear_directorio(data_path)
        guardar_metadatos(person_path, person_name)
    elif opcion == "existente":
        person_name = input("Introduce el nombre de la persona existente: ")
        person_path = os.path.join(data_path, person_name)
        if not os.path.exists(person_path):
            print("La persona no existe. Creando nueva carpeta.")
            person_path, person_name = crear_directorio(data_path, person_name)
            guardar_metadatos(person_path, person_name)
    else:
        print("Opción no válida. Saliendo.")
        return

    cap = seleccionar_fuente()
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = obtener_contador_inicial(person_path)
    print(f"Iniciando desde la imagen número {count}.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rostro = cv2.resize(gray[y:y+h, x:x+w], (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{person_path}/rostro_{count}.jpg", rostro)
            print(f"Imagen guardada: {person_path}/rostro_{count}.jpg")
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:  # Tecla ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    DATA_PATH = "Data"
    recolectar_datos(DATA_PATH)
