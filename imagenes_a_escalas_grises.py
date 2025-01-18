import cv2
import os

def convertir_a_grises(data_path):
    # Recorrer todas las carpetas y subcarpetas
    for root, dirs, files in os.walk(data_path):
        for file_name in files:
            # Construir la ruta completa del archivo
            file_path = os.path.join(root, file_name)
            
            # Verificar si es un archivo de imagen permitido
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Leer la imagen
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Advertencia: No se pudo cargar {file_path}. Ignorando.")
                    continue

                # Convertir la imagen a escala de grises
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Sobrescribir el archivo original con la imagen en escala de grises
                cv2.imwrite(file_path, gray_image)
                print(f"Convertida a escala de grises: {file_path}")
            else:
                # Si no es una imagen válida, se ignora
                print(f"Archivo no válido ignorado: {file_path}")

if __name__ == "__main__":
    DATA_PATH = "Data"
    convertir_a_grises(DATA_PATH)
