import os
import subprocess

def convertir_heic_a_jpg_en_misma_carpeta(folder_path):
    """Convierte archivos HEIC a JPG en la misma carpeta y elimina los archivos HEIC."""
    # Validar si la carpeta existe
    if not os.path.exists(folder_path):
        print(f"La carpeta {folder_path} no existe.")
        return
    
    # Recorrer todos los archivos en la carpeta
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".heic"):
            # Rutas de entrada y salida
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(folder_path, os.path.splitext(filename)[0] + ".jpg")
            
            # Comando de conversión usando ImageMagick
            command = f'magick "{input_path}" "{output_path}"'
            try:
                # Ejecutar el comando
                subprocess.run(command, shell=True, check=True)
                print(f"Convertido: {input_path} -> {output_path}")
                
                # Intentar eliminar el archivo HEIC original
                try:
                    os.remove(input_path)
                    print(f"Eliminado archivo original: {input_path}")
                except OSError as e:
                    print(f"No se pudo eliminar el archivo {input_path}: {e}")
            except subprocess.CalledProcessError as e:
                print(f"Error al convertir {input_path}: {e}")

# Carpeta de entrada (puedes cambiarla según tu caso)
input_folder = "imagenes_entrada"

# Ejecutar la conversión
convertir_heic_a_jpg_en_misma_carpeta(input_folder)
