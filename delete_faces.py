import os


def delete_images_in_folder(folder_path):
    # Definiere die Liste der Bilddateiendungen
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg']

    # Zähle die gelöschten Dateien
    deleted_files_count = 0

    # Überprüfe, ob der angegebene Ordner existiert
    if not os.path.isdir(folder_path):
        print(f"Der Ordner {folder_path} existiert nicht.")
        return

    # Durchlaufe alle Dateien im Ordner
    for filename in os.listdir(folder_path):
        # Hole die Dateiendung
        file_extension = os.path.splitext(filename)[1].lower()

        # Überprüfe, ob die Datei eine Bilddatei ist
        if file_extension in image_extensions:
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f"{filename} wurde gelöscht.")
                deleted_files_count += 1
            except Exception as e:
                print(f"Konnte {filename} nicht löschen: {e}")

    if deleted_files_count == 0:
        print("Es wurden keine Bilddateien gefunden.")
    else:
        print(f"Insgesamt wurden {deleted_files_count} Bilddateien gelöscht.")


# Beispielaufruf der Funktion
ordner_pfad = 'src/generic_pipeline/data/faces'
delete_images_in_folder(ordner_pfad)
