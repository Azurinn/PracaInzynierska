import json
import os
from pathlib import Path
import shutil


def convert_supervisely_to_yolo(ann_folder, img_folder, output_folder):
    """
    Konwertuje adnotacje z formatu Supervisely JSON do YOLOv8

    Args:
        ann_folder (str): Ścieżka do folderu z plikami JSON (adnotacje)
        img_folder (str): Ścieżka do folderu ze zdjęciami
        output_folder (str): Ścieżka do folderu wyjściowego
    """

    # Tworzenie struktury folderów dla YOLOv8
    output_path = Path(output_folder)
    labels_path = output_path / "labels"
    images_path = output_path / "images"

    # Tworzenie folderów jeśli nie istnieją
    labels_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)

    # Słownik do mapowania nazw klas na ID
    class_mapping = {}
    class_id = 0

    # Lista wszystkich plików JSON w folderze ann
    ann_files = list(Path(ann_folder).glob("*.json"))

    print(f"Znaleziono {len(ann_files)} plików JSON do przetworzenia")

    # Debugowanie - sprawdźmy jakie pliki obrazów są dostępne
    img_folder_path = Path(img_folder)
    all_images = list(img_folder_path.glob("*"))
    image_extensions = set()
    for img in all_images:
        if img.is_file():
            image_extensions.add(img.suffix.lower())

    print(f"Znalezione rozszerzenia obrazów w folderze {img_folder}: {image_extensions}")
    print(f"Liczba wszystkich plików w folderze obrazów: {len(all_images)}")

    # Tworzenie słownika plików obrazów dla szybszego wyszukiwania
    image_files = {}
    for img_file in img_folder_path.glob("*"):
        if img_file.is_file():
            # Używamy nazwy bez rozszerzenia jako klucza
            name_without_ext = img_file.stem
            image_files[name_without_ext] = img_file

    processed_files = 0
    skipped_files = 0

    for ann_file in ann_files:
        try:
            # Wczytanie pliku JSON
            with open(ann_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Nazwa pliku bez rozszerzenia .json
            json_name = ann_file.stem

            # Usuń '.jpg' z końca nazwy JSON jeśli istnieje
            if json_name.endswith('.jpg'):
                base_name = json_name[:-4]
            else:
                base_name = json_name

            # Sprawdzenie czy odpowiadający plik obrazu istnieje
            img_file = None
            if base_name in image_files:
                img_file = image_files[base_name]

            if img_file is None:
                print(f"Ostrzeżenie: Nie znaleziono pliku obrazu dla {base_name}")
                skipped_files += 1
                continue

            # Pobranie wymiarów obrazu
            img_width = data['size']['width']
            img_height = data['size']['height']

            # Lista linii dla pliku YOLO
            yolo_lines = []

            # Przetwarzanie obiektów (jeśli istnieją)
            if 'objects' in data and data['objects']:
                for obj in data['objects']:
                    class_name = obj.get('classTitle', 'unknown')

                    # Dodanie klasy do mapowania jeśli jeszcze nie istnieje
                    if class_name not in class_mapping:
                        class_mapping[class_name] = class_id
                        class_id += 1

                    # Pobranie geometrii obiektu
                    if 'points' in obj and 'exterior' in obj['points']:
                        points = obj['points']['exterior']

                        if obj['geometryType'] == 'rectangle':
                            # Dla prostokąta - konwersja do formatu YOLO
                            x1, y1 = points[0]
                            x2, y2 = points[1]

                            # Normalizacja współrzędnych
                            center_x = (x1 + x2) / 2 / img_width
                            center_y = (y1 + y2) / 2 / img_height
                            width = abs(x2 - x1) / img_width
                            height = abs(y2 - y1) / img_height

                            yolo_line = f"{class_mapping[class_name]} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                            yolo_lines.append(yolo_line)

                        elif obj['geometryType'] == 'polygon':
                            # Dla wielokąta - konwersja punktów
                            normalized_points = []
                            for point in points:
                                x, y = point
                                norm_x = x / img_width
                                norm_y = y / img_height
                                normalized_points.extend([norm_x, norm_y])

                            points_str = ' '.join([f"{p:.6f}" for p in normalized_points])
                            yolo_line = f"{class_mapping[class_name]} {points_str}"
                            yolo_lines.append(yolo_line)

            # Jeśli nie ma obiektów, ale są tagi (jak w Twoim przykładzie)
            elif 'tags' in data and data['tags']:
                for tag in data['tags']:
                    class_name = tag['name']

                    # Dodanie klasy do mapowania jeśli jeszcze nie istnieje
                    if class_name not in class_mapping:
                        class_mapping[class_name] = class_id
                        class_id += 1

                # Dla tagów bez obiektów, można utworzyć pusty plik lub pominąć
                # W tym przypadku tworzymy pusty plik
                pass

            # Zapisanie pliku YOLO (nawet jeśli pusty)
            yolo_file = labels_path / f"{base_name}.txt"
            with open(yolo_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))

            # Skopiowanie pliku obrazu
            shutil.copy2(img_file, images_path / img_file.name)

            processed_files += 1

        except Exception as e:
            print(f"Błąd podczas przetwarzania {ann_file}: {e}")
            continue

    # Zapisanie mapowania klas
    classes_file = output_path / "classes.txt"
    with open(classes_file, 'w', encoding='utf-8') as f:
        for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{class_id}: {class_name}\n")

    # Utworzenie pliku data.yaml dla YOLOv8
    yaml_content = f"""# YOLOv8 dataset configuration
path: {output_path.absolute()}
train: images
val: images

# Classes
nc: {len(class_mapping)}
names: {list(class_mapping.keys())}
"""

    yaml_file = output_path / "data.yaml"
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\nKonwersja zakończona!")
    print(f"Przetworzono: {processed_files} plików")
    print(f"Pominięto (brak obrazu): {skipped_files} plików")
    print(f"Znalezione klasy: {list(class_mapping.keys())}")
    print(f"Pliki zapisane w: {output_path}")
    print(f"- Obrazy: {images_path}")
    print(f"- Etykiety: {labels_path}")
    print(f"- Mapowanie klas: {classes_file}")
    print(f"- Konfiguracja YOLO: {yaml_file}")

    # Dodatkowe informacje diagnostyczne
    if skipped_files > 0:
        print(f"\nDiagnostyka:")
        print(f"Przykładowe nazwy plików JSON: {[f.name for f in ann_files[:5]]}")
        print(f"Przykładowe nazwy plików obrazów: {list(image_files.keys())[:5]}")


# Przykład użycia
if __name__ == "__main__":
    # Ścieżki do folderów
    ann_folder = "ds/ann"  # Folder z plikami JSON
    img_folder = "ds/img"  # Folder ze zdjęciami
    output_folder = "yolo_dataset"  # Folder wyjściowy

    # Sprawdzenie czy foldery istnieją
    if not os.path.exists(ann_folder):
        print(f"Błąd: Folder {ann_folder} nie istnieje!")
    elif not os.path.exists(img_folder):
        print(f"Błąd: Folder {img_folder} nie istnieje!")
    else:
        # Uruchomienie konwersji
        convert_supervisely_to_yolo(ann_folder, img_folder, output_folder)