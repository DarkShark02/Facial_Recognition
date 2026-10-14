import argparse
import os
import re
import sys
import time

import cv2
import face_recognition
import numpy as np

# =========================================
# CONFIGURAZIONE
# =========================================

KNOWN_FACES_DIR = "faces"

TOLERANCE = 0.6
MODEL = "hog"
CAMERA_INDEX = 0

SCALE_FACTOR = 0.25
MAX_SCAN_FRAMES = 30

AUTO_SAVE_UNKNOWN = True
UNKNOWN_BUFFER_SIZE = 10

# margine aggiunto attorno al volto salvato (frazione della dimensione del volto),
# serve perché il rilevatore trovi di nuovo il volto quando la foto viene ricaricata
CROP_MARGIN = 0.6

MAX_NAME_LENGTH = 30

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

UNKNOWN_LABEL = "Sconosciuto"
WINDOW_NAME = "Face Recognition System"

KEY_ENTER = (13, 10)
KEY_BACKSPACE = (8, 127)
KEY_ESC = 27

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)

# =========================================
# DATABASE VOLTI
# =========================================

def load_known_faces(faces_dir):
    """
    Carica i volti noti. Sono supportati due formati:
      faces/<nome>.jpg            -> una foto per persona
      faces/<nome>/<qualsiasi>.jpg -> più foto per la stessa persona
    """

    encodings = []
    names = []

    os.makedirs(faces_dir, exist_ok=True)

    print("Caricamento volti...")

    for entry in sorted(os.listdir(faces_dir)):

        if entry.startswith("."):
            continue

        path = os.path.join(faces_dir, entry)

        if os.path.isdir(path):
            person_name = entry
            image_paths = [
                os.path.join(path, f)
                for f in sorted(os.listdir(path))
                if is_image(f)
            ]
        elif is_image(entry):
            person_name = os.path.splitext(entry)[0]
            image_paths = [path]
        else:
            continue

        loaded = 0

        for image_path in image_paths:

            try:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
            except Exception as e:
                print(f"Errore caricamento {image_path}: {e}")
                continue

            if len(face_encodings) == 0:
                print(f"Nessun volto trovato in {image_path}")
                continue

            encodings.append(face_encodings[0])
            names.append(person_name)
            loaded += 1

        if loaded > 0:
            print(f"Caricato: {person_name} ({loaded} foto)")

    print(f"Volti caricati: {len(set(names))} persone, {len(encodings)} foto.")

    return encodings, names


def is_image(file_name):
    return (
        not file_name.startswith(".")
        and os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS
    )


def sanitize_name(name):
    """Tiene solo lettere, numeri, spazi, '-' e '_' così il nome è sicuro come nome di cartella."""

    name = re.sub(r"[^A-Za-z0-9 _-]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name[:MAX_NAME_LENGTH]


def crop_with_margin(frame, face_box, margin):

    top, right, bottom, left = face_box

    height, width = frame.shape[:2]

    margin_y = int((bottom - top) * margin)
    margin_x = int((right - left) * margin)

    top = max(0, top - margin_y)
    bottom = min(height, bottom + margin_y)
    left = max(0, left - margin_x)
    right = min(width, right + margin_x)

    return frame[top:bottom, left:right]


def save_image(path, image):
    """
    Come cv2.imwrite, ma funziona anche con percorsi che contengono
    caratteri non ASCII (cv2.imwrite su Windows fallisce in quel caso).
    """

    ok, buffer = cv2.imencode(os.path.splitext(path)[1], image)

    if not ok:
        return False

    try:
        buffer.tofile(path)
    except OSError:
        return False

    return True


def register_new_face(faces_dir, name, frame, encoding, face_box):
    """Salva la foto in faces/<nome>/<timestamp>.jpg. Restituisce True se è andato a buon fine."""

    face_crop = crop_with_margin(frame, face_box, CROP_MARGIN)

    if face_crop.size == 0:
        print("Errore crop volto")
        return False

    person_dir = os.path.join(faces_dir, name)

    # un vecchio faces/<nome>.jpg viene spostato nella cartella della persona
    for ext in IMAGE_EXTENSIONS:
        legacy_path = os.path.join(faces_dir, name + ext)
        if os.path.isfile(legacy_path):
            os.makedirs(person_dir, exist_ok=True)
            os.replace(legacy_path, os.path.join(person_dir, "1" + ext))

    os.makedirs(person_dir, exist_ok=True)

    file_path = os.path.join(
        person_dir,
        time.strftime("%Y%m%d_%H%M%S") + ".jpg"
    )

    if not save_image(file_path, face_crop):
        print(f"Errore salvataggio {file_path}")
        return False

    print(f"Registrato: {name} -> {file_path}")

    return True

# =========================================
# APPLICAZIONE
# =========================================

class FaceRecognitionApp:

    def __init__(self, faces_dir, tolerance, model):

        self.faces_dir = faces_dir
        self.tolerance = tolerance
        self.model = model

        self.known_face_encodings, self.known_face_names = load_known_faces(faces_dir)

        self.recognition_active = False
        self.register_mode = False
        self.running = True

        self.processing_frame_count = 0

        self.face_locations = []
        self.face_names = []

        self.unknown_face_buffer = []

        self.typed_name = ""

        self.unknown_face_image = None
        self.unknown_face_encoding = None
        self.unknown_face_box = None

        self.set_status("Premi S per avviare", COLOR_CYAN)

    def set_status(self, message, color):
        self.status_message = message
        self.status_color = color

    def reset_scan(self):
        self.recognition_active = False
        self.processing_frame_count = 0
        self.unknown_face_buffer = []
        self.face_locations = []
        self.face_names = []

    # =====================================
    # SCANSIONE
    # =====================================

    def start_scan(self):

        self.reset_scan()

        self.recognition_active = True

        self.set_status("Scansione in corso...", COLOR_CYAN)

        print("\nScansione avviata")

    def identify(self, encoding):

        if len(self.known_face_encodings) == 0:
            return UNKNOWN_LABEL

        distances = face_recognition.face_distance(
            self.known_face_encodings,
            encoding
        )

        best_match_index = int(np.argmin(distances))

        if distances[best_match_index] <= self.tolerance:
            return self.known_face_names[best_match_index]

        return UNKNOWN_LABEL

    def add_to_unknown_buffer(self, encoding, face_box):
        """
        Il buffer deve contenere sempre la stessa persona: se arriva un volto
        diverso dal primo del buffer, si ricomincia da capo.
        """

        if self.unknown_face_buffer:

            first_encoding = self.unknown_face_buffer[0][0]

            distance = face_recognition.face_distance([first_encoding], encoding)[0]

            if distance > self.tolerance:
                self.unknown_face_buffer = []

        self.unknown_face_buffer.append((encoding, face_box))

    def process_frame(self, frame):

        self.processing_frame_count += 1

        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=SCALE_FACTOR,
            fy=SCALE_FACTOR
        )

        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(
            rgb_small,
            model=self.model
        )

        face_encodings = face_recognition.face_encodings(
            rgb_small,
            face_locations
        )

        scale = 1 / SCALE_FACTOR

        self.face_locations = []
        self.face_names = []

        recognized = []
        unknown = []

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):

            face_box = (
                int(top * scale),
                int(right * scale),
                int(bottom * scale),
                int(left * scale)
            )

            name = self.identify(encoding)

            self.face_locations.append(face_box)
            self.face_names.append(name)

            if name == UNKNOWN_LABEL:
                unknown.append((encoding, face_box))
            else:
                recognized.append(name)

        # =================================
        # VOLTO NOTO TROVATO
        # =================================

        if recognized:

            names = ", ".join(dict.fromkeys(recognized))

            print(f"Riconosciuto: {names}")

            self.reset_scan()

            self.set_status(f"Riconosciuto: {names}", COLOR_GREEN)

            return

        # =================================
        # REGISTRAZIONE AUTOMATICA
        # =================================

        # con più volti sconosciuti nell'inquadratura non si sa chi registrare
        if len(unknown) == 1:
            self.add_to_unknown_buffer(*unknown[0])

        if AUTO_SAVE_UNKNOWN and len(self.unknown_face_buffer) >= UNKNOWN_BUFFER_SIZE:

            print("Volto sconosciuto stabile rilevato")

            encodings = [e for e, _ in self.unknown_face_buffer]

            self.unknown_face_image = frame.copy()
            self.unknown_face_encoding = np.mean(encodings, axis=0)
            self.unknown_face_box = self.unknown_face_buffer[-1][1]

            self.reset_scan()

            self.register_mode = True
            self.typed_name = ""

            self.set_status("Inserisci nome e premi ENTER (ESC annulla)", COLOR_YELLOW)

            return

        # =================================
        # TIMEOUT
        # =================================

        if self.processing_frame_count >= MAX_SCAN_FRAMES:

            if unknown or self.unknown_face_buffer:
                message = "Nessun volto noto rilevato"
            else:
                message = "Nessun volto rilevato"

            print(message)

            self.reset_scan()

            self.set_status(message, COLOR_RED)

    # =====================================
    # REGISTRAZIONE
    # =====================================

    def confirm_registration(self):

        name = sanitize_name(self.typed_name)

        if name == "":

            self.set_status("Nome non valido", COLOR_RED)

        elif register_new_face(
            self.faces_dir,
            name,
            self.unknown_face_image,
            self.unknown_face_encoding,
            self.unknown_face_box
        ):

            self.known_face_encodings.append(self.unknown_face_encoding)
            self.known_face_names.append(name)

            self.set_status(f"Registrato: {name}", COLOR_GREEN)

        else:

            self.set_status("Errore durante il salvataggio", COLOR_RED)

        self.exit_register_mode()

    def cancel_registration(self):

        print("Registrazione annullata")

        self.set_status("Registrazione annullata", COLOR_CYAN)

        self.exit_register_mode()

    def exit_register_mode(self):

        self.register_mode = False
        self.typed_name = ""

        self.unknown_face_image = None
        self.unknown_face_encoding = None
        self.unknown_face_box = None

        self.reset_scan()

    # =====================================
    # TASTIERA
    # =====================================

    def handle_key(self, key):

        if key == 255:
            return

        if self.register_mode:

            if key in KEY_ENTER:
                self.confirm_registration()

            elif key == KEY_ESC:
                self.cancel_registration()

            elif key in KEY_BACKSPACE:
                self.typed_name = self.typed_name[:-1]

            elif 32 <= key <= 126 and len(self.typed_name) < MAX_NAME_LENGTH:
                self.typed_name += chr(key)

            return

        if key in (ord("q"), ord("Q"), KEY_ESC):
            self.running = False

        elif key in (ord("s"), ord("S")) and not self.recognition_active:
            self.start_scan()

    # =====================================
    # DISEGNO
    # =====================================

    def draw(self, frame):

        cv2.putText(
            frame,
            self.status_message,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.status_color,
            2
        )

        cv2.putText(
            frame,
            "S = scan | Q = quit",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_WHITE,
            1
        )

        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):

            color = COLOR_RED if name == UNKNOWN_LABEL else COLOR_GREEN

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            cv2.putText(
                frame,
                name,
                (left, max(15, top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_WHITE,
                1
            )

        if self.register_mode:

            cv2.rectangle(frame, (20, 60), (frame.shape[1] - 20, 120), (40, 40, 40), -1)

            cv2.putText(
                frame,
                "Inserisci nome:",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR_WHITE,
                2
            )

            cv2.putText(
                frame,
                self.typed_name + "_",
                (30, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLOR_YELLOW,
                2
            )

    # =====================================
    # LOOP PRINCIPALE
    # =====================================

    def run(self, camera_index):

        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"Impossibile aprire la webcam (indice {camera_index})")
            return 1

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\nPremi:")
        print("S -> avvia scansione")
        print("Q -> esci")

        try:

            while self.running:

                ret, frame = cap.read()

                if not ret:
                    print("Errore lettura webcam")
                    break

                # effetto specchio
                frame = cv2.flip(frame, 1)

                if self.recognition_active:
                    self.process_frame(frame)

                self.draw(frame)

                cv2.imshow(WINDOW_NAME, frame)

                self.handle_key(cv2.waitKey(1) & 0xFF)

                # chiusura con la X della finestra
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

        finally:

            cap.release()
            cv2.destroyAllWindows()

        return 0

# =========================================
# MAIN
# =========================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Riconoscimento facciale in tempo reale da webcam."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help=f"indice della webcam (default: {CAMERA_INDEX})"
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=TOLERANCE,
        help=f"soglia di riconoscimento, più bassa = più severa (default: {TOLERANCE})"
    )

    parser.add_argument(
        "--model",
        choices=["hog", "cnn"],
        default=MODEL,
        help=f"modello di rilevamento volti (default: {MODEL})"
    )

    parser.add_argument(
        "--faces-dir",
        default=KNOWN_FACES_DIR,
        help=f"cartella dei volti noti (default: {KNOWN_FACES_DIR})"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    app = FaceRecognitionApp(args.faces_dir, args.tolerance, args.model)

    return app.run(args.camera)


if __name__ == "__main__":
    sys.exit(main())
