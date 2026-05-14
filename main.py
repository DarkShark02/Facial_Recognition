import cv2
import face_recognition
import os
import sys
import numpy as np
import time

# =========================================
# CONFIGURAZIONE
# =========================================

KNOWN_FACES_DIR = "faces"

TOLERANCE = 0.6
MODEL = "hog"

SCALE_FACTOR = 0.25
MAX_SCAN_FRAMES = 30

AUTO_SAVE_UNKNOWN = True
UNKNOWN_BUFFER_SIZE = 10

# =========================================
# VARIABILI GLOBALI
# =========================================

known_face_encodings = []
known_face_names = []

recognition_active = False
register_mode = False

processing_frame_count = 0

last_face_locations = []
last_face_names = []

status_message = "Premi S per avviare"
status_color = (255, 255, 0)

last_recognized = None

unknown_face_buffer = []

typed_name = ""

unknown_face_image = None
unknown_face_encoding = None
unknown_face_box = None

# =========================================
# FUNZIONE REGISTRAZIONE
# =========================================

def register_new_face(name, frame, encoding, face_box):

    global known_face_encodings
    global known_face_names

    t, r, b, l = face_box

    t = max(0, t)
    r = max(0, r)
    b = max(0, b)
    l = max(0, l)

    face_crop = frame[t:b, l:r]

    if face_crop.size == 0:
        print("Errore crop volto")
        return False

    filename = os.path.join(
        KNOWN_FACES_DIR,
        f"{name}.jpg"
    )

    cv2.imwrite(filename, face_crop)

    known_face_encodings.append(encoding)
    known_face_names.append(name)

    print(f"Registrato nuovo utente: {name}")

    return True

# =========================================
# CARICAMENTO VOLTI
# =========================================

print("Caricamento volti...")

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

for file_name in os.listdir(KNOWN_FACES_DIR):

    if file_name.startswith("."):
        continue

    path = os.path.join(
        KNOWN_FACES_DIR,
        file_name
    )

    person_name = os.path.splitext(file_name)[0]

    try:

        image = face_recognition.load_image_file(path)

        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:

            known_face_encodings.append(encodings[0])
            known_face_names.append(person_name)

            print(f"Caricato: {person_name}")

        else:
            print(f"Nessun volto trovato in {file_name}")

    except Exception as e:
        print(f"Errore caricamento {file_name}: {e}")

print("Volti caricati.")

# =========================================
# WEBCAM
# =========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Errore webcam")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nPremi:")
print("S -> avvia scansione")
print("Q -> esci")

# =========================================
# LOOP PRINCIPALE
# =========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Errore lettura webcam")
        break

    # effetto specchio
    frame = cv2.flip(frame, 1)

    # =====================================
    # SCANSIONE ATTIVA
    # =====================================

    if recognition_active and not register_mode:

        processing_frame_count += 1

        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=SCALE_FACTOR,
            fy=SCALE_FACTOR
        )

        rgb_small = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        face_locations = face_recognition.face_locations(
            rgb_small,
            model=MODEL
        )

        face_encodings = face_recognition.face_encodings(
            rgb_small,
            face_locations
        )

        last_face_locations = []
        last_face_names = []

        for (top, right, bottom, left), encoding in zip(
            face_locations,
            face_encodings
        ):

            name = "Sconosciuto"

            # =============================
            # CONFRONTO VOLTI
            # =============================

            if len(known_face_encodings) > 0:

                matches = face_recognition.compare_faces(
                    known_face_encodings,
                    encoding,
                    TOLERANCE
                )

                distances = face_recognition.face_distance(
                    known_face_encodings,
                    encoding
                )

                best_match_index = np.argmin(distances)

                if matches[best_match_index]:

                    name = known_face_names[best_match_index]

                    if name != last_recognized:

                        print(f"Riconosciuto: {name}")

                        last_recognized = name

                        status_message = f"Riconosciuto: {name}"
                        status_color = (0, 255, 0)

                        recognition_active = False
                        processing_frame_count = 0

                        unknown_face_buffer = []

                        last_face_locations = []
                        last_face_names = []

                        break

            # =============================
            # RIPORTA COORDINATE
            # =============================

            scale = int(1 / SCALE_FACTOR)

            top *= scale
            right *= scale
            bottom *= scale
            left *= scale

            last_face_locations.append(
                (top, right, bottom, left)
            )

            last_face_names.append(name)

            # =============================
            # BUFFER SCONOSCIUTI
            # =============================

            if name == "Sconosciuto":

                unknown_face_buffer.append(
                    (
                        encoding,
                        (top, right, bottom, left)
                    )
                )

        # =====================================
        # REGISTRAZIONE AUTOMATICA
        # =====================================

        if recognition_active and AUTO_SAVE_UNKNOWN:

            if len(unknown_face_buffer) >= UNKNOWN_BUFFER_SIZE:

                print("Volto sconosciuto stabile rilevato")

                encs = [
                    e for e, _ in unknown_face_buffer
                ]

                avg_encoding = np.mean(
                    encs,
                    axis=0
                )

                t, r, b, l = unknown_face_buffer[-1][1]

                unknown_face_image = frame.copy()
                unknown_face_encoding = avg_encoding
                unknown_face_box = (t, r, b, l)

                register_mode = True

                recognition_active = False

                processing_frame_count = 0

                unknown_face_buffer = []

                status_message = "Inserisci nome e premi ENTER"
                status_color = (0, 255, 255)

                # reset box
                last_face_locations = []
                last_face_names = []

        # =====================================
        # TIMEOUT
        # =====================================

        if recognition_active and processing_frame_count >= MAX_SCAN_FRAMES:

            print("Nessun volto noto rilevato")

            recognition_active = False

            processing_frame_count = 0

            unknown_face_buffer = []

            last_face_locations = []
            last_face_names = []

            status_message = "Nessun volto rilevato"
            status_color = (0, 0, 255)

    # =====================================
    # UI
    # =====================================

    cv2.putText(
        frame,
        status_message,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )

    cv2.putText(
        frame,
        "S = scan | Q = quit",
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    # =====================================
    # BOX VOLTI
    # =====================================

    for (top, right, bottom, left), name in zip(
        last_face_locations,
        last_face_names
    ):

        color = (0, 255, 0)

        if name == "Sconosciuto":
            color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            color,
            2
        )

        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    # =====================================
    # INPUT NOME DA INTERFACCIA
    # =====================================

    if register_mode:

        cv2.rectangle(
            frame,
            (20, 60),
            (600, 120),
            (40, 40, 40),
            -1
        )

        cv2.putText(
            frame,
            "Inserisci nome:",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            typed_name,
            (30, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    # =====================================
    # MOSTRA FRAME
    # =====================================

    cv2.imshow(
        "Face Recognition System",
        frame
    )

    # =====================================
    # TASTIERA
    # =====================================

    key = cv2.waitKey(1) & 0xFF

    # =====================================
    # ESCI
    # =====================================

    if key == ord('q'):
        break

    # =====================================
    # MODALITA REGISTRAZIONE
    # =====================================

    if register_mode:

        # ENTER
        if key == 13:

            if typed_name.strip() != "":

                success = register_new_face(
                    typed_name.strip(),
                    unknown_face_image,
                    unknown_face_encoding,
                    unknown_face_box
                )

                if success:

                    status_message = f"Registrato: {typed_name}"
                    status_color = (0, 255, 0)

            else:

                status_message = "Nome non valido"
                status_color = (0, 0, 255)

            # RESET COMPLETO

            register_mode = False

            typed_name = ""

            unknown_face_image = None
            unknown_face_encoding = None
            unknown_face_box = None

            recognition_active = False

            processing_frame_count = 0

            last_face_locations = []
            last_face_names = []

            unknown_face_buffer = []

        # BACKSPACE
        elif key == 8:

            typed_name = typed_name[:-1]

        # TESTO
        elif 32 <= key <= 126:

            typed_name += chr(key)

    # =====================================
    # AVVIA SCANSIONE
    # =====================================

    else:

        if key == ord('s'):

            if not recognition_active:

                recognition_active = True

                processing_frame_count = 0

                unknown_face_buffer = []

                last_recognized = None

                last_face_locations = []
                last_face_names = []

                status_message = "Scanning..."
                status_color = (255, 255, 0)

                print("\nScansione avviata")

# =========================================
# CLEANUP
# =========================================

cap.release()

cv2.destroyAllWindows()

sys.exit()