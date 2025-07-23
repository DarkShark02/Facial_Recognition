import cv2
import face_recognition
import os
import sys
import numpy as np
import time

# --- Configurazione ---
KNOWN_FACES_DIR = "faces" # Cartella dove salverai le foto delle persone da riconoscere
TOLERANCE = 0.6  # Più basso è, più stringente è il riconoscimento (0.6 è un buon valore di partenza)
FRAME_THICKNESS = 2
FONT_THICKNESS = 2
MODEL = "hog" # "hog" per CPU (più veloce per CPU), "cnn" per GPU (più lento su CPU, più accurato)

# Ottimizzazioni per il lag (lasciate attive ma potremmo rimuovere se il "scan-on-demand" è abbastanza veloce)
SCALE_FACTOR = 0.25 # Riduce la dimensione del frame a 1/4 (0.25) per elaborazione
PROCESS_EVERY_N_FRAMES = 1 # Processa ogni frame quando il riconoscimento è attivo (o aumenta se serve più velocità)

# --- Parametri per la nuova logica "scan-on-demand" ---
MAX_SCAN_FRAMES = 30 # Numero massimo di frame da elaborare dopo aver premuto 'S'
                     # 30 frames a 30fps sono 1 secondo di scansione
                     # Regola questo valore per la durata desiderata della scansione attiva
                     
# --- Carica i Volti Noti ---
print("Caricamento volti noti...")

known_face_encodings = []
known_face_names = []

# Assicurati che la cartella esista
if not os.path.exists(KNOWN_FACES_DIR):
    print(f"Errore: La cartella '{KNOWN_FACES_DIR}' non esiste. Creala e inserisci le immagini dei volti noti.")
    sys.exit()

for name in os.listdir(KNOWN_FACES_DIR):
    # Ignora file di sistema o non immagine (es. .DS_Store su macOS)
    if name.startswith('.'):
        continue

    person_name = os.path.splitext(name)[0]
    image_path = os.path.join(KNOWN_FACES_DIR, name)
    
    try:
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)
        
        if face_encoding:
            known_face_encodings.append(face_encoding[0])
            known_face_names.append(person_name)
            print(f"Caricato volto di: {person_name}")
        else:
            print(f"Attenzione: Nessun volto trovato in {image_path}. Saltato.")
    except Exception as e:
        print(f"Errore nel caricamento o elaborazione di {image_path}: {e}")

if not known_face_encodings:
    print("Nessun volto valido caricato. Assicurati che le immagini nella cartella 'faces' contengano volti chiari.")
    sys.exit()

print("Caricamento volti completato.")

# --- Setup Webcam ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Imposta larghezza
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Imposta altezza

if not cap.isOpened():
    print("Errore: Impossibile accedere alla webcam. Assicurati che non sia già in uso e che i driver siano installati.")
    sys.exit()

print("Avvia il riconoscimento facciale. Premi 'q' per uscire.")

# --- Variabili di stato per la nuova logica ---
recognition_active = False          # Flag per controllare se la scansione è in corso
processing_frame_count = 0          # Contatore dei frame elaborati durante una scansione
last_face_locations = []            # Ultima posizione dei volti rilevati
last_face_names = []                # Ultimi nomi dei volti rilevati
current_display_message = "Premi 'S' per avviare il riconoscimento" # Messaggio a schermo (non usato più qui ma per chiarezza)
status_message = "Premi 'S' per avviare il riconoscimento" # Messaggio di stato principale
status_color = (255, 255, 0) # Giallo di default (BGR)
last_recognized_person_printed = None # Per stampare in console il nome solo una volta

while True:
    ret, frame = cap.read()
    if not ret:
        print("Errore nella lettura del frame.")
        break

    # Capovolgi il frame orizzontalmente per un effetto specchio
    frame = cv2.flip(frame, 1)

    # --- Logica di Scansione ---
    if recognition_active:
        processing_frame_count += 1

        small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Resetta le liste per i risultati di questo frame PRIMA di popolarle
        last_face_locations = [] 
        last_face_names = []     
        found_known_person_in_frame = False 

        for face_location, face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, TOLERANCE)
            name = "Sconosciuto"

            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    found_known_person_in_frame = True
            
            # Scala le posizioni del volto per il frame originale
            scale_factor_inv = int(1/SCALE_FACTOR)
            top = int(face_location[0] * scale_factor_inv)
            right = int(face_location[1] * scale_factor_inv)
            bottom = int(face_location[2] * scale_factor_inv)
            left = int(face_location[3] * scale_factor_inv)

            last_face_locations.append((top, right, bottom, left))
            last_face_names.append(name)

            # Stampa in console solo se è un nuovo riconoscimento e non è "Sconosciuto"
            if name != "Sconosciuto" and name != last_recognized_person_printed:
                print(f"Riconosciuto: {name}")
                last_recognized_person_printed = name
                # Una volta riconosciuto, interrompi la scansione e torna in attesa
                recognition_active = False
                status_message = f"Riconosciuto: {name}."
                status_color = (0, 255, 0) # Verde
                processing_frame_count = 0 
                break # Esci dal ciclo for, abbiamo trovato quello che cercavamo
        
        # Se la scansione è ancora attiva (non interrotta da un riconoscimento)
        if recognition_active:
            if not last_face_locations and processing_frame_count > 0: # Nessun volto trovato nel frame
                status_message = "Cercando volti..."
                status_color = (255, 255, 0) # Giallo
            elif not found_known_person_in_frame and last_face_locations: # Volti trovati ma tutti sconosciuti
                status_message = "Volto sconosciuto."
                status_color = (0, 0, 255) # Rosso
                print("Volto sconosciuto.")
                recognition_active = False
                processing_frame_count = 0
                last_recognized_person_printed = None
            
            # Se la scansione ha raggiunto il limite di frame e non ha trovato nessuno
            if recognition_active and processing_frame_count >= MAX_SCAN_FRAMES:
                print("Scansione completata. Nessun volto noto riconosciuto nel periodo.")
                status_message = "Nessun volto noto."
                status_color = (0, 0, 255) # Rosso
                recognition_active = False
                processing_frame_count = 0
                last_face_locations = [] # Pulisci i box del frame precedente se non riconosciuto
                last_face_names = []
                last_recognized_person_printed = None
                
    else: # recognition_active è False (modalità attesa)
        # Se siamo in modalità attesa e c'è stato un riconoscimento precedente, i box restano
        # Se invece era "Nessun volto noto" o "Sconosciuto" i box sono già stati puliti o non c'erano
        
        # Mantieni il messaggio di stato corretto se è un risultato, altrimenti metti quello di default
        if not status_message.startswith("Riconosciuto:") and \
           not status_message.startswith("Nessun volto noto.") and \
           not status_message.startswith("Volto sconosciuto."):
            status_message = "Premi 'S' per avviare il riconoscimento"
            status_color = (255, 255, 0) # Giallo
        
        last_recognized_person_printed = None # Resetta il flag per la stampa in console

    # Mostra il messaggio di stato in alto a sinistra
    cv2.putText(frame, status_message, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA) 
    
    # Messaggi in basso a sinistra
    bottom_text_y_start = frame.shape[0] - 50 # Un po' più in alto del bordo inferiore
    cv2.putText(frame, "Premi 'S' per avviare riconoscimento", (10, bottom_text_y_start),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA) # Bianco
    cv2.putText(frame, "Premi 'Q' per uscire", (10, bottom_text_y_start + 25), # 25 pixel sotto
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA) # Bianco

    # Mostra il frame risultante
    cv2.imshow('Identificazione Facciale', frame)

    # Gestione input da tastiera
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if not recognition_active: # Avvia la scansione solo se non è già attiva
            recognition_active = True
            processing_frame_count = 0 # Resetta il contatore per una nuova scansione
            print("Avviato il riconoscimento facciale.")
            last_recognized_person_printed = None # Resetta il flag per la stampa in console
            status_message = "Cercando volti..." # Reset del messaggio di stato all'avvio
            status_color = (255, 255, 0) # Giallo
            last_face_locations = [] # Pulisci immediatamente i box precedenti all'avvio di una nuova scansione
            last_face_names = []
        # Se premi S mentre è già attivo, la logica attuale ignora il secondo 'S' per non interrompere una scansione attiva.

# --- Pulizia ---
cap.release()
cv2.destroyAllWindows()
sys.exit()