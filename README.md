# Facial Recognition

Riconoscimento facciale in tempo reale da webcam, scritto in Python con [OpenCV](https://opencv.org/) e [face_recognition](https://github.com/ageitgey/face_recognition) (basata su dlib).

Il programma riconosce i volti già registrati e, quando trova un volto sconosciuto stabile davanti alla camera, ti chiede un nome per salvarlo direttamente dall'interfaccia.

## Funzionalità

- Riconoscimento dei volti in tempo reale dalla webcam
- Registrazione di nuovi volti dall'interfaccia, senza toccare file o codice
- Rilevamento di un volto sconosciuto "stabile": l'encoding è la media di più frame della stessa persona, quindi è più affidabile
- Più foto per la stessa persona, per migliorare il riconoscimento
- Riquadri colorati: verde per i volti riconosciuti, rosso per gli sconosciuti
- Opzioni da riga di comando per webcam, tolleranza e modello
- Tutto in locale: nessun dato viene inviato in rete

## Requisiti

- Python 3.9 o superiore (testato con Python 3.10 su Windows)
- Una webcam
- Per compilare `dlib` servono **CMake** e un compilatore C++:
  - **Windows**: [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) con il workload "Sviluppo di applicazioni desktop con C++", più `pip install cmake`
  - **Linux**: `sudo apt install build-essential cmake`
  - **macOS**: `xcode-select --install` e `brew install cmake`

## Installazione

```bash
git clone https://github.com/DarkShark02/Facial_Recognition.git
cd Facial_Recognition

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

L'installazione di `dlib` può richiedere qualche minuto, perché viene compilato.

## Utilizzo

```bash
python main.py
```

### Tasti

| Tasto       | Azione                                   |
|-------------|------------------------------------------|
| `S`         | Avvia una scansione                      |
| `Q` / `ESC` | Esci                                     |

In modalità registrazione:

| Tasto       | Azione                    |
|-------------|---------------------------|
| `ENTER`     | Conferma il nome          |
| `BACKSPACE` | Cancella un carattere     |
| `ESC`       | Annulla la registrazione  |

### Opzioni

```bash
python main.py --camera 1 --tolerance 0.5 --model cnn --faces-dir faces
```

| Opzione       | Default | Descrizione |
|---------------|---------|-------------|
| `--camera`    | `0`     | Indice della webcam (prova `1` se hai più camere). |
| `--tolerance` | `0.6`   | Soglia di somiglianza. Più bassa = più severa (meno falsi positivi). |
| `--model`     | `hog`   | `hog` è veloce su CPU; `cnn` è più preciso ma molto più lento senza GPU. |
| `--faces-dir` | `faces` | Cartella dei volti noti. |

### Come funziona una scansione

1. Premi `S`: il programma analizza i frame successivi.
2. Se trova un volto noto, mostra `Riconosciuto: <nome>` e si ferma.
3. Se vede la stessa persona sconosciuta per più frame, entra in **modalità registrazione**: scrivi il nome e premi `ENTER`. La foto viene salvata in `faces/<nome>/`.
4. Se entro un certo numero di frame non riconosce nessuno, la scansione si ferma.

Se registri di nuovo una persona già presente con lo stesso nome, la nuova foto si aggiunge alle sue, e il riconoscimento migliora.

### Aggiungere volti a mano

Puoi anche mettere le foto direttamente nella cartella `faces/`. Sono supportati due formati:

```
faces/
├── mario.jpg          # una foto: il nome del file è il nome della persona
└── giulia/            # più foto: il nome della cartella è il nome della persona
    ├── 1.jpg
    └── 2.png
```

Formati supportati: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`. Usa foto ben illuminate, con il volto frontale e una sola persona per foto.

## Configurazione avanzata

Altri parametri si trovano all'inizio di [`main.py`](main.py):

| Parametro             | Default | Descrizione |
|-----------------------|---------|-------------|
| `SCALE_FACTOR`        | `0.25`  | Ridimensionamento dei frame prima dell'analisi (più basso = più veloce, ma i volti lontani vengono persi). |
| `MAX_SCAN_FRAMES`     | `30`    | Frame analizzati prima che la scansione vada in timeout. |
| `AUTO_SAVE_UNKNOWN`   | `True`  | Propone di registrare i volti sconosciuti. |
| `UNKNOWN_BUFFER_SIZE` | `10`    | Frame con lo stesso volto sconosciuto necessari per avviare la registrazione. |
| `CROP_MARGIN`         | `0.6`   | Margine attorno al volto quando la foto viene salvata. |

## Problemi comuni

- **`pip install` fallisce su `dlib`**: mancano CMake o il compilatore C++, vedi [Requisiti](#requisiti).
- **"Impossibile aprire la webcam"**: la camera è usata da un altro programma, oppure ha un indice diverso (`--camera 1`).
- **Riconosce le persone sbagliate**: abbassa la tolleranza (`--tolerance 0.5`) e aggiungi più foto per persona.
- **Non riconosce chi è registrato**: alza la tolleranza (`--tolerance 0.65`) o aggiungi foto con un'illuminazione simile a quella della webcam.

## Privacy

Le immagini dei volti sono dati biometrici. La cartella `faces/` è esclusa da git (`.gitignore`), così le foto restano sul tuo computer. Non registrare persone senza il loro consenso.

## Licenza

Distribuito con licenza [MIT](LICENSE).

## Crediti

- [face_recognition](https://github.com/ageitgey/face_recognition) di Adam Geitgey
- [dlib](http://dlib.net/) di Davis King
- [OpenCV](https://opencv.org/)
