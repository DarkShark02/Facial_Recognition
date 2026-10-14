# Facial Recognition

Real-time face recognition from your webcam, written in Python with [OpenCV](https://opencv.org/) and [face_recognition](https://github.com/ageitgey/face_recognition) (built on dlib).

The program recognizes faces you have already registered. When it sees the same unknown face for a few frames, it asks you for a name and saves the face right from the interface.

## Features

- Real-time face recognition from the webcam
- Register new faces from the interface, without touching files or code
- Stable unknown-face detection: the face encoding is averaged over several frames of the same person, so it is more reliable
- Multiple photos per person for better accuracy
- Colored boxes: green for recognized faces, red for unknown ones
- Command-line options for camera, tolerance and detection model
- Fully offline: no data leaves your computer

## Requirements

- Python 3.9 or newer (tested with Python 3.10 on Windows)
- A webcam
- **CMake** and a C++ compiler, needed to build `dlib`:
  - **Windows**: [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the "Desktop development with C++" workload, plus `pip install cmake`
  - **Linux**: `sudo apt install build-essential cmake`
  - **macOS**: `xcode-select --install` and `brew install cmake`

## Installation

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

Installing `dlib` can take a few minutes because it is compiled from source.

## Usage

```bash
python main.py
```

### Keys

| Key         | Action          |
|-------------|-----------------|
| `S`         | Start a scan    |
| `Q` / `ESC` | Quit            |

While registering a new face:

| Key         | Action                  |
|-------------|-------------------------|
| `ENTER`     | Confirm the name        |
| `BACKSPACE` | Delete a character      |
| `ESC`       | Cancel registration     |

### Options

```bash
python main.py --camera 1 --tolerance 0.5 --model cnn --faces-dir faces
```

| Option        | Default | Description |
|---------------|---------|-------------|
| `--camera`    | `0`     | Webcam index (try `1` if you have more than one camera). |
| `--tolerance` | `0.6`   | Similarity threshold. Lower = stricter (fewer false matches). |
| `--model`     | `hog`   | `hog` is fast on CPU; `cnn` is more accurate but much slower without a GPU. |
| `--faces-dir` | `faces` | Folder containing the known faces. |

### How a scan works

1. Press `S`: the program analyzes the next frames.
2. If it finds a known face, it shows `Recognized: <name>` and stops.
3. If it sees the same unknown person for several frames, it enters **registration mode**: type a name and press `ENTER`. The photo is saved in `faces/<name>/`.
4. If nobody is recognized within a set number of frames, the scan stops.

Registering a person again with the same name adds a new photo to the existing ones, which improves recognition.

### Adding faces manually

You can also put photos directly in the `faces/` folder. Two layouts are supported:

```
faces/
├── mario.jpg          # one photo: the file name is the person's name
└── giulia/            # several photos: the folder name is the person's name
    ├── 1.jpg
    └── 2.png
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`. Use well-lit, front-facing photos with only one person in each.

## Advanced configuration

More settings are at the top of [`main.py`](main.py):

| Setting               | Default | Description |
|-----------------------|---------|-------------|
| `SCALE_FACTOR`        | `0.25`  | Frames are resized by this factor before analysis (lower = faster, but distant faces are missed). |
| `MAX_SCAN_FRAMES`     | `30`    | Frames analyzed before a scan times out. |
| `AUTO_SAVE_UNKNOWN`   | `True`  | Offer to register unknown faces. |
| `UNKNOWN_BUFFER_SIZE` | `10`    | Frames with the same unknown face needed before registration starts. |
| `CROP_MARGIN`         | `0.6`   | Margin added around the face when the photo is saved. |

## Troubleshooting

- **`pip install` fails on `dlib`**: CMake or the C++ compiler is missing. See [Requirements](#requirements).
- **"Cannot open webcam"**: another program is using the camera, or it has a different index (`--camera 1`).
- **Wrong people are recognized**: lower the tolerance (`--tolerance 0.5`) and add more photos per person.
- **Registered people are not recognized**: raise the tolerance (`--tolerance 0.65`) or add photos taken in lighting similar to your webcam's.

## Privacy

Face images are biometric data. The `faces/` folder is excluded from git (see `.gitignore`), so your photos stay on your computer. Don't register people without their consent.

## License

Released under the [MIT License](LICENSE).

## Credits

- [face_recognition](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- [dlib](http://dlib.net/) by Davis King
- [OpenCV](https://opencv.org/)
