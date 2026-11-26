# Animal Sound Detection

An audio classification project that predicts **dog**, **cat**, or **sheep** from an uploaded recording. It converts audio into log-Mel spectrogram features, classifies the features with a 1D convolutional neural network (CNN), and provides a Streamlit interface with confidence scores, approximate detection intervals, and prediction history.

## Project structure

```text
ANIMAL_SOUND_DETECTION/
├── dataset/
│   ├── raw/                 # Add your source audio here (not included in Git)
│   └── processed/           # Generated 2-second WAV clips (not included in Git)
├── models/
│   └── animal_sound_model.keras
├── results/                 # Evaluation and training plots
├── src/
│   ├── app.py               # Streamlit application
│   ├── preprocessing.py     # Prepare source audio
│   ├── feature_extraction.py
│   ├── model_training.py    # Encode labels and split data
│   ├── cnn_model.py         # Train the 1D CNN
│   └── model_evaluation.py
└── README.md
```

The training scripts read and write data files in the project root, so run their commands from the repository root. Generated `.npy` data and raw/processed audio are excluded by `.gitignore` and need to be recreated for a fresh training run.

## 1. Requirements

- Python 3.10 or 3.11 recommended
- pip
- Source audio recordings for the three supported classes

The app can use the model and class file already present in the project. To train from scratch, you will also need audio recordings, as the raw dataset is not included in the repository.

## 2. Set up the environment

Open a terminal in the project directory, then create and activate a virtual environment.

**Windows PowerShell:**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell prevents activation, use Command Prompt and run `.venv\Scripts\activate.bat`, or invoke `.venv\Scripts\python.exe` directly in the commands below.

Install the packages used by the application and training scripts:

```powershell
python -m pip install streamlit numpy librosa tensorflow pandas altair scikit-learn matplotlib seaborn soundfile
```

## 3. Prepare training audio

Create these folders and place audio recordings in the matching class folder. WAV, MP3, FLAC, and OGG files are accepted by the preprocessing script.

```text
dataset/raw/dog/
dataset/raw/cat/
dataset/raw/sheep/
```

Use clearly labeled recordings and include examples for all three classes. The folder name supplies the training label. Keep the audio you have rights to use; this repository does not download or bundle a dataset.

## 4. Preprocess the audio

From the repository root, run:

```powershell
python src/preprocessing.py
```

For each source recording, the script converts audio to mono at 16 kHz, trims quiet portions, normalizes its level, and saves a two-second WAV file under `dataset/processed/<class>/`.

## 5. Extract log-Mel features

```powershell
python src/feature_extraction.py
```

This reads the processed class folders and creates `X_features.npy`, `y_labels.npy`, and `feature_file_names.npy` in the project root. Each recording becomes a 64-band log-Mel spectrogram; audio is represented as a fixed two-second sample.

## 6. Encode labels and split the data

```powershell
python src/model_training.py
```

This script encodes the class names, makes a stratified 80/20 train/test split, and writes `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, and `classes.npy` in the project root.

## 7. Train the CNN

```powershell
python src/cnn_model.py
```

Training uses a 1D CNN for up to 30 epochs. The script evaluates on the held-out test split and saves the model to `models/animal_sound_model.keras`, plus accuracy and loss plots under `results/`. Training time depends on the size of your dataset and computer.

## 8. Evaluate the model

```powershell
python src/model_evaluation.py
```

This prints per-class precision, recall, and F1 scores for the test split. Review these scores along with the saved plots when assessing model quality.

## 9. Run the application

```powershell
streamlit run src/app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in your browser. Upload a WAV, MP3, OGG, or FLAC recording and choose **Predict Animal**. The app resamples it to 16 kHz, displays the predicted class and confidence, and estimates intervals using overlapping two-second windows. Detection intervals are shown only for windows with at least 60% confidence. Uploaded-file and interval history is stored in `detection_history.db`.

## How prediction works

1. Audio is loaded as mono 16 kHz samples.
2. A two-second window is trimmed or padded, then converted to a 64-band log-Mel spectrogram.
3. The spectrogram is transposed to the 1D CNN input shape `(126, 64)`.
4. The CNN returns probabilities for the saved animal classes.
5. For interval estimates, the app scans overlapping two-second windows every half-second and merges adjacent same-class detections above the confidence threshold.

## Notes

- The class mapping in `classes.npy` must correspond to the output order of the saved model. The training pipeline generates both together.
- The app expects the model at `models/animal_sound_model.keras` and the class mapping at `classes.npy` (it can fall back to `animal_classes.npy`).
- The repository also contains `src/train_model.py`, an alternate 2D CNN training script. The Streamlit app uses the 1D CNN architecture trained by `src/cnn_model.py`; use the steps above for a compatible model.
- This is a three-class classifier. Its confidence is the model's predicted probability, not a calibrated guarantee that an animal is present.
