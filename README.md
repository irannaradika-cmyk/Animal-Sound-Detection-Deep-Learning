# Animal Sound Detection

An audio classification project that predicts **dog**, **cat**, or **sheep** from an uploaded recording. It converts audio into log-Mel spectrogram features, classifies the features with a 1D convolutional neural network (CNN), and provides a Streamlit interface with confidence scores, approximate detection intervals, and prediction history.

## Project structure

```text
ANIMAL_SOUND_DETECTION/
├── dataset/
│   ├── raw/                 # Add your source audio here (not included in Git)
│   └── processed/           # Generated 2-second WAV clips (not included in Git)
├── models/
│   ├── animal_sound_cnn.keras
│   └── animal_classes.npy
├── results/                 # Evaluation plots and predictions
├── src/
│   ├── app.py               # Streamlit application
│   ├── preprocess.py        # Prepare source audio
│   ├── feature_extraction.py
│   ├── train_model.py       # Train the 1D CNN
│   └── model_evaluation.py
└── README.md
```

The training scripts read and write feature and split data files in the project root, so run their commands from the repository root. The trained model and class mapping are saved under `models/`; evaluation plots and predictions are saved under `results/`. Generated `.npy` data and raw/processed audio are excluded by `.gitignore` and need to be recreated for a fresh training run.

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
python src/preprocess.py
```

For each source recording, the script converts audio to mono at 16 kHz, trims quiet portions, normalizes its level, and saves a two-second WAV file under `dataset/processed/<class>/`.

## 5. Extract log-Mel features

```powershell
python src/feature_extraction.py
```

This reads the processed class folders and creates `X_features.npy`, `y_labels.npy`, and `feature_file_names.npy` in the project root. Each recording becomes a 64-band log-Mel spectrogram; audio is represented as a fixed two-second sample.

## 6. Train the CNN

```powershell
python src/train_model.py
```

This script encodes the class names, creates stratified training, validation, and test splits, and trains a 1D CNN for up to 30 epochs. It saves `models/animal_sound_cnn.keras`, the matching class mapping, split indices, and accuracy and loss plots under `results/`. Training time depends on the size of your dataset and computer.

## 7. Evaluate the model

```powershell
python src/model_evaluation.py
```

This prints per-class precision, recall, and F1 scores for the test split, and saves the confusion-matrix plot and predictions under `results/`. Review these scores along with the saved plots when assessing model quality.

## 8. Run the application

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

- The class mapping in `models/animal_classes.npy` must correspond to the output order of `models/animal_sound_cnn.keras`. The training pipeline generates both together.
- The repository also contains `src/cnn_model.py`, an older alternative training script. Use `src/train_model.py` for the model used by the Streamlit app.
- This is a three-class classifier. Its confidence is the model's predicted probability, not a calibrated guarantee that an animal is present.
