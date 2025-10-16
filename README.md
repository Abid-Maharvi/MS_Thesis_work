
# EEG-Based Identification of Native and Non-Native Language Perception Using Deep Learning

**Short description**
A collection of code, notebooks, and resources for training and evaluating deep learning models (LSTM, GRU, CNN-LSTM, Bi-LSTM, DNN) to classify EEG responses to native vs. non-native song stimuli (in-ear and bone-conducting headphones). The original paper and dataset description are provided in the project PDF. 

---

## Table of contents

* [Features](#features)
* [Dataset](#dataset)
* [Repository structure](#repository-structure)
* [Requirements & Installation](#requirements--installation)
* [Quickstart (Google Colab)](#quickstart-google-colab)
* [Training](#training)
* [Evaluation & Results](#evaluation--results)
* [How to cite](#how-to-cite)
* [License](#license)
* [Contact](#contact)

---

## Features

* Preprocessing pipeline: trimming, filtering (1–40 Hz Butterworth + 50 Hz notch), normalization, sequence generation.
* Implementations of multiple deep learning models: LSTM, GRU, CNN-LSTM, Bi-LSTM, and DNN.
* Training utilities with early stopping, model checkpointing, and model saving (Colab/GDrive friendly).
* Evaluation tools: confusion matrix, ROC, classification report.
* Example notebooks formatted in Kaggle/Colab style with cell explanations.

---

## Dataset

* EEG recordings from **20 subjects** collected during music/listening experiments (native, non-native, neutral music) using two modalities: **in-ear headphones** and **bone-conducting headphones**.
* Four EEG channels: **P4, Cz, F8, T7**.
* Data splits include resting-state (eyes open/closed), and auditory experiments. Two-minute segments (filtered/segmented) are provided; `data_trim.csv`, `Subjects.csv` and `Songs.csv` are included as supporting files. Details in the project paper. 



---

## Requirements & Installation

Create a Python virtual environment (recommended) and install dependencies:

```bash
python -m venv venv
source venv/bin/activate       # or venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

Example `requirements.txt` (adjust versions as needed):

```
numpy
pandas
scikit-learn
matplotlib
seaborn
tensorflow>=2.10
keras
wfdb             # if using WFDB formatted files
jupyter
tqdm
```

> Note: For Colab, you can skip virtualenv and just run the notebook cells.

---

## Quickstart (Google Colab)

A minimal Colab-friendly flow:

1. Upload `data/` folder to your Google Drive.
2. Mount Drive in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
DATA_DIR = '/content/drive/MyDrive/EEG_project/data'
```

3. Install dependencies (if needed):

```bash
!pip install -r /content/drive/MyDrive/EEG_project/requirements.txt
```

4. Run the preprocessing notebook (`02-preprocessing.ipynb`) to generate sequences and train/val/test splits.
5. Run `04-train-lstm-gru-bilstm.ipynb` to train models and save the best checkpoints to Drive.

**Important**: Notebooks in `notebooks/` include explanatory cells and `model.save()` checkpoints. Save final models to `saved_models/` or Google Drive.

---

## Training

Example CLI to train a model (script `train.py`):

```bash
python src/train.py \
  --data-dir data/filtered \
  --model bilstm \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-4 \
  --save-dir saved_models/
```

Training details (recommended settings from experiments):

* Optimizer: Adam (lr = 1e-4)
* Loss: Binary cross-entropy
* Early stopping: patience = 5 (monitor validation loss)
* Sequence length: e.g., 36 timesteps (10 sec windows as in paper)
* Normalization: StandardScaler per channel

---

## Evaluation & Results

The project paper reports the following performance (summary):

* **Bi-LSTM**: **98% accuracy** (best model).
* **GRU**: ~96% accuracy.
* **LSTM**: ~94% accuracy.
* **CNN-LSTM**: ~83% accuracy.
* **DNN**: ~70% accuracy.

Confusion matrices, ROC curves, and classification reports are included in `notebooks/05-evaluation.ipynb`. For reproducibility, evaluate using the same data splits and preprocessing as used in training. Refer to the published paper for the full experimental details. 

---

## Reproducibility checklist

* Use the same filter (1st order 1–40 Hz Butterworth) and 50 Hz notch (Q factor 30).
* Use sequence length and sampling settings consistent with the paper (36 time steps ~ 10s windows).
* Use the provided `Subjects.csv` and `data_trim.csv` to reconstruct segmentation if needed.
* Save model weights and training logs (TensorBoard or CSV) for traceability.

---

## How to cite

If you use this repository or dataset in your research, please cite the conference paper:

> Muhammad Abid Hussain and Imran Usman, *EEG-Based Identification of Native and Non-Native Language Perception Using Deep Learning*, 2025 International Conference on Communication Technologies (ComTech). 

(You can include the full PDF from `paper/` folder.)

---

## License

Choose an appropriate license (e.g., MIT, Apache-2.0). Example: add `LICENSE` file with MIT license and include short notice here.

---

## Contact

Author: **Muhammad Abid Hussain**
Email: [abidmaharvi@hotmail.com](mailto:abidmaharvi@hotmail.com) (see paper for contact). 

---


