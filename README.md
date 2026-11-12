# Emotion Classifier

A **multi-label emotion classification** baseline built in Python with scikit-learn. Given short text, the model can predict more than one emotion using TF-IDF text features and one-vs-rest logistic regression.

**[Open the training and evaluation notebook](notebooks/01_baseline_goemotions_multilabel.ipynb)**

## How it works

1. Load the simplified **GoEmotions** dataset with Hugging Face `datasets`.
2. Convert emotion-label lists into multi-label targets.
3. Transform text into TF-IDF features.
4. Fit a one-vs-rest logistic regression classifier.
5. Evaluate precision, recall, and F1, and compare thresholding with top-k predictions.

The notebook contains the data preparation, training, evaluation, and visualization steps. This repository is a notebook-based ML experiment, not a deployed prediction service.

## Reported baseline results

The notebook's reported evaluation uses **samples-averaged** multi-label metrics:

| Prediction approach | Samples F1 | Samples precision | Samples recall |
| --- | ---: | ---: | ---: |
| Probability threshold = 0.20 | **0.54** | 0.54 | 0.58 |
| Top 2 predicted labels | 0.48 | 0.38 | 0.68 |

Thresholding can return a variable number of labels for each text, while top-k returns a fixed number. These are results from this baseline and dataset; they are not a claim of production performance on arbitrary text.

## Tools

Python 3.11 · scikit-learn · pandas · NumPy · Hugging Face datasets · Jupyter

## Run the notebook

From a local clone of this repository:

```bash
conda create -n emotion python=3.11 -y
conda activate emotion
conda install -y numpy pandas scikit-learn matplotlib jupyter ipykernel
python -m pip install datasets
jupyter notebook notebooks/01_baseline_goemotions_multilabel.ipynb
```

The notebook downloads GoEmotions when run, so an internet connection is needed for the initial data load. You can also open it in VS Code and choose the `emotion` Python environment as the notebook kernel.

## What I learned

This project explores the difference between single-label and multi-label classification, how decision thresholds change precision and recall, and how to interpret a baseline before considering more complex models.

**Starting point:** [`notebooks/01_baseline_goemotions_multilabel.ipynb`](notebooks/01_baseline_goemotions_multilabel.ipynb).
