# Predicting Rural Water Point Functionality in East Africa

Machine learning summative project: a binary classifier that predicts whether a rural water
point (hand pump, protected spring, tapstand) has water, trained on the
**Water Point Data Exchange (WPdx+)** for Uganda, Ethiopia, Kenya and Tanzania and tested for
transfer on **Rwanda**. Traditional machine learning (scikit-learn) and deep learning
(TensorFlow/Keras with `tf.data`) are compared across 20 logged experiments.

| Resource | Link |
|---|---|
| Report: https://docs.google.com/document/d/1dpqUZBlZu7pNKXzpeVLMPzX1SvohW3l5bb2XyopgHJc/edit?tab=t.0
| Google Drive folder: https://drive.google.com/drive/u/3/folders/14FgFhuDFQENEMx3xEksdaqpxxRA4xvU9
| Video presentation | _link to be added_ |
| Open the notebook in Colab: https://colab.research.google.com/drive/1IHv_NzVnCxRwanLZ1Hmr_Hzb1lS0fQGL

## Problem

About a quarter of hand pumps in sub-Saharan Africa are not working at any given time, and
water authorities usually learn of a failure only after a community has lost its supply. If the
points most likely to be broken can be ranked from data that is already recorded, inspection and
repair visits can be sent where they are most needed.

**Task:** given a water point's recorded attributes (technology, water source, age, management,
payment, location, distances to roads and towns, local population), predict whether water was
available at its last visit (`status_id`), with *not functional* as the positive class.

## Data

* Source: WPdx+ country extracts on the Humanitarian Data Exchange (`wpdx_uga`, `wpdx_eth`,
  `wpdx_ken`, `wpdx_tza`, `wpdx_rwa`), Global Water Challenge. Licensed CC BY 4.0 by WPdx; listed as
  CC BY-SA on HDX.
* 168,840 water points; 165,130 with a known label; 17.9% not functional in the training countries.
* Snapshot downloaded 20 October 2026 and pinned with SHA-256 checksums in the notebook.
* **Raw data is not committed to this repository.** The notebook loads it, in order, from
  `data/raw/`, from `wpdx_raw_data.zip` in the Google Drive folder (mounted in Colab), or directly
  from HDX.

## Results

| Model (test set, 23,900 points) | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Random forest (ML-3, threshold 0.48) | 0.863 | 0.634 | 0.632 | 0.534 | 0.579 |
| Embedding network with districts (DL-7, threshold 0.25) | 0.850 | 0.611 | 0.690 | 0.454 | 0.548 |
| Logistic regression baseline (ML-2) | 0.779 | 0.478 | 0.701 | 0.333 | 0.452 |

* Inspecting the riskiest 10% of points (random forest) finds **39% of all broken points**,
  with 7 in 10 visits finding a broken point.
* Tree ensembles beat neural networks on this tabular data; class imbalance is best handled by
  tuning the decision threshold.
* **Transfer to Rwanda fails** (random forest ROC-AUC 0.60, PR-AUC 0.21): Rwanda's points are mostly
  piped-scheme tapstands, its management and payment fields are almost empty, and failure relates
  to the features differently.

The full experiment table is in `results/experiment_table.csv` and in Section 7 of the notebook.

## Repository structure

```
water_point_functionality.ipynb   Main notebook (runs top to bottom in Colab; outputs included)
requirements.txt                  Dependencies
figures/                          All figures saved by the notebook (figures/report: unnumbered copies)
results/                          Experiment table, test and transfer metrics, importance, shift tables
logs/                             TensorBoard logs (HParams for every run; per-epoch curves for networks)
```

Not committed (shared through Google Drive): `data/raw/` and `models/`
(`best_traditional_model.joblib`, `best_deep_learning_model.keras`,
`best_deep_learning_preprocessing.joblib`, `model_card.json`).

## How to run

**Google Colab (recommended).**

1. Copy `wpdx_raw_data.zip` from the shared Drive folder into `My Drive/water-point-functionality/`
   (or skip this: the notebook then downloads the data from HDX).
2. Open the notebook in Colab and choose *Runtime → Run all*. Allow Drive access when asked.
   A CPU runtime is enough; the full run takes about 25–35 minutes.

**Locally.**

```bash
pip install -r requirements.txt
mkdir -p data/raw && unzip wpdx_raw_data.zip -d data/raw
jupyter notebook water_point_functionality.ipynb
```

## Experiment tracking

Every experiment logs its hyperparameters and metrics to TensorBoard's **HParams** dashboard,
and the deep learning runs also log per-epoch training curves:

```bash
tensorboard --logdir logs
```

In Colab, the last cell of the notebook opens TensorBoard inline.

## Reproducibility

* All random seeds (`PYTHONHASHSEED`, Python, NumPy, TensorFlow) are fixed to 42.
* All learned preprocessing is fitted on the training split only, inside scikit-learn pipelines.
* Results in this README and the report come from a clean top-to-bottom run with Python 3.13,
  pandas 2.3.3, NumPy 2.5.3, scikit-learn 1.9.1, imbalanced-learn 0.14.2, TensorFlow 2.21.0 and
  Matplotlib 3.11.2. Other versions (for example Colab's) may change the last decimal places.

## Data attribution

Water point data © the contributing organisations, compiled by the Water Point Data Exchange
(Global Water Challenge), https://www.waterpointdata.org, distributed via the Humanitarian Data
Exchange, https://data.humdata.org. Used under Creative Commons Attribution terms (CC BY 4.0 per WPdx;
CC BY-SA as listed on HDX).
