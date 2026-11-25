# app-decoding-full-epochs

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Decodes conditions from MEG gradiometer epochs with a logistic-regression classifier in MNE-Python and plots the 5-fold cross-validated ROC AUC scores.**

Selects the 'auditory' epochs and gradiometer channels, then runs a Scaler-Vectorizer-LogisticRegression pipeline with mne.decoding.cross_val_multiscore (5 splits, ROC AUC) to classify the event codes, and saves a box plot of the scores. The condition name is hard-coded, so it only works for epochs with an 'auditory' condition (as in the MNE sample dataset); it looks like a demonstration App.

| | |
|---|---|
| 📥 **Inputs** | `neuro/meeg/mne/epochs` |
| 📤 **Outputs** | `generic/image/png` |
| 🐳 **Container** | `brainlife/mne:1.0.2` |
| ⚙️ **Resources** | 1 CPUs |
| 🔗 **Run it** | [doi.org/10.25663/brainlife.app.650](https://doi.org/10.25663/brainlife.app.650) |

#### 📥 Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `epo` | `neuro/meeg/mne/epochs` | yes | Epoched MEG data (MNE epochs). Must contain an 'auditory' event condition and gradiometer channels. |

#### 📤 Outputs

| Output | Datatype | Description |
|---|---|---|
| `out_dir` | `generic/image/png` | Box plot (PNG) of cross-validation ROC AUC scores. |

<!-- brainlife-app:end -->

