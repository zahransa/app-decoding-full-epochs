
import mne
import os
import numpy as np
from sklearn.pipeline import make_pipeline
from mne.decoding import Scaler, Vectorizer, cross_val_multiscore
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import json
#to fix
#qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load inputs from config.json
with open('config.json') as config_json:
    config = json.load(config_json)

# == LOAD DATA ==
fname = config['epo']
epochs = mne.read_epochs(fname)

epochs_auditory = epochs['auditory']




# First, create X and y.
epochs_auditory_grad = epochs_auditory.copy().pick_types(meg='grad')
X = epochs_auditory_grad.get_data()
y = epochs_auditory_grad.events[:, 2]

# Classifier pipeline.
clf = make_pipeline(
    # An MNE scaler that correctly handles different channel types –
    # isn't that great?!
    Scaler(epochs_auditory_grad.info),
    # Remember this annoying and error-prone NumPy array reshaping we had to do
    # earlier? Not anymore, thanks to the MNE vectorizer!
    Vectorizer(),
    # And, finally, the actual classifier.
    LogisticRegression())

# Run cross-validation.
# Note that we're using MNE's cross_val_multiscore() here, not scikit-learn's
# cross_val_score() as above. We simply pass the number of desired CV splits,
# and MNE will automatically do the rest for us.
n_splits = 5
scoring = 'roc_auc'
scores = cross_val_multiscore(clf, X, y, cv=5, scoring='roc_auc')

# Mean and standard deviation of ROC AUC across cross-validation runs.
roc_auc_mean = round(np.mean(scores), 3)
roc_auc_std = round(np.std(scores), 3)


print(f'CV scores: {scores}')
print(f'Mean ROC AUC = {roc_auc_mean:.3f} (SD = {roc_auc_std:.3f})')


fig, ax = plt.subplots()
ax.boxplot(scores,
           showmeans=True, # Green triangle marks the mean.
           whis=(0, 100),  # Whiskers span the entire range of the data.
           labels=['Left vs Right'])
ax.set_ylabel('Score')
ax.set_title('Cross-Validation Scores')
fig.savefig(os.path.join('out_figs', 'Cross-Validation-Scores.png'))



