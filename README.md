# Dissertation

This repo contains the implementation of the method proposed in my dissertation with scripts for the results.

The repository is structured as follows in order of importance:

- src - all of the functional code including the method implementation, models, and other utily scripts.
- requirements.txt - the requirements necessary to run the files.
- result_scripts - the scripts used to generate plots and tables found in results.
- quantiles - the quantiles experiments which is the basis of the dissertation.
- scripts - scripts to train models and evaluate them.
- images - the generated graphics used in the dissertation.
- experiments - the general experiments.
- random - other experiments I tried.
- r_scripts - R plots I've experimented with for easy plotting.
- saved_results - .csv files used for r_scripts etc.
- old_experiments - old experiments that are not used in the current version of the dissertation.
- experiments.typ - notes from old_experiments not used in the current version of the dissertation.
- notes.typ - notes from old_experiments not used in the current version of the dissertation.

Inside src/:

- method.py - contains the implementation of the proposed methods.
- utils.py - contains utility functions used throughout all scripts.
- datasets.py - contains the dataset loading, creation, and dataset-related utility functions.
- models.py - contains the models architectures used.
- attacks.py - functions to enable generating adversarial attacks.

Note that any files ran should be run from the top-level directory, for example by running:

`python3 result_scripts/results_01.py`
