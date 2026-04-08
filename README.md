# Prosody_Semantics_Integration_Irony

Project for fMRI encoding of prosody & semantics in irony and sarcasm.
This repository contains a minimal, documented set of scripts to reproduce the
encoding pipeline used in the project.

Files
- `analysis_helpers.py` — path helpers and dataset loader wrapper
- `dataset.py` — dataset loader that reads behavioral files, fMRI arrays and embeddings
- `audio_text_embeddings.py` — generation of context-sensitive text and openSMILE audio embeddings
- `ridge_cv.py` — ridge regression + GroupKFold CV (adapted from https://github.com/HuthLab/deep-fMRI-dataset)
- `encoding.py` — main runner script
- `results_analysis.py` — analysis of results maps and table creation
- `requirements.txt` — Python deps

Quick usage
1. Adjust paths in `analysis_helpers.get_paths()` to match your layout.
2. Run:
```bash
python encoding.py --use_text --use_audio --use_base --use_pca --pca_threshold 0.55 --n_splits 5 --optimize_alpha --include_tasks irony sarcasm
```

