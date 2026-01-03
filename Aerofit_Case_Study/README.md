# Aerofit Case Study

This folder contains a cleaned and refactored analysis of the Aerofit treadmill dataset and a baseline modeling notebook to produce product recommendations.

Contents
- `refactored_modeling.ipynb` — A refactored Jupyter notebook that includes: data loading, preprocessing helpers, EDA functions, statistical tests, a baseline classifier (RandomForest) with evaluation and feature importances, and concise business recommendations.
- `requirements.txt` — Python packages required to run the notebook.

Data source
- The original dataset is loaded from: https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/001/125/original/aerofit_treadmill.csv?1639992749

How to run
1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
.venv\Scripts\activate     # on Windows
```

2. Install dependencies:

```bash
pip install -r Aerofit_Case_Study/requirements.txt
```

3. Open the notebook in this folder (`refactored_modeling.ipynb`) with JupyterLab / Jupyter Notebook and run the cells.

Notes & suggestions
- The notebook is structured into reusable helper functions (in a single notebook) to keep the analysis DRY and reproducible.
- If you want a Python module instead of a notebook, the helper functions can be moved into a `.py` file for reuse.
- The notebook sets a random seed for reproducibility, uses stratified splits for modeling, and includes basic checks for duplicates and outliers.

If you'd like, I can split the helper functions into a separate Python module and add more model runs (hyperparameter tuning, SHAP explanations, or a saved model artifact).
