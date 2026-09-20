# Customer Clustering ML - K-Means

A machine-learning project that groups mall customers into segments with **K-Means clustering**,
validates the model on unseen customers, and **predicts the segment of a new customer** in an
interactive Streamlit app.

## What makes it complete
- **Train / Test split** (160 / 40 customers) - the model is evaluated on customers it never saw
- **K chosen with 5-fold cross-validation** (silhouette) on the training set only - no data leakage
- **Test metrics**: silhouette, Davies-Bouldin, Calinski-Harabasz, compared with a random-label baseline
- **Stability check**: 30 bootstrap re-trainings compared with the final clusters (Adjusted Rand Index)
- **Saved model** (`models/kmeans_ml_model.joblib`) + metrics (`models/ml_metrics.json`)
- **24 automated tests** (`test_ml_model.py`)
- **Prediction**: single customer (sliders + examples), prediction log, batch CSV prediction
- **Auto-recovery**: if the saved model cannot be used (e.g. different scikit-learn version), the app trains
  itself on start-up, so there is never a "model not found" error

## Dataset
`dataset.csv` - 200 mall customers: `CustomerID, Gender, Age, Education, Marital Status,
Annual Income (k$), Spending Score (1-100)`.
Clustering uses `Age`, `Annual Income (k$)` and `Spending Score (1-100)`; the categorical columns are used
afterwards to describe the segments.

## Project structure
```
kmeans_ml_app.py     Streamlit app (UI)
ml_pipeline.py       cleaning, training, testing, prediction logic
train_ml_model.py    trains, tests and saves the model
test_ml_model.py     automated tests
dataset.csv          data
models/              saved model + metrics
requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
python train_ml_model.py                 # optional: re-train and re-save the model
python -m unittest test_ml_model -v      # optional: run the tests
streamlit run kmeans_ml_app.py
```

## Deploy on Streamlit Community Cloud
1. Upload all files (including the `models` and `.streamlit` folders) to a GitHub repository.
2. Open https://share.streamlit.io -> **Create app** -> choose the repo and branch `main`.
3. Set **Main file path** to `kmeans_ml_app.py`.
4. In **Advanced settings** choose **Python 3.12**, then **Deploy**.

## Tech stack
Python, pandas, NumPy, scikit-learn, Plotly, Streamlit
