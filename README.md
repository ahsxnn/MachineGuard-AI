# MachineGuard AI — Predictive Maintenance and Machine Failure Risk Assistant

A Master's-level academic prototype built with Python, Streamlit, pandas, scikit-learn, Plotly, and joblib.

## Features

- Industrial-style dashboard with dataset KPIs and sensor visualizations
- CSV upload, column validation, numeric checks, and basic data-quality report
- Data explorer with product-type and outcome filters, plus CSV export
- Random Forest classifier with class weighting and configurable hyperparameters
- Held-out evaluation: precision, recall, F1-score, accuracy, ROC-AUC, confusion matrix, ROC curve, classification report, and feature importance
- Single-record risk assessment using the saved model
- Downloadable joblib model artifact
- Downloadable text reports for model evaluation and individual machine risk assessments
- Transparent rule-based maintenance suggestions and global feature-importance chart
- Research question, methodology, dataset citation, limitations, and setup instructions in the app

## Dataset

Download `ai4i2020.csv` from the [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).

Put the file at `data/ai4i2020.csv` for automatic loading, or upload it in the app sidebar.

The dataset is synthetic and contains 10,000 records. Cite it in your report:

> AI4I 2020 Predictive Maintenance Dataset. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Workflow

1. Download and upload the CSV.
2. Review data validation and use Data Explorer to inspect the dataset.
3. Open Model Laboratory, configure hyperparameters, and train.
4. Review held-out metrics, confusion matrix, ROC curve, and feature importance.
5. Open Risk Assistant to assess one machine condition at a time.
6. Download the trained `joblib` model if needed.

## Modeling notes and academic considerations

- Predictors are `Type`, air temperature, process temperature, rotational speed, torque, and tool wear.
- The target is `Machine failure`.
- Identifier fields (`UDI`/`UID`, `Product ID`) and failure subtype labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are deliberately excluded from predictors to avoid target leakage.
- A stratified holdout split is used. `class_weight="balanced_subsample"` helps address the rare failure class, but does not eliminate the need to inspect precision/recall and false negatives.
- The risk assistant threshold is configurable. Threshold tuning should be done on validation data and selected according to the cost of missed failures versus unnecessary inspections.
- The dataset is synthetic, not live telemetry. Performance on it does not establish production reliability.
- Feature importance is an association signal, not causal proof.
- The saved model uses joblib/pickle serialization. Only load model files from trusted sources.

## Research question

How effectively can a Random Forest classifier identify machine failure from operating-condition measurements in the synthetic UCI AI4I 2020 dataset?

## Methodology

1. Validate the CSV schema, required sensor columns, missing values, numeric fields, and binary target.
2. Use product type and five sensor measurements as predictors. Exclude identifiers and failure subtype columns to avoid leakage.
3. Create a reproducible stratified train/test split.
4. Train a Random Forest with `class_weight="balanced_subsample"` to address class imbalance.
5. Evaluate the held-out test set using precision, recall, F1-score, confusion matrix, and ROC-AUC where appropriate; accuracy is supplementary.
6. Interpret global feature importance and provide cautious rule-based suggestions for operating measurements.

## Limitations

AI4I 2020 is synthetic. Performance on this dataset is not evidence of real-world predictive validity. Failure probability may not be calibrated, feature importance is not causal, and simple threshold-based maintenance rules are illustrative. This application is an academic demonstration, not a validated safety system. Do not use it as the sole basis for operational decisions.

## Suggested Master's report structure

1. Abstract and problem statement
2. Literature review: predictive vs. preventive maintenance
3. Dataset description and exploratory analysis
4. Data validation and leakage prevention
5. Methodology and model design
6. Experimental setup and metrics
7. Results, limitations, and error analysis
8. Risk assistant and dashboard design
9. Ethical, safety, and deployment considerations
10. Conclusion and future work

## Project structure

```text
MachineGuard_AI/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── (place ai4i2020.csv here)
└── models/
    └── (trained model is saved here)
```
