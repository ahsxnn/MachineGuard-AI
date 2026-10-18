from pathlib import Path
import io
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "machineguard_random_forest.joblib"
TARGET = "Machine failure"
NUMERIC_FEATURES = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
]
CATEGORICAL_FEATURES = ["Type"]
EXPECTED_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
FAILURE_SUBTYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
ID_COLUMNS = ["UDI", "UID", "Product ID"]

st.set_page_config(
    page_title="MachineGuard AI | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: #0b1220; color: #e7edf7; }
    [data-testid="stSidebar"] { background: #101a2b; border-right: 1px solid #25334a; }
    [data-testid="stMetric"] { background: linear-gradient(135deg,#15243a,#101a2b); padding: 16px 18px; border: 1px solid #263750; border-radius: 14px; }
    [data-testid="stMetricLabel"] { color: #9fb1c9; }
    [data-testid="stMetricValue"] { color: #f3f7ff; }
    h1, h2, h3 { color: #f4f7fb; }
    .mg-subtitle { color:#9fb1c9; margin-top:-10px; }
    .mg-card { background:#111d30; border:1px solid #263750; border-radius:14px; padding:18px; }
    .mg-note { color:#9fb1c9; font-size:0.88rem; }
    div[data-testid="stAlert"] { border-radius: 12px; }
    .stButton>button { border-radius:9px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and normalize common aliases without changing source values."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    aliases = {
        "Air temperature": "Air temperature [K]",
        "Process temperature": "Process temperature [K]",
        "Rotational speed": "Rotational speed [rpm]",
        "Torque": "Torque [Nm]",
        "Tool wear": "Tool wear [min]",
        "machine failure": "Machine failure",
        "Machine_Failure": "Machine failure",
        "Product_ID": "Product ID",
    }
    return out.rename(columns={c: aliases[c] for c in out.columns if c in aliases})


def validate_data(df: pd.DataFrame):
    problems, notes = [], []
    if df.empty:
        problems.append("The CSV contains no rows.")
    if TARGET not in df.columns:
        problems.append(f"Required target column '{TARGET}' was not found.")
    missing_features = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing_features:
        problems.append("Missing required feature columns: " + ", ".join(missing_features))
    if problems:
        return False, problems, notes, df

    out = df.copy()
    for col in NUMERIC_FEATURES + [TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    before = len(out)
    out = out.dropna(subset=EXPECTED_FEATURES + [TARGET]).copy()
    dropped = before - len(out)
    if dropped:
        notes.append(f"Removed {dropped:,} row(s) with missing or non-numeric required values.")
    if out.empty:
        problems.append("No valid rows remain after checking required values.")
        return False, problems, notes, out

    invalid_target = ~out[TARGET].isin([0, 1])
    if invalid_target.any():
        problems.append("The target must contain only 0 (no failure) and 1 (failure).")
    if not out["Type"].astype(str).isin(["L", "M", "H"]).all():
        notes.append("Some Type values are outside the expected L/M/H categories; the model can still encode them.")
    if out[NUMERIC_FEATURES].lt(0).any().any():
        notes.append("Negative numeric sensor values were found. Verify they are physically valid.")
    if out[TARGET].nunique() < 2:
        problems.append("Training requires both target classes (0 and 1).")
    out["Type"] = out["Type"].astype(str)
    out[TARGET] = out[TARGET].astype(int)
    return len(problems) == 0, problems, notes, out


def make_pipeline(random_state=42, n_estimators=350, max_depth=None, min_samples_leaf=1):
    prep = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=None if max_depth == 0 else int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        class_weight="balanced_subsample",
        random_state=int(random_state),
        n_jobs=-1,
    )
    return Pipeline([("preprocess", prep), ("classifier", model)])


def feature_importance_frame(pipeline):
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["classifier"]
    names = list(pre.get_feature_names_out())
    values = model.feature_importances_
    frame = pd.DataFrame({"Feature": names, "Importance": values})
    frame["Feature"] = frame["Feature"].str.replace("categorical__", "", regex=False).str.replace("numeric__", "", regex=False)
    return frame.sort_values("Importance", ascending=False)


def get_feature_frame(df):
    return df[EXPECTED_FEATURES].copy()


# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ MachineGuard AI")
    st.caption("Predictive Maintenance Intelligence")
    page = st.radio(
        "WORKSPACE",
        ["Executive Overview", "Data Explorer", "Model Laboratory", "Risk Assistant", "Research & Methodology"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Dataset source**")
    st.markdown("[UCI AI4I 2020 Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)")
    uploaded = st.file_uploader("Upload AI4I-compatible CSV", type=["csv"], help="Expected columns include Type, sensor measurements, and Machine failure.")
    st.caption("CSV upload applies to this session. No data is sent to an external service.")

# Load uploaded data or local default
df_raw = None
source_name = "No dataset loaded"
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        source_name = uploaded.name
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
elif (APP_DIR / "data" / "ai4i2020.csv").exists():
    try:
        df_raw = pd.read_csv(APP_DIR / "data" / "ai4i2020.csv")
        source_name = "data/ai4i2020.csv"
    except Exception:
        df_raw = None

if df_raw is not None:
    df_raw = normalize_columns(df_raw)
    valid, errors, notes, df = validate_data(df_raw)
else:
    valid, errors, notes, df = False, [], [], pd.DataFrame()

# Header
st.markdown("# MachineGuard AI")
st.markdown('<p class="mg-subtitle">Predict machine failure before it disrupts production.</p>', unsafe_allow_html=True)
st.markdown("---")

if df_raw is None:
    st.info("Upload the UCI AI4I CSV in the sidebar to begin. Place the downloaded file at `data/ai4i2020.csv` to load it automatically next time.")
    st.markdown("### Getting started")
    st.markdown("""
    1. Download `ai4i2020.csv` from the [UCI dataset page](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
    2. Upload it using the sidebar, or save it to the project's `data/` directory.
    3. Open **Data Explorer** to inspect data quality and **Model Laboratory** to train the classifier.
    """)
    st.stop()

with st.expander(f"Data validation · {source_name}", expanded=not valid):
    if valid:
        st.success(f"Validation passed: {len(df):,} usable rows and all required columns are present.")
    else:
        st.error("The uploaded file is not ready for modeling.")
        for item in errors:
            st.write(f"• {item}")
    for item in notes:
        st.warning(item)
    st.caption("Validation checks required columns, numeric conversion, missing values, and binary target labels.")

if not valid:
    st.stop()

failures = int(df[TARGET].sum())
total = len(df)
failure_rate = failures / total if total else 0
healthy = total - failures

if page == "Executive Overview":
    st.subheader("Plant health overview")
    st.caption("Historical dataset analytics — not a live industrial sensor connection.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records analyzed", f"{total:,}")
    c2.metric("Failure events", f"{failures:,}")
    c3.metric("Observed failure rate", f"{failure_rate:.2%}")
    c4.metric("Sensor features", f"{len(EXPECTED_FEATURES):,}")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("#### Machine outcome distribution")
        outcomes = pd.DataFrame({"Outcome": ["No failure", "Failure"], "Records": [healthy, failures]})
        fig = px.pie(outcomes, names="Outcome", values="Records", hole=0.66,
                     color="Outcome", color_discrete_map={"No failure": "#20c997", "Failure": "#ff6b6b"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Records by product type")
        by_type = df.groupby("Type", as_index=False).agg(Records=(TARGET, "size"), Failures=(TARGET, "sum"))
        by_type["Failure rate"] = by_type["Failures"] / by_type["Records"]
        fig = px.bar(by_type, x="Type", y="Records", color="Type", text="Records",
                     color_discrete_map={"L": "#36a2eb", "M": "#8b7cf6", "H": "#f6bd60"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Operating condition profiles")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig = px.histogram(df, x="Air temperature [K]", color=df[TARGET].map({0: "No failure", 1: "Failure"}),
                           barmode="overlay", opacity=0.72, nbins=35,
                           color_discrete_map={"No failure": "#36a2eb", "Failure": "#ff6b6b"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title="Outcome")
        st.plotly_chart(fig, use_container_width=True)
    with chart_col2:
        scatter_df = df.sample(min(2500, len(df)), random_state=42).copy()
        scatter_df["Outcome"] = scatter_df[TARGET].map({0: "No failure", 1: "Failure"})
        fig = px.scatter(scatter_df,
                         x="Rotational speed [rpm]", y="Torque [Nm]", color="Outcome",
                         size="Tool wear [min]", hover_data=["Type", "Air temperature [K]", "Process temperature [K]"],
                         color_discrete_map={"No failure": "#36a2eb", "Failure": "#ff6b6b"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title="Outcome")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Sensor summary")
    st.dataframe(df[NUMERIC_FEATURES].describe().T.round(2), use_container_width=True)

elif page == "Data Explorer":
    st.subheader("Data explorer & quality checks")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", f"{df.shape[1]:,}")
    c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    st.markdown("#### Filter records")
    type_options = sorted(df["Type"].astype(str).unique().tolist())
    selected_types = st.multiselect("Product type", type_options, default=type_options)
    outcome_choice = st.selectbox("Failure outcome", ["All", "No failure (0)", "Failure (1)"])
    view = df[df["Type"].astype(str).isin(selected_types)]
    if outcome_choice == "No failure (0)":
        view = view[view[TARGET] == 0]
    elif outcome_choice == "Failure (1)":
        view = view[view[TARGET] == 1]
    st.dataframe(view.head(500), use_container_width=True, height=360)
    st.caption(f"Showing {min(500, len(view)):,} of {len(view):,} filtered records.")
    st.download_button("Download filtered data", data=view.to_csv(index=False).encode("utf-8"),
                       file_name="machineguard_filtered_data.csv", mime="text/csv")
    st.markdown("#### Data quality report")
    quality = pd.DataFrame({
        "Column": df_raw.columns,
        "Data type": [str(df_raw[c].dtype) for c in df_raw.columns],
        "Missing values": [int(df_raw[c].isna().sum()) for c in df_raw.columns],
        "Unique values": [int(df_raw[c].nunique(dropna=True)) for c in df_raw.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.markdown("#### Failure rate by product type")
    type_stats = df.groupby("Type", as_index=False).agg(Records=(TARGET, "size"), Failures=(TARGET, "sum"))
    type_stats["Failure rate (%)"] = type_stats["Failures"] / type_stats["Records"] * 100
    st.dataframe(type_stats.round(3), use_container_width=True, hide_index=True)

elif page == "Model Laboratory":
    st.subheader("Random Forest model laboratory")
    st.write("Train a reproducible binary classifier and evaluate it on a held-out test set.")
    st.warning("AI4I is highly imbalanced. Accuracy alone can be misleading; prioritize failure-class recall, precision, F1-score, and ROC-AUC.")
    with st.form("training_form"):
        col1, col2, col3, col4 = st.columns(4)
        test_size = col1.slider("Test set (%)", 10, 40, 25, 5) / 100
        n_estimators = col2.slider("Trees", 100, 700, 350, 50)
        max_depth = col3.selectbox("Maximum depth", ["Unlimited", 8, 12, 18, 24], index=0)
        min_leaf = col4.selectbox("Minimum leaf samples", [1, 2, 4, 6], index=0)
        random_state = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
        train_clicked = st.form_submit_button("Train and evaluate model", type="primary", use_container_width=True)

    if train_clicked:
        X = get_feature_frame(df)
        y = df[TARGET]
        if y.value_counts().min() < 2:
            st.error("At least two examples of each target class are required for a stratified split.")
        else:
            with st.spinner("Training Random Forest and calculating evaluation metrics..."):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=int(random_state), stratify=y
                )
                depth = 0 if max_depth == "Unlimited" else int(max_depth)
                pipeline = make_pipeline(random_state, n_estimators, depth, min_leaf)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                metrics = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "F1-score": f1_score(y_test, y_pred, zero_division=0),
                    "ROC-AUC": roc_auc_score(y_test, y_prob) if y_test.nunique() == 2 else float("nan"),
                }
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                artifact = {
                    "pipeline": pipeline,
                    "features": EXPECTED_FEATURES,
                    "target": TARGET,
                    "metrics": metrics,
                    "random_state": int(random_state),
                    "test_size": test_size,
                    "training_rows": len(X_train),
                    "test_rows": len(X_test),
                }
                joblib.dump(artifact, MODEL_PATH)
                st.session_state["mg_metrics"] = metrics
                st.session_state["mg_confusion"] = confusion_matrix(y_test, y_pred, labels=[0, 1])
                st.session_state["mg_report"] = classification_report(y_test, y_pred, labels=[0, 1], target_names=["No failure", "Failure"], output_dict=True, zero_division=0)
                st.session_state["mg_y_test"] = y_test.to_numpy()
                st.session_state["mg_y_prob"] = y_prob
                st.session_state["mg_importance"] = feature_importance_frame(pipeline)
                st.session_state["mg_artifact_info"] = artifact
                st.success(f"Training complete. Model saved to `{MODEL_PATH.relative_to(APP_DIR)}`.")

    if "mg_metrics" in st.session_state:
        metrics = st.session_state["mg_metrics"]
        st.markdown("#### Held-out test performance")
        cards = st.columns(5)
        for card, name in zip(cards, ["Precision", "Recall", "F1-score", "ROC-AUC", "Accuracy"]):
            card.metric(name, f"{metrics[name]:.3f}")
        st.caption("Metrics are calculated on the held-out test set. Results depend on the selected split and hyperparameters.")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Confusion matrix")
            cm = st.session_state["mg_confusion"]
            fig = px.imshow(cm, text_auto=True, x=["Predicted no failure", "Predicted failure"],
                            y=["Actual no failure", "Actual failure"], color_continuous_scale="Blues",
                            labels={"x": "", "y": "", "color": "Count"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("#### ROC curve")
            yt, yp = st.session_state["mg_y_test"], st.session_state["mg_y_prob"]
            fpr, tpr, _ = roc_curve(yt, yp)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Random Forest (AUC={metrics['ROC-AUC']:.3f})", line=dict(width=3, color="#20c997")))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance baseline", line=dict(dash="dash", color="#8795a8")))
            fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                              template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Classification report")
        report = st.session_state["mg_report"]
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.round(3), use_container_width=True)
        report_text = [
            "MachineGuard AI — Model Evaluation Report",
            "==========================================",
            f"Dataset source: {source_name}",
            f"Usable rows: {len(df)}",
            f"Training rows: {st.session_state['mg_artifact_info']['training_rows']}",
            f"Held-out test rows: {st.session_state['mg_artifact_info']['test_rows']}",
            f"Random seed: {st.session_state['mg_artifact_info']['random_state']}",
            f"Test fraction: {st.session_state['mg_artifact_info']['test_size']:.0%}",
            "",
            "Metrics:",
            *[f"{key}: {value:.4f}" for key, value in metrics.items()],
            "",
            "Confusion matrix (rows=actual [0,1], columns=predicted [0,1]):",
            str(st.session_state["mg_confusion"]),
            "",
            "Classification report:",
            report_df.round(4).to_string(),
            "",
            "Research question: How effectively can a Random Forest classifier identify machine failure from operating-condition measurements in the synthetic AI4I 2020 dataset?",
            "Methodology: data validation; predictor/target separation; stratified train/test split; class-weighted Random Forest; held-out evaluation using precision, recall, F1-score and ROC-AUC.",
            "Leakage prevention: identifiers and failure subtype columns are not used as predictors.",
            "Limitations: AI4I 2020 is synthetic. Results are a demonstration and are not validated for real-world industrial safety decisions.",
        ]
        st.download_button("Download model evaluation report (.txt)", data="\n".join(report_text).encode("utf-8"),
                           file_name="machineguard_model_report.txt", mime="text/plain")
        st.markdown("#### Feature importance")
        importance = st.session_state["mg_importance"].head(10).sort_values("Importance")
        fig = px.bar(importance, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Teal")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        report_lines = [
            "MachineGuard AI — Model Evaluation Report",
            f"Dataset source: {source_name}",
            f"Rows used: {len(df)}",
            f"Target: {TARGET}",
            f"Predictors: {', '.join(EXPECTED_FEATURES)}",
            f"Training rows: {st.session_state['mg_artifact_info']['training_rows']}",
            f"Test rows: {st.session_state['mg_artifact_info']['test_rows']}",
            "", "Held-out test metrics:",
        ]
        report_lines.extend([f"{name}: {value:.4f}" for name, value in metrics.items()])
        report_lines.extend(["", "Confusion matrix (rows=actual [0,1], columns=predicted [0,1]):", str(st.session_state["mg_confusion"]),
                             "", "Limitations: AI4I 2020 is synthetic; results do not establish real-world predictive validity. This is not a validated safety system."])
        st.download_button("Download model evaluation report (.txt)", data="\n".join(report_lines),
                           file_name="machineguard_model_report.txt", mime="text/plain")
        with open(MODEL_PATH, "rb") as model_file:
            st.download_button("Download trained model (.joblib)", data=model_file.read(),
                               file_name="machineguard_random_forest.joblib", mime="application/octet-stream")
    else:
        st.info("Configure the training settings above and select **Train and evaluate model** to see results.")

elif page == "Risk Assistant":
    st.subheader("Machine failure risk assistant")
    st.write("Estimate failure risk for a single set of operating conditions using a trained model.")
    if not MODEL_PATH.exists():
        st.info("Train a model first in **Model Laboratory**. The saved model will then be available here.")
    else:
        try:
            artifact = joblib.load(MODEL_PATH)
            pipeline = artifact["pipeline"]
            st.success(f"Loaded trained model · {artifact.get('training_rows', '?'):,} training rows")
            with st.form("risk_form"):
                c1, c2, c3 = st.columns(3)
                machine_type = c1.selectbox("Product type", ["L", "M", "H"], index=1)
                air = c1.number_input("Air temperature (K)", min_value=250.0, max_value=350.0, value=300.0, step=0.1)
                process = c2.number_input("Process temperature (K)", min_value=250.0, max_value=400.0, value=310.0, step=0.1)
                speed = c2.number_input("Rotational speed (rpm)", min_value=0, max_value=10000, value=1500, step=25)
                torque = c3.number_input("Torque (Nm)", min_value=0.0, max_value=200.0, value=40.0, step=0.5)
                wear = c3.number_input("Tool wear (min)", min_value=0, max_value=1000, value=100, step=5)
                threshold = st.slider("High-risk alert threshold (%)", 5, 90, 35, 5) / 100
                predict_clicked = st.form_submit_button("Assess machine risk", type="primary", use_container_width=True)
            if predict_clicked:
                row = pd.DataFrame([{
                    "Type": machine_type,
                    "Air temperature [K]": air,
                    "Process temperature [K]": process,
                    "Rotational speed [rpm]": speed,
                    "Torque [Nm]": torque,
                    "Tool wear [min]": wear,
                }], columns=EXPECTED_FEATURES)
                probability = float(pipeline.predict_proba(row)[0, 1])
                prediction = int(probability >= threshold)
                risk = "HIGH" if prediction else ("MODERATE" if probability >= threshold * 0.6 else "LOW")
                color = {"HIGH": "#ff6b6b", "MODERATE": "#f6bd60", "LOW": "#20c997"}[risk]
                st.markdown(f"""
                <div class="mg-card">
                  <div style="color:#9fb1c9;font-size:0.9rem">MODEL-ESTIMATED FAILURE PROBABILITY</div>
                  <div style="font-size:2.6rem;font-weight:750;color:{color}">{probability:.1%}</div>
                  <div style="font-size:1.1rem;font-weight:650;color:{color}">{risk} RISK</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(min(1.0, max(0.0, probability)))
                st.markdown("#### Input-based explanation")
                importance_frame = feature_importance_frame(pipeline)
                st.write("The chart below shows which features influenced the trained forest most overall. It is global feature importance, not a causal explanation of this individual prediction.")
                top_importance = importance_frame.head(6).sort_values("Importance")
                importance_fig = px.bar(top_importance, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Teal")
                importance_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
                st.plotly_chart(importance_fig, use_container_width=True)

                recommendations = []
                if wear >= 200:
                    recommendations.append("Tool wear is high (200 minutes or more): inspect tool condition and compare against the manufacturer's replacement interval.")
                if torque >= 50:
                    recommendations.append("Torque is elevated (50 Nm or more): check load conditions and review whether the operating point is expected for this machine.")
                if speed <= 1200:
                    recommendations.append("Rotational speed is relatively low (1,200 rpm or less): compare against the machine's normal operating range and investigate unexpected slowdowns.")
                if process - air < 8:
                    recommendations.append("The process-to-air temperature difference is small (under 8 K): verify sensor readings and compare with normal process conditions.")
                if process - air > 15:
                    recommendations.append("The process-to-air temperature difference is large (over 15 K): review cooling and thermal operating conditions against equipment specifications.")
                if prediction:
                    recommendations.insert(0, "Model probability exceeds the selected alert threshold: arrange a qualified inspection and review recent maintenance and sensor history.")
                elif risk == "MODERATE":
                    recommendations.insert(0, "Risk is near the selected alert threshold: monitor subsequent readings and consider a routine inspection if the pattern persists.")
                if not recommendations:
                    recommendations.append("No configured rule-based alert was triggered. Continue routine monitoring and follow the manufacturer's maintenance schedule.")
                st.markdown("#### Cautious maintenance recommendations")
                for item in recommendations:
                    st.write(f"- {item}")
                st.caption("Recommendations are simple illustrative rules, not diagnoses or instructions to override site procedures. Confirm actions with qualified maintenance personnel.")

                risk_report = [
                    "MachineGuard AI — Single-Machine Risk Report",
                    "============================================",
                    f"Product type: {machine_type}",
                    f"Air temperature (K): {air}",
                    f"Process temperature (K): {process}",
                    f"Rotational speed (rpm): {speed}",
                    f"Torque (Nm): {torque}",
                    f"Tool wear (min): {wear}",
                    f"Estimated failure probability: {probability:.4%}",
                    f"Selected alert threshold: {threshold:.2%}",
                    f"Risk label: {risk}",
                    "",
                    "Top global model feature importances:",
                    importance_frame.head(10).to_string(index=False),
                    "",
                    "Rule-based recommendations:",
                    *[f"- {item}" for item in recommendations],
                    "",
                    "Disclaimer: AI4I 2020 is synthetic data. This estimate is a demonstration, not a validated real-world safety system. Do not use it as the sole basis for operational or maintenance decisions.",
                ]
                st.download_button("Download machine risk report (.txt)", data="\n".join(risk_report).encode("utf-8"),
                                   file_name="machineguard_risk_report.txt", mime="text/plain")
                st.caption("This is a model estimate from historical synthetic data, not a guarantee of future failure or a substitute for industrial safety procedures.")
        except Exception as exc:
            st.error(f"Could not load the saved model: {exc}")

elif page == "Research & Methodology":
    st.subheader("Research question and methodology")
    st.markdown("**Research question:** How effectively can a Random Forest classifier identify machine failure from operating-condition measurements in the synthetic UCI AI4I 2020 dataset?")
    st.markdown("#### Methodology")
    st.markdown("""
    1. **Data acquisition:** obtain the AI4I 2020 CSV from the UCI Machine Learning Repository.
    2. **Validation:** verify required columns, convert sensor values to numeric types, inspect missing values, and check that the target is binary.
    3. **Predictors and leakage prevention:** use product type and five sensor measurements. Exclude row identifiers and failure subtype labels from predictors.
    4. **Split:** create a reproducible stratified train/test split to preserve the rare failure class in both partitions.
    5. **Model and imbalance:** fit a Random Forest with `class_weight=balanced_subsample`; report minority-class metrics rather than relying on accuracy alone.
    6. **Evaluation:** report precision, recall, F1-score, confusion matrix, and ROC-AUC where both classes are present. Accuracy is supplementary.
    7. **Interpretation:** inspect global feature importance and apply transparent operating-threshold rules for cautious maintenance suggestions.
    """)
    st.markdown("#### Dataset citation")
    st.code("AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C", language="text")
    st.markdown("#### Limitations and responsible use")
    st.markdown("""
    - The AI4I dataset is synthetic and does not establish performance on actual industrial equipment.
    - Class imbalance means failures can be missed even when overall accuracy looks high; inspect false negatives and failure-class recall.
    - Global feature importance is not causal and does not fully explain an individual prediction.
    - Model probability is not necessarily calibrated to real-world failure probability.
    - The application is an academic demonstration, not a validated real-world safety system. Do not use it as the sole basis for operational decisions.
    """)
    st.markdown("#### Setup commands")
    st.code(r"python -m venv .venv\n.venv\Scripts\Activate.ps1\npip install -r requirements.txt\nstreamlit run app.py", language="powershell")
    st.markdown("[UCI dataset page](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) · [Reference design](https://multiqos.com/portfolio/ai-predictive-maintenance-system/)")

st.markdown("---")
st.caption("MachineGuard AI · Academic prototype · UCI AI4I 2020 dataset · Not connected to live industrial equipment.")
