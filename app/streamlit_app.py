import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from mindscope.config import LABELS, METRICS_PATH, MODEL_PATH, PROCESSED_DIR

REFERENCE_PATH = PROCESSED_DIR / "X_train.csv"

ORDERED_COLUMNS = {
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Diet_Quality": ["Poor", "Average", "Good"],
}

st.set_page_config(page_title="MindScope", layout="wide")


@st.cache_resource
def load_model(last_modified):
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics(last_modified):
    return json.loads(METRICS_PATH.read_text())


@st.cache_data
def load_reference(last_modified):
    return pd.read_csv(REFERENCE_PATH)


def show_performance(metrics):
    st.header("Model performance")
    st.caption(
        f"Evaluated on {metrics['n_test']} test students (the model was trained on {metrics['n_train']} students)."
    )

    st.subheader("Global metrics")
    names = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1_score": "F1-score", "roc_auc": "ROC-AUC"}
    columns = st.columns(len(names))
    for column, (key, label) in zip(columns, names.items()):
        column.metric(label, f"{metrics['metrics'][key]:.3f}")
    st.caption("Precision, recall and F1-score are averaged over the three risk levels. ROC-AUC is one-vs-rest.")

    st.subheader("Metrics by risk level")
    per_class = pd.DataFrame(metrics["per_class"]).T[["precision", "recall", "f1-score"]]
    st.dataframe(per_class.round(3))
    fig, ax = plt.subplots(figsize=(7, 4))
    per_class.plot(kind="bar", ax=ax)
    ax.set_title("Precision, recall and F1-score by risk level")
    ax.set_xlabel("Risk level")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)

    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=metrics["labels"],
        yticklabels=metrics["labels"],
        ax=ax,
    )
    ax.set_title("Confusion matrix (test set)")
    ax.set_xlabel("Predicted risk level")
    ax.set_ylabel("True risk level")
    st.pyplot(fig)

    st.subheader("Most important features")
    importance = pd.Series(metrics["feature_importance"]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Average absolute weight of each feature in the model")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)


def input_widget(name, series):
    label = name.replace("_", " ")
    if pd.api.types.is_numeric_dtype(series):
        low, high, median = series.min(), series.max(), series.median()
        if (series.dropna() % 1 == 0).all():
            return st.number_input(label, min_value=int(low), max_value=int(high), value=int(round(median)), step=1)
        return st.number_input(label, min_value=float(low), max_value=float(high), value=float(round(median, 1)), step=0.1)
    options = ORDERED_COLUMNS.get(name, sorted(series.dropna().unique()))
    return st.selectbox(label, options)


def show_prediction(model, reference):
    st.header("Test the model on new data")

    st.subheader("One student")
    st.caption("The values are limited to the ranges found in the training data.")
    with st.form("single_student"):
        columns = st.columns(3)
        values = {}
        for i, name in enumerate(reference.columns):
            with columns[i % 3]:
                values[name] = input_widget(name, reference[name])
        submitted = st.form_submit_button("Predict risk level")

    if submitted:
        row = pd.DataFrame([values])
        try:
            prediction = model.predict(row)[0]
            probabilities = pd.Series(model.predict_proba(row)[0], index=model.classes_)[LABELS]
        except Exception as error:
            st.error(f"Could not make the prediction: {error}")
        else:
            message = f"Predicted risk level: {prediction}"
            if prediction == "High":
                st.error(message)
            elif prediction == "Medium":
                st.warning(message)
            else:
                st.success(message)
            fig, ax = plt.subplots(figsize=(5, 3))
            probabilities.plot(kind="bar", ax=ax)
            ax.set_title("Probability of each risk level")
            ax.set_xlabel("Risk level")
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=0)
            st.pyplot(fig)

    st.subheader("Several students (CSV file)")
    st.download_button("Download an example file", reference.head(5).to_csv(index=False), "example_students.csv", "text/csv")
    uploaded = st.file_uploader("Upload a CSV file with the same columns as the example file", type="csv")

    if uploaded is not None:
        try:
            new_data = pd.read_csv(uploaded)
            missing_columns = [name for name in reference.columns if name not in new_data.columns]
            if missing_columns:
                st.error(f"Missing columns: {', '.join(missing_columns)}")
            else:
                features = new_data[reference.columns]
                predictions = pd.Series(model.predict(features), name="Predicted_Risk")
                probabilities = (
                    pd.DataFrame(model.predict_proba(features), columns=model.classes_)[LABELS]
                    .round(3)
                    .add_prefix("Probability_")
                )
                result = pd.concat([new_data, predictions, probabilities], axis=1)
                st.dataframe(result)
                counts = predictions.value_counts().reindex(LABELS, fill_value=0)
                st.write("Number of students by predicted risk level:")
                st.dataframe(counts.rename("Students"))
                st.download_button("Download the predictions", result.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as error:
            st.error(f"Could not read the file or make the predictions: {error}")


st.title("MindScope - Student Mental Health Risk")
st.warning(
    "Educational prototype only. This model is not a diagnostic tool and must never be used "
    "to make decisions about real students."
)

missing_files = [path.name for path in (MODEL_PATH, METRICS_PATH, REFERENCE_PATH) if not path.exists()]
if missing_files:
    st.error(
        f"Missing files: {', '.join(missing_files)}. "
        "Run the Airflow DAG 'student_risk_pipeline' first."
    )
    st.stop()

model = load_model(MODEL_PATH.stat().st_mtime)
metrics = load_metrics(METRICS_PATH.stat().st_mtime)
reference = load_reference(REFERENCE_PATH.stat().st_mtime)

page = st.sidebar.radio("Navigation", ["Model performance", "Test the model"])
st.sidebar.caption(f"Model: {metrics['model_name']}")
st.sidebar.caption(f"Last training: {metrics['trained_at']}")

if page == "Model performance":
    show_performance(metrics)
else:
    show_prediction(model, reference)