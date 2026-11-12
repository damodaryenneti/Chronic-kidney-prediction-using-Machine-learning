
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="CKD Care | Kidney Health Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ckd_model.pkl"
KIDNEY_IMAGE = BASE_DIR / "kidney image.jpg"

PAGES = [
    "Dashboard",
    "Patient Assessment",
    "Model Insights",
    "About"
]

# =========================================================
# 2. SESSION STATE AND NAVIGATION CALLBACKS
# =========================================================
if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"

if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Dashboard"


def change_page():
    """Update the displayed page when the sidebar radio changes."""
    st.session_state["page"] = st.session_state["navigation"]


def go_to_assessment():
    """Navigate from Dashboard to Patient Assessment."""
    st.session_state["page"] = "Patient Assessment"
    st.session_state["navigation"] = "Patient Assessment"


def go_to_dashboard():
    """Navigate to Dashboard."""
    st.session_state["page"] = "Dashboard"
    st.session_state["navigation"] = "Dashboard"


def go_to_insights():
    """Navigate to Model Insights."""
    st.session_state["page"] = "Model Insights"
    st.session_state["navigation"] = "Model Insights"


# =========================================================
# 3. CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1220;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #111c30;
    }

    [data-testid="stSidebar"] * {
        color: #f1f5f9;
    }

    h1, h2, h3, h4, p, label {
        color: #f1f5f9;
    }

    .hero-card {
        background: linear-gradient(135deg, #14253d, #123c48);
        padding: 28px;
        border-radius: 18px;
        border: 1px solid #24546a;
        margin-bottom: 20px;
    }

    .hero-card h1 {
        color: #ffffff;
        font-size: 34px;
    }

    .hero-card p {
        color: #cbd5e1;
        font-size: 16px;
    }

    .info-card {
        background-color: #142136;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #293b53;
        min-height: 130px;
    }

    .info-card h3 {
        color: #5eead4;
        margin-bottom: 8px;
    }

    .info-card p {
        color: #cbd5e1;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #0d9488;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 10px 18px;
        font-weight: 600;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #14b8a6;
        color: white;
    }

    [data-testid="stMetric"] {
        background-color: #142136;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #293b53;
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1;
    }

    [data-testid="stMetricValue"] {
        color: #5eead4;
    }

    .disclaimer {
        background-color: #352b19;
        border-left: 4px solid #fbbf24;
        padding: 14px;
        border-radius: 8px;
        color: #fde68a;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 4. SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown("## 🩺 CKD Care")
    st.caption("Kidney Health Prediction System")
    st.divider()

    st.radio(
        "Navigation",
        PAGES,
        key="navigation",
        on_change=change_page
    )

    st.divider()
    st.caption("Machine Learning Screening Application")
    st.caption("For educational purposes only")

# The current page is controlled by session state.
page = st.session_state["page"]

# =========================================================
# 5. LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


model = load_model()

if model is None:
    st.error("Model file not found.")
    st.write("Expected location:")
    st.code(str(MODEL_PATH))
    st.stop()

# =========================================================
# 6. EXTRACT PIPELINE FEATURES
# =========================================================
try:
    preprocessing = model.named_steps["Data Preprocessing"]
    classifier = model.named_steps["Model Building"]

except (AttributeError, KeyError):
    st.error(
        "Your saved model does not contain the expected pipeline steps. "
        "Expected: 'Data Preprocessing' and 'Model Building'. "
        "Check the step names in your trained pipeline."
    )
    st.stop()


def get_columns(columns):
    """Convert a transformer column specification to a list."""
    if isinstance(columns, str):
        return [columns]

    if isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
        return list(columns)

    return []


def get_pipeline_columns(transformer_name):
    """Extract original columns from the fitted ColumnTransformer."""
    for name, transformer, columns in preprocessing.transformers_:
        if name == transformer_name:
            return get_columns(columns)

    return []


numeric_columns = get_pipeline_columns("Numeric")
ordinal_columns = get_pipeline_columns("Ordinal")
nominal_columns = get_pipeline_columns("Nominal")

all_columns = list(dict.fromkeys(
    numeric_columns + ordinal_columns + nominal_columns
))

if not all_columns:
    st.error(
        "No input features were found. Check your Numeric, Ordinal, "
        "and Nominal transformer names."
    )
    st.stop()


def find_encoder(transformer_name, step_name):
    """Find a fitted encoder within a named preprocessing pipeline."""
    try:
        transformer = preprocessing.named_transformers_[transformer_name]

        if hasattr(transformer, "named_steps"):
            return transformer.named_steps.get(step_name)

    except (KeyError, AttributeError, TypeError):
        pass

    return None


ordinal_encoder = find_encoder("Ordinal", "Ordinalencoding")
nominal_encoder = find_encoder("Nominal", "Nominal Encoding")


def get_categories(encoder, column, column_group):
    """Get category options learned during model training."""
    if encoder is None or not hasattr(encoder, "categories_"):
        return []

    try:
        if hasattr(encoder, "feature_names_in_"):
            feature_names = list(encoder.feature_names_in_)
            index = feature_names.index(column)
        else:
            index = column_group.index(column)

        categories = list(encoder.categories_[index])

        return [
            value for value in categories
            if not pd.isna(value)
        ]

    except (ValueError, IndexError, TypeError):
        return []


def get_input_options(column, transformer_name):
    """Get valid categorical values from the fitted encoder."""
    if transformer_name == "Ordinal":
        options = get_categories(
            ordinal_encoder, column, ordinal_columns
        )
    else:
        options = get_categories(
            nominal_encoder, column, nominal_columns
        )

    if options:
        return options

    return ["Yes", "No", "Normal", "Abnormal"]


# =========================================================
# 7. DASHBOARD
# =========================================================
if page == "Dashboard":

    st.markdown(
        """
        <div class="hero-card">
            <h1>Kidney Health Prediction</h1>
            <p>
                A machine learning application designed to support
                Chronic Kidney Disease (CKD) screening using
                patient health information.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.subheader("Welcome to CKD Care")

        st.write(
            "Use this application to enter patient health measurements, "
            "obtain a model prediction, and explore information about "
            "the prediction system."
        )

        st.markdown("### What you can do")

        st.markdown(
            """
            - Enter patient health measurements.
            - Generate a prediction using the trained ML pipeline.
            - Explore available model insights.
            - Download an assessment summary.
            """
        )

        # Callback updates both states before the next rerun.
        st.button(
            "Start Patient Assessment",
            key="dashboard_assessment",
            on_click=go_to_assessment,
            use_container_width=True
        )

        st.button(
            "Explore Model Insights",
            key="dashboard_insights",
            on_click=go_to_insights,
            use_container_width=True
        )

    with right:
        if KIDNEY_IMAGE.exists():
            st.image(
                str(KIDNEY_IMAGE),
                caption="Kidney Health",
                use_container_width=True
            )
        else:
            st.warning("Kidney image not found.")
            st.write("Expected image location:")
            st.code(str(KIDNEY_IMAGE))

    st.divider()
    st.subheader("Application Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>Patient Assessment</h3>
                <p>Enter the patient information required by the model.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>ML Prediction</h3>
                <p>Generate a classification using the saved pipeline.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h3>Model Insights</h3>
                <p>Explore the available features and model importance.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Medical disclaimer:</strong>
            This application is for educational and screening-support
            purposes only. Its output is not a medical diagnosis.
            Consult a qualified healthcare professional for medical advice.
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# 8. PATIENT ASSESSMENT
# =========================================================
elif page == "Patient Assessment":

    st.title("🧑‍⚕️ Patient Assessment")

    st.write(
        "Enter the patient information required by your trained model. "
        "Mark a measurement unavailable if you do not have its value."
    )

    with st.form("ckd_assessment_form"):

        input_values = {}

        # Numeric inputs
        if numeric_columns:
            st.markdown("### Numeric Measurements")

            numeric_layout = st.columns(2)

            for index, column in enumerate(numeric_columns):
                with numeric_layout[index % 2]:

                    missing = st.checkbox(
                        f"{column}: value unavailable",
                        key=f"missing_numeric_{column}"
                    )

                    value = st.number_input(
                        column,
                        value=0.0,
                        step=1.0,
                        format="%.4f",
                        disabled=missing,
                        key=f"numeric_{column}"
                    )

                    input_values[column] = (
                        np.nan if missing else value
                    )

        # Ordinal categorical inputs
        if ordinal_columns:
            st.markdown("### Ordinal Categorical Features")

            ordinal_layout = st.columns(2)

            for index, column in enumerate(ordinal_columns):
                with ordinal_layout[index % 2]:

                    options = get_input_options(column, "Ordinal")

                    missing = st.checkbox(
                        f"{column}: value unavailable",
                        key=f"missing_ordinal_{column}"
                    )

                    value = st.selectbox(
                        column,
                        options,
                        disabled=missing,
                        key=f"ordinal_{column}"
                    )

                    input_values[column] = (
                        np.nan if missing else value
                    )

        # Nominal categorical inputs
        if nominal_columns:
            st.markdown("### Other Categorical Features")

            nominal_layout = st.columns(2)

            for index, column in enumerate(nominal_columns):
                with nominal_layout[index % 2]:

                    options = get_input_options(column, "Nominal")

                    missing = st.checkbox(
                        f"{column}: value unavailable",
                        key=f"missing_nominal_{column}"
                    )

                    value = st.selectbox(
                        column,
                        options,
                        disabled=missing,
                        key=f"nominal_{column}"
                    )

                    input_values[column] = (
                        np.nan if missing else value
                    )

        submitted = st.form_submit_button(
            "Predict CKD Status",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [[input_values[column] for column in all_columns]],
            columns=all_columns
        )

        try:
            prediction = model.predict(input_df)[0]

            st.session_state["last_input"] = input_df.copy()
            st.session_state["last_prediction"] = str(prediction)

            st.divider()
            st.subheader("Assessment Result")

            label = str(prediction).strip().lower()

            if label in ["ckd", "1", "true"]:
                st.error("The model classified this input as CKD.")

                st.warning(
                    "This is a model output, not a confirmed diagnosis. "
                    "Consult a qualified healthcare professional."
                )

            elif label in ["notckd", "not ckd", "0", "false"]:
                st.success(
                    "The model classified this input as not CKD."
                )

                st.info(
                    "This result does not rule out kidney disease. "
                    "Seek medical advice if you have concerns."
                )

            else:
                st.info(f"Model prediction: {prediction}")

            # Probability estimates, when supported
            if hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(input_df)[0]
                    classes = model.classes_

                    probability_df = pd.DataFrame({
                        "Class": [str(c) for c in classes],
                        "Model score (%)": [
                            round(float(p) * 100, 2)
                            for p in probabilities
                        ]
                    })

                    st.markdown("### Model Scores")

                    st.dataframe(
                        probability_df,
                        hide_index=True,
                        use_container_width=True
                    )

                    st.caption(
                        "These scores are model probability estimates "
                        "and are not necessarily calibrated clinical risks."
                    )

                except Exception:
                    pass

            # Download the assessment summary
            summary_df = input_df.copy()
            summary_df["Model_Prediction"] = str(prediction)

            csv_data = summary_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Assessment Summary (CSV)",
                data=csv_data,
                file_name="ckd_assessment_summary.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as error:
            st.error(
                "Prediction failed. Check that the input values and "
                "categories match those used to train your model."
            )
            st.exception(error)

# =========================================================
# 9. MODEL INSIGHTS
# =========================================================
elif page == "Model Insights":

    st.title("📊 Model Insights")

    st.write(
        "Explore the available information from your trained CKD model."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Input Features", len(all_columns))

    with col2:
        if hasattr(classifier, "classes_"):
            st.metric("Output Classes", len(classifier.classes_))
        elif hasattr(model, "classes_"):
            st.metric("Output Classes", len(model.classes_))
        else:
            st.metric("Output Classes", "See trained model")

    st.markdown("### Input Feature Groups")

    feature_group_df = pd.DataFrame({
        "Feature Type": ["Numeric", "Ordinal", "Nominal"],
        "Number of Features": [
            len(numeric_columns),
            len(ordinal_columns),
            len(nominal_columns)
        ]
    })

    st.dataframe(
        feature_group_df,
        hide_index=True,
        use_container_width=True
    )

    st.markdown("### Input Features")
    st.write(all_columns)

    # Display feature importance for supported estimators
    if hasattr(classifier, "feature_importances_"):

        importances = classifier.feature_importances_

        try:
            feature_names = list(
                preprocessing.get_feature_names_out()
            )
        except Exception:
            feature_names = []

        if len(feature_names) != len(importances):
            feature_names = [
                f"Feature {i + 1}"
                for i in range(len(importances))
            ]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        st.markdown("### Feature Importance")

        top_features = importance_df.head(15).sort_values(
            "Importance",
            ascending=True
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            top_features["Feature"],
            top_features["Importance"]
        )

        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Top Feature Importances")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Feature importance describes model behavior; it does not "
            "establish medical causation."
        )

    else:
        st.info(
            "Feature importance is not directly available for this model."
        )

    st.markdown("### Previous Assessment")

    if "last_prediction" in st.session_state:
        st.write(
            f"Last model output: {st.session_state['last_prediction']}"
        )
    else:
        st.info("No assessment has been run in this session yet.")

    st.button(
        "Back to Dashboard",
        key="insights_dashboard",
        on_click=go_to_dashboard
    )

# =========================================================
# 10. ABOUT
# =========================================================
elif page == "About":

    st.title("ℹ️ About CKD Care")

    st.markdown(
        """
        ### Project Overview

        CKD Care demonstrates how patient health information can be
        used in a machine learning classification workflow for
        Chronic Kidney Disease screening.

        ### Main Components

        - **Data preprocessing:** uses the preprocessing steps saved
          with the trained pipeline.
        - **Classification:** generates a prediction using the saved model.
        - **Interactive dashboard:** accepts the model's required inputs.
        - **Assessment export:** downloads the input values and prediction.

        ### Important Limitations

        - Prediction quality depends on the training data and validation.
        - A model prediction is not a medical diagnosis.
        - This educational application cannot replace laboratory testing
          or professional medical evaluation.
        """
    )

    st.divider()
    st.markdown("### Required Project Files")

    st.code(
        "Chronic kidney disease/\n"
        "├── app.py\n"
        "├── ckd_model.pkl\n"
        "└── kidney image.jpg"
    )

    if KIDNEY_IMAGE.exists():
        st.success("Kidney image found.")

        st.image(
            str(KIDNEY_IMAGE),
            caption="Kidney Health",
            use_container_width=True
        )
    else:
        st.warning(
            "Kidney image not found. Expected location:"
        )
        st.code(str(KIDNEY_IMAGE))

    st.button(
        "Back to Dashboard",
        key="about_dashboard",
        on_click=go_to_dashboard
    )

# =========================================================
# 11. FOOTER
# =========================================================
st.divider()

st.caption(
    "CKD Care | Educational Machine Learning Project | "
    "Not a substitute for professional medical advice"
)
