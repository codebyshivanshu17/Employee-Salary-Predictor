import streamlit as st
import pandas as pd
import joblib

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💰",
    layout="wide"
)


# Sidebar
with st.sidebar:
    st.title("💰 Salary Predictor")

    st.markdown("---")

    st.markdown("""
    ### 📌 About Project

    This application predicts an estimated
    employee salary using machine learning
    and ensemble regression techniques.
    """)

    st.markdown("---")

    st.markdown("### 🤖 Models Used")

    st.write("• Random Forest")
    st.write("• Gradient Boosting")
    st.write("• Extra Trees")
    st.write("• Voting Regressor")

    st.markdown("---")

    st.markdown("### 🛠️ Technologies")

    st.write("🐍 Python")
    st.write("📊 Pandas")
    st.write("🧠 Scikit-learn")
    st.write("🌐 Streamlit")
    st.write("💾 Joblib")
# ==============================
# Load Model
# ==============================

@st.cache_resource
def load_model():
    return joblib.load("models/salary_model.pkl")


model = load_model()


# ==============================
# Title
# ==============================

st.title("💰 Employee Salary Predictor")

st.write(
    "Predict an employee's estimated salary using "
    "machine learning and ensemble techniques."
)

st.divider()


# ==============================
# Input Section
# ==============================

st.subheader("👤 Employee Information")


col1, col2 = st.columns(2)


with col1:

    work_year = st.number_input(
        "Work Year",
        min_value=2020,
        max_value=2030,
        value=2023,
        step=1
    )

    experience_level = st.selectbox(
        "Experience Level",
        [
            "EN",
            "MI",
            "SE",
            "EX"
        ]
    )

    employment_type = st.selectbox(
        "Employment Type",
        [
            "FT",
            "PT",
            "CT",
            "FL"
        ]
    )

    job_title = st.text_input(
        "Job Title",
        value="Data Scientist"
    )


with col2:

    employee_residence = st.text_input(
        "Employee Residence",
        value="US"
    )

    remote_ratio = st.slider(
        "Remote Work Ratio (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=50
    )

    company_location = st.text_input(
        "Company Location",
        value="US"
    )

    company_size = st.selectbox(
        "Company Size",
        [
            "S",
            "M",
            "L"
        ]
    )


# ==============================
# Prediction
# ==============================

st.divider()

if st.button("💰 Predict Salary", use_container_width=True):

    # Basic input validation
    if not job_title.strip():
        st.error("❌ Please enter a job title.")
        st.stop()

    if not employee_residence.strip():
        st.error("❌ Please enter employee residence.")
        st.stop()

    if not company_location.strip():
        st.error("❌ Please enter company location.")
        st.stop()

    input_data = pd.DataFrame({
        "work_year": [work_year],
        "experience_level": [experience_level],
        "employment_type": [employment_type],
        "job_title": [job_title.strip()],
        "employee_residence": [employee_residence.strip()],
        "remote_ratio": [remote_ratio],
        "company_location": [company_location.strip()],
        "company_size": [company_size]
    })

    try:
        prediction = model.predict(input_data)[0]

        # Prevent invalid model output
        if prediction < 0:
            st.error("❌ Invalid salary prediction.")
            st.stop()

        # Save prediction
        st.session_state.prediction_history.append({
            "Job Title": job_title.strip(),
            "Experience": experience_level,
            "Employment Type": employment_type,
            "Remote Work": f"{remote_ratio}%",
            "Company Size": company_size,
            "Predicted Salary": round(prediction, 2)
        })

        # Estimated presentation range
        lower_salary = prediction * 0.90
        upper_salary = prediction * 1.10

        st.divider()

        st.subheader("🎯 Salary Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Estimated Salary",
                f"${prediction:,.0f}"
            )

        with col2:
            st.metric(
                "Estimated Lower Range",
                f"${lower_salary:,.0f}"
            )

        with col3:
            st.metric(
                "Estimated Upper Range",
                f"${upper_salary:,.0f}"
            )

        st.success(
            f"🎉 Estimated annual salary for a "
            f"**{job_title.strip()}** with **{experience_level}** "
            f"experience: **${prediction:,.2f} USD**"
        )

        st.write("### 📋 Prediction Summary")

        summary = pd.DataFrame({
            "Parameter": [
                "Job Title",
                "Experience Level",
                "Employment Type",
                "Employee Residence",
                "Remote Work",
                "Company Location",
                "Company Size"
            ],
            "Value": [
                job_title.strip(),
                experience_level,
                employment_type,
                employee_residence.strip(),
                f"{remote_ratio}%",
                company_location.strip(),
                company_size
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.error(
            "❌ Prediction failed. Please check your inputs "
            "and try again."
        )
        st.caption(f"Technical details: {e}")

        # Save prediction to history
    st.session_state.prediction_history.append({
        "Job Title": job_title,
        "Experience": experience_level,
        "Employment Type": employment_type,
        "Remote Work": f"{remote_ratio}%",
        "Company Size": company_size,
        "Predicted Salary": round(prediction, 2)
    })

    # Estimated salary range
    lower_salary = prediction * 0.90
    upper_salary = prediction * 1.10

    st.divider()

    st.subheader("🎯 Salary Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Estimated Salary",
            f"${prediction:,.0f}"
        )

    with col2:
        st.metric(
            "Estimated Lower Range",
            f"${lower_salary:,.0f}"
        )

    with col3:
        st.metric(
            "Estimated Upper Range",
            f"${upper_salary:,.0f}"
        )

    st.success(
        f"🎉 Estimated annual salary for a **{job_title}** "
        f"with **{experience_level}** experience: "
        f"**${prediction:,.2f} USD**"
    )

    st.write("### 📋 Prediction Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Job Title",
            "Experience Level",
            "Employment Type",
            "Employee Residence",
            "Remote Work",
            "Company Location",
            "Company Size"
        ],
        "Value": [
            job_title,
            experience_level,
            employment_type,
            employee_residence,
            f"{remote_ratio}%",
            company_location,
            company_size
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )
    # ==============================
# Model Performance
# ==============================

st.divider()

st.subheader("📊 Model Performance")

try:
    results = pd.read_csv("src/model_results.csv")

    st.dataframe(
        results,
        use_container_width=True
    )

    st.subheader("🏆 R² Score Comparison")

    chart_data = results.set_index("Model")["R2"]

    st.bar_chart(chart_data)

except FileNotFoundError:
    st.warning(
        "Model results not found. "
        "Please run train_model.py first."
    )
    st.divider()

st.subheader("📈 Salary Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Prediction Type",
        "Salary Regression"
    )

with col2:
    st.metric(
        "ML Approach",
        "Ensemble Learning"
    )

with col3:
    st.metric(
        "Target Variable",
        "Salary (USD)"
    )

st.divider()

st.subheader("💡 Project Insights")

st.markdown("""
### 🔍 How the prediction works

The system estimates salary using multiple employee and company attributes:

- **Experience Level** – Entry, Mid, Senior, or Executive
- **Employment Type** – Full-time, Part-time, Contract, or Freelance
- **Job Title** – Employee's professional role
- **Employee Residence** – Employee's location
- **Remote Work Ratio** – Percentage of remote work
- **Company Location** – Company's location
- **Company Size** – Small, Medium, or Large

The machine learning pipeline automatically preprocesses categorical
features using **One-Hot Encoding** and combines multiple regression
models using **ensemble learning**.
""")

st.info(
    "💡 Tip: Try different combinations of experience level, job title, "
    "remote ratio, location, and company size to observe how the estimated "
    "salary changes."
)

st.divider()

st.subheader("📊 Salary Analytics")

try:
    salary_df = pd.read_csv("data/salary_data.csv")

    # Average salary by experience level
    st.write("### 💼 Average Salary by Experience Level")

    experience_salary = (
        salary_df.groupby("experience_level")["salary_in_usd"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(experience_salary)

    # Top job titles
    st.write("### 🧑‍💻 Top Job Titles by Average Salary")

    job_salary = (
        salary_df.groupby("job_title")["salary_in_usd"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(job_salary)

    # Company size
    st.write("### 🏢 Average Salary by Company Size")

    company_salary = (
        salary_df.groupby("company_size")["salary_in_usd"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(company_salary)

except FileNotFoundError:
    st.warning(
        "Salary dataset not found. Please make sure "
        "data/salary_data.csv exists."
    )

    st.divider()

st.subheader("🧠 Model Explainability")

st.write(
    "Feature importance helps understand which input variables "
    "have the strongest influence on the machine learning model."
)

try:
    best_pipeline = model

    preprocessor = best_pipeline.named_steps["preprocessor"]
    trained_model = best_pipeline.named_steps["model"]

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    # Extract feature importance
    if hasattr(trained_model, "feature_importances_"):
        importances = trained_model.feature_importances_

    elif hasattr(trained_model, "estimators_"):
        importances = sum(
            estimator.feature_importances_
            for estimator in trained_model.estimators_
        ) / len(trained_model.estimators_)

    else:
        importances = None

    if importances is not None:

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        ).head(15)

        st.write("### 🔝 Top 15 Important Features")

        st.bar_chart(
            importance_df.set_index("Feature")["Importance"]
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "Feature importance is not available for the selected model."
        )

except Exception as e:
    st.warning(
        f"Feature importance could not be displayed: {e}"
    )

    st.divider()

st.subheader("🕘 Prediction History")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑️ Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()

else:
    st.info(
        "No predictions yet. Make a salary prediction to see it here."
    )

    st.divider()

st.subheader("📥 Download Prediction Report")

if st.session_state.prediction_history:

    report_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    csv_data = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv_data,
        file_name="salary_prediction_report.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info(
        "Make at least one prediction to generate a downloadable report."
    )

    st.divider()

st.subheader("📋 Dataset Overview")

try:
    dataset = pd.read_csv("data/salary_data.csv")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Records",
            f"{len(dataset):,}"
        )

    with col2:
        st.metric(
            "Features",
            dataset.shape[1]
        )

    with col3:
        st.metric(
            "Job Titles",
            dataset["job_title"].nunique()
        )

    with col4:
        st.metric(
            "Countries",
            dataset["company_location"].nunique()
        )

except FileNotFoundError:
    st.warning("Dataset not found.")