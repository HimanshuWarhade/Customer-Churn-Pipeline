"""
streamlit_app.py
-----------------
A visually polished dashboard for the churn model: a sidebar form for
customer details, a gauge chart + risk badge for the result, and a
feature-importance chart giving business context on what drives churn
generally. All prediction logic still lives in src/predict.py -- this
file only handles layout, input, and display.

Run with (from the project root):
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from predict import predict_churn
from train import MODELS_DIR, PROCESSED_DATA_PATH

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

# -----------------------------------------------------------------------
# Custom styling. Streamlit's defaults are fine but generic; a handful
# of CSS rules make this look like a built dashboard rather than a
# tutorial form. `unsafe_allow_html=True` is required for Streamlit to
# actually render raw HTML/CSS instead of showing it as text.
# -----------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
    }
    .risk-badge {
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        font-size: 1.1rem;
    }
    .risk-high   { background-color: #fee2e2; color: #b91c1c; }
    .risk-medium { background-color: #fef3c7; color: #b45309; }
    .risk-low    { background-color: #dcfce7; color: #15803d; }
    .outcome-box {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
    }
    .outcome-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.2rem;
    }
    .outcome-value {
        font-size: 1.5rem;
        font-weight: 700;
    }
    .outcome-churn { color: #dc2626; }
    .outcome-stay  { color: #16a34a; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------
# Cached loaders -- @st.cache_resource means the model/importance data
# load from disk ONCE per app session, not on every single form submit
# (Streamlit re-runs the whole script top-to-bottom on every interaction,
# so this caching is what keeps the app fast).
# -----------------------------------------------------------------------
@st.cache_resource
def get_feature_importance():
    """Returns the model's top 10 most influential features, if the
    model type supports it (tree-based models expose feature_importances_;
    linear models expose coef_ instead)."""
    model = joblib.load(MODELS_DIR / "churn_model.pkl")
    columns = pd.read_csv(PROCESSED_DATA_PATH, nrows=0).columns.tolist()
    columns = [c for c in columns if c != "Exited"]

    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        scores = abs(model.coef_[0])
    else:
        return None

    importance_df = pd.DataFrame({"Feature": columns, "Importance": scores})
    return importance_df.sort_values("Importance", ascending=False).head(10)


def render_gauge(probability: float) -> go.Figure:
    """A gauge chart reads at a glance in a way a plain percentage
    number doesn't -- useful when a retention rep is scanning many
    customers quickly."""
    if probability >= 0.6:
        bar_color = "#dc2626"
    elif probability >= 0.3:
        bar_color = "#d97706"
    else:
        bar_color = "#16a34a"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%"},
        title={"text": "Churn Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, 30], "color": "#dcfce7"},
                {"range": [30, 60], "color": "#fef3c7"},
                {"range": [60, 100], "color": "#fee2e2"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.markdown('<h1 class="main-header">📉 Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Estimate a customer\'s churn risk so the retention '
    'team can act before they leave.</p>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------
# Sidebar: input form. Living in the sidebar keeps the main area free
# for results -- a common layout for "input -> result" tools.
# -----------------------------------------------------------------------
with st.sidebar:
    st.header("Customer Details")
    with st.form("customer_form"):
        credit_score = st.number_input("Credit Score", 300, 900, 650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", 18, 100, 35)
        tenure = st.number_input("Tenure (years)", 0, 15, 3)
        balance = st.number_input("Account Balance", 0.0, value=0.0, step=1000.0)
        num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active_member = st.selectbox("Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", 0.0, value=50000.0, step=1000.0)
        satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
        card_type = st.selectbox("Card Type", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"])
        point_earned = st.number_input("Points Earned", 0, value=500, step=10)

        submitted = st.form_submit_button("🔍 Predict Churn Risk", use_container_width=True)

# -----------------------------------------------------------------------
# Main area
# -----------------------------------------------------------------------
if submitted:
    customer = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active_member == "Yes" else 0,
        "EstimatedSalary": estimated_salary,
        "Satisfaction Score": satisfaction_score,
        "Card Type": card_type,
        "Point Earned": point_earned,
    }

    result = predict_churn(customer)
    probability = result["churn_probability"]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.plotly_chart(render_gauge(probability), use_container_width=True)

    with col2:
        st.write("")  # small vertical spacer to visually align with the gauge
        st.write("")
        if probability >= 0.6:
            st.markdown('<div class="risk-badge risk-high">🔴 High Risk — recommend proactive outreach</div>', unsafe_allow_html=True)
        elif probability >= 0.3:
            st.markdown('<div class="risk-badge risk-medium">🟠 Medium Risk — worth monitoring</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-badge risk-low">🟢 Low Risk</div>', unsafe_allow_html=True)

        # st.metric doesn't support per-value text color, so we render
        # this ourselves with the CSS classes defined at the top of the
        # file -- red for churn, green for stay.
        if result["churn_prediction"] == 1:
            outcome_class, outcome_text = "outcome-churn", "Likely to churn"
        else:
            outcome_class, outcome_text = "outcome-stay", "Likely to stay"

        st.markdown(
            f'<div class="outcome-box">'
            f'<div class="outcome-label">Predicted outcome</div>'
            f'<div class="outcome-value {outcome_class}">{outcome_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander("See raw input sent to the model"):
        st.json(customer)

else:
    st.info("👈 Fill in the customer's details in the sidebar and click **Predict Churn Risk**.")

st.divider()

# -----------------------------------------------------------------------
# Feature importance: shown regardless of whether a prediction has been
# made yet -- gives useful context on its own ("what drives churn in
# general"), not just a byproduct of a prediction.
# -----------------------------------------------------------------------
st.subheader("📊 What drives churn, overall")
importance_df = get_feature_importance()

if importance_df is not None:
    fig = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
    )
    fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Based on the trained model's feature importances across all training "
        "data -- not specific to the customer above."
    )
else:
    st.caption("Feature importance isn't available for this model type.")