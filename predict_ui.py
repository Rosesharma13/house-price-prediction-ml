"""
predict_ui.py — House Price Predictor
California Housing Dataset — Linear Regression
Run: streamlit run predict_ui.py
"""

import streamlit as st
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #1a3c6e;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        border: none;
    }
    .result-box {
        background: white;
        border-radius: 12px;
        padding: 24px;
        border-left: 6px solid #1a3c6e;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .price-display {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a3c6e;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def train_model():
    """Train model on startup — no pkl file needed."""
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler


# Train model on first load
with st.spinner("Loading model..."):
    model, scaler = train_model()

# ── Header ─────────────────────────────────────────────────────
st.markdown("# 🏠 California House Price Predictor")
st.markdown("*Powered by Linear Regression · Maincrafts Technology Task 1*")
st.markdown("---")

st.markdown("### Enter House Details:")

col1, col2 = st.columns(2)

with col1:
    med_inc = st.slider("💰 Median Income (x$10,000)",
                        min_value=0.5, max_value=15.0,
                        value=4.5, step=0.1,
                        help="Median income of the block group")
    house_age = st.slider("🏠 House Age (years)",
                          min_value=1, max_value=52,
                          value=25,
                          help="Median house age in block group")
    ave_rooms = st.slider("🚪 Average Rooms",
                          min_value=1.0, max_value=20.0,
                          value=5.0, step=0.5,
                          help="Average number of rooms per household")
    ave_bedrms = st.slider("🛏️ Average Bedrooms",
                           min_value=0.5, max_value=5.0,
                           value=1.1, step=0.1,
                           help="Average bedrooms per household")

with col2:
    population = st.slider("👥 Population",
                           min_value=10, max_value=10000,
                           value=1500, step=50,
                           help="Block group population")
    ave_occup = st.slider("🏘️ Average Occupants",
                          min_value=1.0, max_value=10.0,
                          value=3.0, step=0.1,
                          help="Average household members")
    latitude = st.slider("📍 Latitude",
                         min_value=32.0, max_value=42.0,
                         value=34.0, step=0.1,
                         help="Geographic latitude")
    longitude = st.slider("📍 Longitude",
                          min_value=-124.0, max_value=-114.0,
                          value=-118.0, step=0.1,
                          help="Geographic longitude")

st.markdown("---")

if st.button("🔮 Predict House Price"):
    input_data = np.array([[med_inc, house_age, ave_rooms, ave_bedrms,
                            population, ave_occup, latitude, longitude]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prediction = max(0.0, prediction)

    st.markdown(f"""
    <div class="result-box">
        <p style='color:#666;margin:0;font-size:1rem'>Predicted Median House Value</p>
        <p class="price-display">${prediction * 100000:,.0f}</p>
        <p style='color:#888;font-size:0.9rem'>({prediction:.4f} in $100,000 units)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Income Level", f"${med_inc * 10000:,.0f}/yr")
    col_b.metric("House Age", f"{house_age} years")
    col_c.metric("Location", f"{latitude:.1f}°N, {longitude:.1f}°E")

st.markdown("---")
st.caption("Built by **Rose Sharma** | CSE AI Graduate | Rungta College, Bhilai")
