import streamlit as st
import pandas as pd
import joblib
import json

from pathlib import Path
from database import save_prediction, get_history

# Project paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "mobile_price_model.pkl"
DATA_PATH = BASE_DIR / "mobile_price.csv"
METRICS_PATH = BASE_DIR / "metrics.json"

# Page configuration
st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
[data-testid="stMetric"] {
    background-color: #eaf2ff;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #cbdcf5;
}

[data-testid="stMetricLabel"] {
    color: #334155 !important;
}

[data-testid="stMetricValue"] {
    color: #1d4ed8 !important;
}

.stButton > button {
    border-radius: 8px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# Check required files
if not MODEL_PATH.exists():
    st.error("Trained model not found. Run train_model.py first.")
    st.stop()

if not DATA_PATH.exists():
    st.error("Dataset not found.")
    st.stop()

# Load model and dataset
model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

brands = sorted(data["brand"].dropna().unique().tolist())

# Prediction function
def predict_price(phone):

    input_data = pd.DataFrame(
        [phone],
        columns=[
            "brand", "ram", "storage",
            "battery", "camera", "screen"
        ]
    )

    result = model.predict(input_data)[0]

    return max(0, float(result))


# Input function
def phone_inputs(prefix):

    brand = st.selectbox(
        "Mobile Brand",
        brands,
        key=f"{prefix}_brand"
    )

    ram = st.selectbox(
        "RAM (GB)",
        [2, 4, 6, 8, 12, 16],
        index=3,
        key=f"{prefix}_ram"
    )

    storage = st.selectbox(
        "Storage (GB)",
        [32, 64, 128, 256, 512],
        index=2,
        key=f"{prefix}_storage"
    )

    battery = st.number_input(
        "Battery (mAh)",
        min_value=1000,
        max_value=10000,
        value=5000,
        key=f"{prefix}_battery"
    )

    camera = st.number_input(
        "Camera (MP)",
        min_value=2,
        max_value=300,
        value=50,
        key=f"{prefix}_camera"
    )

    screen = st.number_input(
        "Screen Size (inches)",
        min_value=4.0,
        max_value=8.0,
        value=6.5,
        step=0.1,
        key=f"{prefix}_screen"
    )

    return {
        "brand": brand,
        "ram": ram,
        "storage": storage,
        "battery": battery,
        "camera": camera,
        "screen": screen
    }


# Sidebar navigation
st.sidebar.title("📱 Mobile Predictor")

st.sidebar.write("Machine Learning Project")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Predict Price",
        "Compare Mobiles",
        "Recommendations",
        "Analytics",
        "Prediction History",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed using Python and Machine Learning")


# HOME PAGE
if page == "Home":

    st.title("📱 Mobile Phone Price Prediction")

    st.write(
        "Welcome to the intelligent mobile phone "
        "price prediction system."
    )

    st.info(
        "Predict estimated mobile prices using "
        "Machine Learning."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(data))

    col2.metric(
        "Mobile Brands",
        data["brand"].nunique()
    )

    col3.metric(
        "Average Price",
        f"₹{data['price'].mean():,.0f}"
    )

    st.markdown("---")

    st.subheader("Project Features")

    c1, c2 = st.columns(2)

    with c1:
        st.write("### 🎯 Price Prediction")
        st.write("Predict the price using mobile specifications.")

        st.write("### 📊 Data Analytics")
        st.write("Analyze mobile prices using graphs.")

        st.write("### 💰 Budget Recommendations")
        st.write("Find suitable sample configurations within a budget.")

    with c2:
        st.write("### 📱 Mobile Comparison")
        st.write("Compare the predicted prices of two phones.")

        st.write("### 🕒 Prediction History")
        st.write("View previously saved predictions.")

        st.write("### 🤖 Machine Learning")
        st.write("Uses Random Forest Regression.")


# PREDICT PRICE PAGE
elif page == "Predict Price":

    st.title("🎯 Predict Mobile Price")

    st.write("Enter the mobile phone specifications.")

    with st.form("prediction_form"):

        phone = phone_inputs("predict")

        submitted = st.form_submit_button(
            "Predict Mobile Price",
            use_container_width=True
        )

    if submitted:

        price = predict_price(phone)

        save_prediction(
            phone["brand"],
            phone["ram"],
            phone["storage"],
            phone["battery"],
            phone["camera"],
            phone["screen"],
            price
        )

        st.success("Price prediction completed!")

        st.metric(
            "Estimated Mobile Price",
            f"₹{price:,.0f}"
        )

        st.caption(
            "This is an estimated price generated "
            "from the available training dataset."
        )


# COMPARE MOBILES PAGE
elif page == "Compare Mobiles":

    st.title("📱 Compare Two Mobile Phones")

    st.write(
        "Enter specifications of two phones "
        "to compare their predicted prices."
    )

    with st.form("compare_form"):

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Mobile A")
            phone_a = phone_inputs("phone_a")

        with col2:
            st.subheader("Mobile B")
            phone_b = phone_inputs("phone_b")

        compare = st.form_submit_button(
            "Compare Mobiles",
            use_container_width=True
        )

    if compare:

        price_a = predict_price(phone_a)
        price_b = predict_price(phone_b)

        col1, col2 = st.columns(2)

        col1.metric(
            "Mobile A Estimated Price",
            f"₹{price_a:,.0f}"
        )

        col2.metric(
            "Mobile B Estimated Price",
            f"₹{price_b:,.0f}"
        )

        if price_a < price_b:
            st.success("Mobile A has the lower predicted price.")

        elif price_b < price_a:
            st.success("Mobile B has the lower predicted price.")

        else:
            st.info("Both phones have the same predicted price.")


# RECOMMENDATIONS PAGE
elif page == "Recommendations":

    st.title("💰 Budget-Based Mobile Recommendations")

    st.write(
        "Enter your budget to find suitable sample "
        "mobile configurations from the dataset."
    )

    budget = st.number_input(
        "Your Budget (₹)",
        min_value=1000,
        max_value=200000,
        value=25000,
        step=1000
    )

    if st.button("Find Recommendations"):

        recommendations = data[
            data["price"] <= budget
        ].sort_values("price", ascending=False)

        if not recommendations.empty:

            st.success(
                f"Found {len(recommendations)} "
                "sample configurations within your budget."
            )

            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No sample configuration found "
                "within this budget."
            )

    st.caption(
        "Recommendations are based on the supplied "
        "dataset, not live market listings."
    )


# ANALYTICS PAGE
elif page == "Analytics":

    st.title("📊 Mobile Price Analytics")

    st.subheader("Brand-wise Average Price")

    brand_prices = (
        data.groupby("brand")["price"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(brand_prices)

    st.subheader("RAM vs Mobile Price")

    st.scatter_chart(
        data,
        x="ram",
        y="price",
        color="brand"
    )

    st.subheader("Storage vs Mobile Price")

    st.scatter_chart(
        data,
        x="storage",
        y="price",
        color="brand"
    )

    st.subheader("Dataset Overview")

    st.dataframe(
        data.describe(),
        use_container_width=True
    )

    st.subheader("Model Performance")

    if METRICS_PATH.exists():

        with open(METRICS_PATH, "r") as file:
            metrics = json.load(file)

        c1, c2, c3 = st.columns(3)

        c1.metric("MAE", metrics["MAE"])
        c2.metric("RMSE", metrics["RMSE"])
        c3.metric("R² Score", metrics["R2 Score"])

        st.caption(
            "R² can be negative on a small test dataset. "
            "Metrics are not reliable estimates of "
            "real-world accuracy with only 24 records."
        )

    else:

        st.warning("Train the model to see evaluation metrics.")


# PREDICTION HISTORY PAGE
elif page == "Prediction History":

    st.title("🕒 Prediction History")

    history = get_history()

    if history:

        history_df = pd.DataFrame(
            history,
            columns=[
                "ID",
                "Brand",
                "RAM",
                "Storage",
                "Battery",
                "Camera",
                "Screen",
                "Predicted Price",
                "Date"
            ]
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No prediction history available yet.")


# ABOUT PROJECT PAGE
elif page == "About Project":

    st.title("ℹ️ About the Project")

    st.write("""
    ### Mobile Phone Price Prediction System

    This project uses Machine Learning to predict
    the estimated price of a mobile phone.

    ### Technologies Used

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Random Forest Regression
    - Streamlit
    - SQLite Database

    ### Input Features

    - Brand
    - RAM
    - Storage
    - Battery Capacity
    - Camera
    - Screen Size

    ### Output

    Estimated mobile phone price in Indian Rupees.

    ### Machine Learning Algorithm

    Random Forest Regression.
    """)

    st.success("Thank you for using Mobile Price Predictor!")