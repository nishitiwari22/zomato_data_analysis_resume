import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Zomato Analytics", layout="wide")
st.title("🍽️ Zomato Consumer Trends & Rating Prediction")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    file_path = os.path.join(os.getcwd(), "data", "zomato.csv")

    if not os.path.exists(file_path):
        st.error(f"❌ File not found at {file_path}")
        st.stop()

    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Clean rating column
    if 'rate' in df.columns:
        df['rate'] = df['rate'].astype(str).str.replace('/5', '', regex=False)
        df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

    return df

df = load_data()

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Dashboard", "Explore Data", "Predict Rating"])

# -------------------------------
# DASHBOARD
# -------------------------------
if option == "Dashboard":
    st.header("📊 Insights Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Cuisines")
        if 'cuisines' in df.columns:
            top_cuisines = df['cuisines'].value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=top_cuisines.values, y=top_cuisines.index, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Column 'cuisines' not found.")

    with col2:
        st.subheader("Rating Distribution")
        if 'rate' in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df['rate'].dropna(), bins=20, kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Column 'rate' not found.")

    st.subheader("Cost vs Rating")
    if 'approx_cost(for two people)' in df.columns and 'rate' in df.columns:
        fig, ax = plt.subplots()
        sns.scatterplot(
            x='approx_cost(for two people)',
            y='rate',
            data=df,
            ax=ax
        )
        st.pyplot(fig)
    else:
        st.warning("Required columns not found.")

# -------------------------------
# EXPLORE DATA
# -------------------------------
elif option == "Explore Data":
    st.header("🔍 Explore Dataset")

    st.write("Shape of dataset:", df.shape)

    if 'cuisines' in df.columns:
        cuisine = st.selectbox("Select Cuisine", df['cuisines'].dropna().unique())
        filtered_df = df[df['cuisines'] == cuisine]
        st.write("Filtered Data", filtered_df.head())
    else:
        st.warning("Column 'cuisines' not found.")

# -------------------------------
# PREDICTION SECTION
# -------------------------------
elif option == "Predict Rating":
    st.header("🤖 Predict Restaurant Rating")

    @st.cache_resource
    def load_model():
        model_path = os.path.join(os.getcwd(), "model.pkl")

        if not os.path.exists(model_path):
            return None

        return pickle.load(open(model_path, "rb"))

    model = load_model()

    if model is None:
        st.warning("⚠️ Model not found. Please upload model.pkl to enable predictions.")
    else:
        # Inputs
        cost = st.number_input("Approx Cost for Two People", min_value=100, max_value=5000)
        votes = st.number_input("Number of Votes", min_value=0, max_value=10000)

        online_order = st.selectbox("Online Order Available?", ["Yes", "No"])
        book_table = st.selectbox("Table Booking Available?", ["Yes", "No"])

        # Convert inputs
        online_order = 1 if online_order == "Yes" else 0
        book_table = 1 if book_table == "Yes" else 0

        if st.button("Predict Rating"):
            features = np.array([[cost, votes, online_order, book_table]])
            prediction = model.predict(features)

            st.success(f"⭐ Predicted Rating: {round(prediction[0], 2)}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Made with ❤️ by Nishi")
