import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

@st.cache_data
def load_data():
    file_path = os.path.join("data", "zomato.csv")
    return pd.read_csv(file_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    
    return pd.read_csv(file_path)

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
    pd.read_csv("data/zomato.csv")
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
        top_cuisines = df['cuisines'].value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=top_cuisines.values, y=top_cuisines.index, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['rate'], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Cost vs Rating")
    fig, ax = plt.subplots()
    sns.scatterplot(x='approx_cost(for two people)', y='rate', data=df, ax=ax)
    st.pyplot(fig)

# -------------------------------
# EXPLORE DATA
# -------------------------------
elif option == "Explore Data":
    st.header("🔍 Explore Dataset")

    st.write("Shape of dataset:", df.shape)

    # Filters
    cuisine = st.selectbox("Select Cuisine", df['cuisines'].dropna().unique())
    filtered_df = df[df['cuisines'] == cuisine]

    st.write("Filtered Data", filtered_df.head())

# -------------------------------
# PREDICTION SECTION
# -------------------------------
elif option == "Predict Rating":
    st.header("🤖 Predict Restaurant Rating")

    # Load model
    @st.cache_resource
    def load_model():
        return pickle.load(open("model.pkl", "rb"))

    model = load_model()

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
