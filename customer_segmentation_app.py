import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🧠 Customer Segmentation using Clustering")

# Upload dataset
uploaded_file = st.file_uploader("Upload your customer data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(df.head())

    # Preprocessing
    df_clean = df.copy()
    st.subheader("Step 1: Preprocessing")

    # Encode categorical columns
    non_numeric = df_clean.select_dtypes(include=['object']).columns
    df_clean = pd.get_dummies(df_clean, columns=non_numeric, drop_first=True)

    # Handle missing values
    if df_clean.isnull().values.any():
        st.warning("Missing values detected. Filling them with column mean.")
        df_clean.fillna(df_clean.mean(numeric_only=True), inplace=True)

    st.write("Processed Data:")
    st.write(df_clean.head())

    # Standardize data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clean)

    # Elbow Method
    st.subheader("Step 2: Optimal Clusters (Elbow Method)")
    inertia = []
    K = range(1, min(11, df_clean.shape[0] + 1))  # avoid K > number of samples
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K, inertia, 'bo-')
    ax.set_xlabel('Number of clusters (K)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method For Optimal K')
    st.pyplot(fig)

    # KMeans Clustering
    st.subheader("Step 3: Clustering")

    max_clusters = min(10, df_clean.shape[0])  # ensure clusters ≤ samples
    default_val = min(3, max_clusters)
    n_clusters = st.slider("Select number of clusters", 2, max_clusters, default_val)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)
    df['Cluster'] = clusters

    st.write("Clustered Data:")
    st.write(df.head())

    # PCA for 2D Visualization
    st.subheader("Step 4: PCA Visualization (2D)")
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_scaled)

    df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = clusters

    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='Set2', ax=ax2)
    ax2.set_title('Customer Segmentation (PCA 2D)')
    st.pyplot(fig2)
else:
    st.info("Please upload a CSV file to proceed.")
