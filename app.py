import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
st.title("Customer Segmentation Using K-Means Clustering")

# Add a file uploader in Streamlit to allow users to upload the CSV file
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    # Preprocessing
    data = data.dropna(subset=['CustomerID'])
    data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]
    data = data.drop_duplicates()
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

    # RFM analysis
    reference_date = data['InvoiceDate'].max()
    rfm = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
    })
    rfm.columns = ['Recency', 'Frequency']
    rfm['Monetary'] = data.groupby('CustomerID').apply(lambda x: (x['Quantity'] * x['UnitPrice']).sum())

    # Add Country and Description to RFM DataFrame for filtering
    rfm['Country'] = data.groupby('CustomerID')['Country'].first()
    rfm['Description'] = data.groupby('CustomerID')['Description'].first()

    # Calculate total revenue per country and description
    country_revenue = data.groupby('Country')['TotalPrice'].sum().reset_index()
    description_revenue = data.groupby('Description')['TotalPrice'].sum().reset_index()

    # Get top 15 countries and descriptions by revenue
    top_countries = country_revenue.sort_values(by='TotalPrice', ascending=False).head(15)
    top_descriptions = description_revenue.sort_values(by='TotalPrice', ascending=False).head(15)

    # Sidebar for filtering
    st.sidebar.header("Filter Options")
    country_filter = st.sidebar.selectbox("Select Country", options=['All'] + sorted(top_countries['Country'].unique().tolist()))
    description_filter = st.sidebar.selectbox("Select Description", options=['All'] + sorted(top_descriptions['Description'].unique().tolist()))

    # Apply filters
    if country_filter != 'All':
        rfm = rfm[rfm['Country'] == country_filter]
    if description_filter != 'All':
        rfm = rfm[rfm['Description'] == description_filter]

    # Apply log transformation
    rfm['Log_Recency'] = np.log1p(rfm['Recency'])
    rfm['Log_Frequency'] = np.log1p(rfm['Frequency'])
    rfm['Log_Monetary'] = np.log1p(rfm['Monetary'])
    log_rfm = rfm[['Log_Recency', 'Log_Frequency', 'Log_Monetary']]

    # Add a slider for selecting the number of clusters
    k = st.sidebar.slider("Select number of clusters", min_value=2, max_value=6, value=3)

    # Load the scaler and model
    scaler = joblib.load('scaler.pkl')  # Ensure that you have the scaler.pkl file saved from the training phase
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    log_rfm_scaled = scaler.transform(log_rfm)
    rfm['KMeans_Cluster'] = kmeans.fit_predict(log_rfm_scaled)

    # Cluster profile calculation
    cluster_profile = rfm.groupby('KMeans_Cluster').agg({
        'Recency': ['mean', 'median', 'std'],
        'Frequency': ['mean', 'median', 'std'],
        'Monetary': ['mean', 'median', 'std']
    })

    # Flatten column names
    cluster_profile.columns = ['Recency_Mean', 'Recency_Median', 'Recency_Std',
                               'Frequency_Mean', 'Frequency_Median', 'Frequency_Std',
                               'Monetary_Mean', 'Monetary_Median', 'Monetary_Std']

    # Add customer count per cluster
    cluster_profile['Customer_Count'] = rfm['KMeans_Cluster'].value_counts().sort_index()

    # Display cluster profile
    st.subheader("Cluster Profiling")
    st.dataframe(cluster_profile)

    # Add download button for the cluster profile
    csv = cluster_profile.to_csv().encode('utf-8')
    st.download_button(
        label="Download Cluster Profile as CSV",
        data=csv,
        file_name='cluster_profile.csv',
        mime='text/csv',
    )

    # Visualize cluster profiling
    st.subheader("Cluster Profiling Visualization")

    # Bar plot for Recency, Frequency, and Monetary values by cluster with customer count annotations
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Recency Plot
    sns.barplot(x=cluster_profile.index, y='Recency_Mean', data=cluster_profile, ax=ax[0], palette='Blues')
    ax[0].set_title('Mean Recency by Cluster')
    ax[0].set_ylabel('Mean Recency')
    for i, row in cluster_profile.iterrows():
        ax[0].text(i, row['Recency_Mean'], f'{int(row["Customer_Count"])} customers', ha='center')

    # Frequency Plot
    sns.barplot(x=cluster_profile.index, y='Frequency_Mean', data=cluster_profile, ax=ax[1], palette='Greens')
    ax[1].set_title('Mean Frequency by Cluster')
    ax[1].set_ylabel('Mean Frequency')
    for i, row in cluster_profile.iterrows():
        ax[1].text(i, row['Frequency_Mean'], f'{int(row["Customer_Count"])} customers', ha='center')

    # Monetary Plot
    sns.barplot(x=cluster_profile.index, y='Monetary_Mean', data=cluster_profile, ax=ax[2], palette='Reds')
    ax[2].set_title('Mean Monetary by Cluster')
    ax[2].set_ylabel('Mean Monetary')
    for i, row in cluster_profile.iterrows():
        ax[2].text(i, row['Monetary_Mean'], f'{int(row["Customer_Count"])} customers', ha='center')

    st.pyplot(fig)

else:
    st.warning("Please upload a CSV file to proceed.")
