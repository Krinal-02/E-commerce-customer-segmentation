from __future__ import print_function
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import datetime as dt
import joblib  # Import joblib to save the model

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import scipy.cluster.hierarchy as sch 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.cluster import DBSCAN

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv("data.csv", encoding='ISO-8859-1')

# Data preprocessing
data = data.dropna(subset=['CustomerID'])

# Remove rows with negative Quantity or UnitPrice values
data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]
# Drop duplicate values
data = data.drop_duplicates()
# Convert InvoiceDate to datetime format
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Calculate total revenue per transaction
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

# RFM analysis
# Set a reference date (e.g., max invoice date in the data)
reference_date = data['InvoiceDate'].max()

# Calculate Recency: Days since the customer's last purchase
rfm = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
})

# Rename columns for clarity
rfm.columns = ['Recency', 'Frequency']

# Calculate Monetary Value
rfm['Monetary'] = data.groupby('CustomerID').apply(lambda x: (x['Quantity'] * x['UnitPrice']).sum())

# Apply log transformation
rfm['Log_Recency'] = np.log1p(rfm['Recency'])    # log(1 + Recency) to avoid log(0)
rfm['Log_Frequency'] = np.log1p(rfm['Frequency'])
rfm['Log_Monetary'] = np.log1p(rfm['Monetary'])

# Select the log-transformed RFM columns
log_rfm = rfm[['Log_Recency', 'Log_Frequency', 'Log_Monetary']]

# Step 1: Standard Scaling
scaler = StandardScaler()
log_rfm_scaled = scaler.fit_transform(log_rfm)

# K-means clustering
k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(log_rfm_scaled)

# Assign clusters to the data
rfm['KMeans_Cluster(k=3)'] = kmeans.labels_

# Calculate customer count per cluster
cluster_profile = rfm.groupby('KMeans_Cluster(k=3)').agg({
    'Recency': ['mean', 'median', 'std'],
    'Frequency': ['mean', 'median', 'std'],
    'Monetary': ['mean', 'median', 'std'],
})

# Flatten multi-level columns
cluster_profile.columns = ['Recency_Mean', 'Recency_Median', 'Recency_Std', 
                           'Frequency_Mean', 'Frequency_Median', 'Frequency_Std', 
                           'Monetary_Mean', 'Monetary_Median', 'Monetary_Std']

# Add customer count per cluster
cluster_profile['Customer_Count'] = rfm['KMeans_Cluster(k=3)'].value_counts().sort_index()

print(cluster_profile)


# Save the KMeans model
joblib.dump(kmeans, 'kmeans_model.pkl')

# Save the scaler if needed (useful for scaling new data in the future)
joblib.dump(scaler, 'scaler.pkl')
