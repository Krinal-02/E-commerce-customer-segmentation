# E-commerce-customer-segmentation-
## Project Overview
This project aims to segment customers of an e-commerce platform using historical transactional data. Customer segmentation is vital for businesses to optimize their marketing strategies, improve customer retention, and increase revenue.
## Table of Contents
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Data Description](#data-description)
- [RFM Analysis](#rfm-analysis)
- [Clustering Algorithms and Evaluation](#clustering-algorithms-and-evaluation)
- [Results](#results)
- [Installation & Usage](#installation--usage)

## Technologies Used
| Technology    | Description                               |
|---------------|-------------------------------------------|
| Python        | Core programming language.                |
| Pandas        | Data manipulation and analysis.           |
| Matplotlib & Seaborn | Data visualization.              |
| Scikit-learn  | Clustering algorithms.                     |
| Streamlit     | Web app framework to deploy the project. |

## Project Structure
The project files and their purposes are as follows:

| File/Directory           | Purpose                                                         |
|--------------------------|-----------------------------------------------------------------|
| `data`                   | Contains the analysis and model training dataset.      |
| `CustomerSegmentation`   | Jupyter notebooks with Exploratory Data Analysis (EDA), feature engineering, and model implementation. |
| `EDA_Plots`              | Visualizations and graphs.                                      |
| `README.md`              | Project documentation and instructions.                        |
| `requirements.txt`       | List of required libraries and dependencies to run the project.|
| `main.py`                | Contains the saved clustering models (K-Means, Hierarchical Clustering, DBSCAN) for deployment. |
| `app.py`                 | Streamlit application for customer segmentation.               |

## Data Description
The dataset contains e-commerce transactions with the following columns:

| Column Name     | Description                                            |
|------------------|-------------------------------------------------------|
| `InvoiceNo`      | Unique identifier for each transaction.               |
| `StockCode`      | Product code.                                         |
| `Description`    | Product description.                                  |
| `Quantity`       | Number of items purchased.                            |
| `InvoiceDate`    | Date of transaction.                                  |
| `UnitPrice`      | Price per item.                                      |
| `CustomerID`     | Unique identifier for each customer.                  |
| `Country`        | Country where the customer is located.               |

## RFM Analysis
- **Recency:** How recently a customer made a purchase.
- **Frequency:** How often a customer makes purchases.
- **Monetary:** How much money the customer has spent.

RFM analysis was applied to derive customer insights. A log transformation was used to address skewed distributions.

## Clustering Algorithms and Evaluation
| Algorithm                     | Parameters                | Silhouette Score |
|-------------------------------|---------------------------|-------------------|
| K-Means Clustering            | K=2                       | 0.4344            |
| K-Means Clustering            | K=3                       | 0.3389            |
| Hierarchical Clustering       | K=2                       | 0.4287            |
| DBSCAN                        | eps=0.5                   | 0.2648            |

## Results
- The analysis revealed distinct customer segments based on purchasing behavior.
- K-Means and hierarchical clustering outperformed DBSCAN.
- K=2 was the most effective number of clusters for K-Means and hierarchical clustering.

## Installation & Usage

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Krinal-02/E-commerce-customer-segmentation.git
2. Navigate to the project directory:
   ```bash
   cd E-commerce-customer-segmentation
3. Install the required packages:
   ```bash
   pip install -r requirements.txt

### Usage
- Run the analysis in a Jupyter Notebook.
- To deploy the project using Streamlit, run:
  ```bash
  streamlit run app.py
