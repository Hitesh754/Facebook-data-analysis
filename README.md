# 📊 Facebook Live Sellers Data Analysis – K-Means Clustering

This project analyzes the **Facebook Live Sellers in Thailand dataset**, which includes engagement metrics from 10 Thai fashion and cosmetics retail Facebook pages. The objective is to apply **K-Means Clustering** to uncover patterns in user interactions and generate actionable insights for targeted marketing and customer segmentation.

---

## 📌 Objectives

- Perform **Exploratory Data Analysis (EDA)** on continuous social media metrics.
- Apply **K-Means Clustering** to identify groups with similar engagement patterns.
- Use the **Elbow Method** and **Silhouette Score** to determine the optimal number of clusters.
- Visualize results and interpret patterns across clusters.

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: 
  - `pandas` 
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  
---

## 📁 Dataset

- **Name**: Facebook Live Sellers in Thailand
- **Features**:
  - `likes`, `shares`, `comments`, `views`, etc.
  - All features are continuous and represent real-time engagement metrics.

> 📌 *Note: Add a source or link to the dataset here if publicly available or cite the academic source.*

---

## 🔍 Analysis Performed

1. **Data Cleaning and Preprocessing**
   - Handled missing values and normalized the dataset for clustering.

2. **Exploratory Data Analysis (EDA)**
   - Visualized relationships and distribution using pairplots, boxplots, and heatmaps.

3. **K-Means Clustering**
   - Determined optimal clusters using Elbow Method and Silhouette Score.
   - Segmented pages/users based on engagement behavior.

4. **Visualization of Clusters**
   - Plotted clusters in 2D space using PCA for easier interpretation.

---

## 📈 Key Insights

- Identified distinct clusters such as:
  - High-likes, low-comments pages
  - Balanced engagement across metrics
  - Low engagement overall
- Highlighted how different sellers focus on varying engagement strategies.
- Suggested that marketers tailor campaigns to cluster-specific behaviors.

---

## 📸 Sample Visuals

> (Include screenshots or saved plots here in your repo, e.g., elbow.png, clusters.png)

- `elbow_plot.png`: Finding optimal K
- `clusters.png`: Visualized clusters
- `heatmap.png`: Feature correlations


