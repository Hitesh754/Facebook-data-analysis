import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df=pd.read_csv(r'C:/Users/hites/Downloads/MLResearch/MLResearch/Facebook Dataset/Facebook_Marketplace_data.csv')
print(df)
print(df.info())
print(df.head())
df.drop(columns=['Column1'],inplace=True)
df.drop(columns=['Column2'],inplace=True)
df.drop(columns=['Column3'],inplace=True)
df.drop(columns=['Column4'],inplace=True)
z=df.describe()


#1.How does the time of upload (`status_published`)  affects the `num_reaction`?
df['status_published']=pd.to_datetime(df['status_published'])
df['hour'] = df['status_published'].dt.hour
plt.scatter(df['hour'],df['num_reactions'])
plt.show()
#What is the average value of num_reaction, num_comments, num_shares for each post type?



#2.Is there a correlation between the number of reactions (num_reactions) and other engagement metrics such as comments (num_comments) and shares (num_shares)? If so, what is the strength and direction of this correlation?
print(df.info())
correlation = df[['num_reactions', 'num_comments', 'num_shares']].corr()
print(correlation)
sns.heatmap(correlation, annot=True, cmap='PuBuGn')
plt.title('Correlation between reactions,comments and shares')
plt.show()
#3.Use the columns status_type, num_reactions, num_comments, num_shares, num_likes, num_loves, num_wows, num_hahas, num_sads, and num_angrys to train a K-Means clustering model on the Facebook Live Sellers dataset.
df.drop(columns=['status_id'], inplace=True)
df.drop(columns=['status_published'],inplace=True)
df.info()
x=df
y=df['status_type']   
le = LabelEncoder()
x['status_type'] = le.fit_transform(x['status_type'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
print(scaled_data)
#4.Use the elbow method to find the optimum number of clusters.
wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init= 'k-means++', random_state=42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=4, init= 'k-means++', random_state=42)
y_kmeans=kmeans.fit_predict(x)
x=x.values
plt.scatter(x[y_kmeans== 0,0],x[y_kmeans==0,1],s = 100,c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans== 1,0],x[y_kmeans==1,1],s = 100,c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans== 2,0],x[y_kmeans==2,1],s = 100,c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans== 3,0],x[y_kmeans==3,1],s = 100,c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_kmeans== 4,0],x[y_kmeans==4,1],s = 100,c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 300,c = 'yellow', label = 'Centroids')
plt.title('Reactions')
plt.xlabel('Categories')
plt.ylabel('number of reactions')
plt.legend()
plt.show()


#What is the count of different types of posts in the dataset?
#'link': 0, 'photo': 1, 'status': 2, 'video': 3
value_counts=df['status_type'].value_counts()
print(value_counts)
#What is the average value of num_reaction, num_comments, num_shares for each post type?
#'link': 0, 'photo': 1, 'status': 2, 'video': 3
grouped_data = df.groupby('status_type')[['num_reactions', 'num_comments', 'num_shares']].mean()
print(grouped_data)