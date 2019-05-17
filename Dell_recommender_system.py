# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 08:52:20 2018

@author: caputr
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from IPython.display import display # Allows the use of display() for DataFrames
# Import supplementary visualizations code visuals.py
import visuals as vs


# Input all possible products on Dell.com
products = ['Latitude 3000', 'Latitude 5000', 'Latitude 7000', 'Latitude Education Series', 'Latitude Rugged Series', 'Vostro 3000', 'Vostro 5000',\
            'Inspiron 3000', 'Inspriron 5000', 'Inspriron 7000', 'Inspiron Chromebook 11', 'Precision Mobile Workstation 3000',\
            'Precision Mobile Workstation 5000', 'Precision Mobile Workstation 7000', 'XPS 13', 'XPS 13 2-in-1', 'XPS 15', 'XPS 15 2-in-1',\
            'Education Chromebook 11', 'Education Chromebook 13', 'G3', 'G5', 'G7', 'OptiPlex 3000', 'OptiPlex 5000', 'OptiPlex 7000',\
            'OptiPlex 3000 All-in-one', 'OptiPlex 5000 All-in-one', 'OptiPlex 7000 All-in-one', 'OptiPlex XE', 'Vostro Desktop',\
            'Inspiron Desktop', 'Inspirion 3000 All-in-one', 'Inspirion 5000 All-in-one', 'Inspirion 7000 All-in-one', 'Precision Fixed 3000',\
            'Precision Fixed 5000', 'Precision Fixed 5000 All-in-one', 'Precision Fixed 7000', 'XPS Tower', 'XPS 27 All-in-one', 'Wyse 3030 Thin Client',\
            'Wyse 3040 Thin Client', 'Wyse 5000 Thin Client', 'Wyse 5070 Thin Client', 'Wyse 7000 Thin Client', 'Wyse 7040 Thin Client', 'Wyse 5030 Zero Client',\
            'Wyse 7030 Zero Client', 'PowerEdge T', 'PowerEdge R', 'PowerEdge C', 'PowerEdge M', 'PowerEdge FX', 'PowerEdge MX', 'PowerEdge VRTX', 'KVM', 'KMM', 'UPS',\
            'Networking X', 'Networking N', 'Networking C', 'Networking S', 'Networking Z', 'Networking M Blade', 'Brocade', 'SonicWall', 'SonicPoint',\
            'UltraSharp 24', 'UltraSharp 27', 'UltraSharp 38 Curved', 'UltraSharp 34 Curved', 'UltraSharp 32 8K', 'UltraSharp 27 4K',\
            '27 P', '22 P', '24 P', '23 P', '24 4K P', '24 Touch P', '24 Video-Conferencing P', '24 E', '17 E', '22 E', '23 E', '70 C', '55 C', '55 C Interactive',\
            '86 Interactive Touch', 'Inkjet Printer', 'Color Laser Printer', 'Black/White Laser Printer', 'Scanner', 'Keys-To-Go', 'Dell Wireless Keyboard & Mouse',\
            'Visiontek Universal Dock', 'Dell Docking Station', 'Dell Business Dock', 'Dell Mobile USB-C Adapter', 'Briefcase', 'Backpack', 'Notebook Sleeve',\
            'Alienware 13', 'Alienware 15', 'Alienware 17', 'Inspiron Gaming Desktop', 'Alienware Aurora', 'Alienware Area-51', 'Alienware 34 Curved Gaming Monitor', 'Dell 27 Gaming Monitor',\
            'Alienware 25 Gaming Monitor', 'HTC Hive', 'Alienware headset', 'Gaming Keyboard', 'Alienware Mouse']

# Segment into hypothesized customer types
segments = {'Enthusiast': ['Latitude 3000', 'Latitude 5000', 'Latitude 7000', 'Latitude Education Series', 'Latitude Rugged Series', 'Vostro 3000', 'Vostro 5000',\
            'Inspiron 3000', 'Inspriron 5000', 'Inspriron 7000', 'Inspiron Chromebook 11', 'Precision Mobile Workstation 3000', \
            'Precision Mobile Workstation 5000', 'Precision Mobile Workstation 7000', 'XPS 13', 'XPS 13 2-in-1', 'XPS 15', 'XPS 15 2-in-1',\
            'Education Chromebook 11', 'Education Chromebook 13', '27 P', '22 P', '24 P', '23 P', '24 4K P', '24 Touch P', 'Dell Mobile USB-C Adapter', 'Briefcase', 'Backpack', 'Notebook Sleeve', 'Keys-To-Go'],\

            'Business': ['OptiPlex 3000', 'OptiPlex 5000', 'OptiPlex 7000',\
            'OptiPlex 3000 All-in-one', 'OptiPlex 5000 All-in-one', 'OptiPlex 7000 All-in-one', 'OptiPlex XE', 'Vostro Desktop',\
            'Inspiron Desktop', 'Inspirion 3000 All-in-one', 'Inspirion 5000 All-in-one', 'Inspirion 7000 All-in-one', 'Precision Fixed 3000',\
            'Precision Fixed 5000', 'Precision Fixed 5000 All-in-one', 'Precision Fixed 7000', 'XPS Tower', 'XPS 27 All-in-one', 'Wyse 3030 Thin Client',\
            'Wyse 3040 Thin Client', 'Wyse 5000 Thin Client', 'Wyse 5070 Thin Client', 'Wyse 7000 Thin Client', 'Wyse 7040 Thin Client', 'Wyse 5030 Zero Client',\
            'Wyse 7030 Zero Client', 'PowerEdge T', 'PowerEdge R', 'PowerEdge C', 'PowerEdge M', 'PowerEdge FX', 'PowerEdge MX', 'PowerEdge VRTX', 'KVM', 'KMM', 'UPS',\
            'Networking X', 'Networking N', 'Networking C', 'Networking S', 'Networking Z', 'Networking M Blade', 'Brocade', 'SonicWall', 'SonicPoint', 'Dell Business Dock', 'Dell Wireless Keyboard & Mouse'],\
            
            'Home_Office': ['UltraSharp 24', 'UltraSharp 27', 'UltraSharp 38 Curved', 'UltraSharp 34 Curved', 'UltraSharp 32 8K', 'UltraSharp 27 4K',\
            '27 P', '22 P', '24 P', '23 P', '24 4K P', '24 Touch P', '24 Video-Conferencing P', '24 E', '17 E', '22 E', '23 E', '70 C', '55 C', '55 C Interactive',\
            '86 Interactive Touch', 'Inkjet Printer', 'Dell Wireless Keyboard & Mouse', 'Vostro Desktop',\
            'Inspiron Desktop', 'Inspirion 3000 All-in-one', 'Inspirion 5000 All-in-one', 'Inspirion 7000 All-in-one', 'Precision Fixed 3000',\
            'Precision Fixed 5000', 'Precision Fixed 5000 All-in-one', 'Precision Fixed 7000', 'XPS Tower', 'XPS 27 All-in-one', 'Dell Docking Station', 'Dell Business Dock'],\
                            
            'Gaming': ['G3', 'G5', 'G7','Visiontek Universal Dock', 'Dell Docking Station', 'Dell Business Dock', 'Dell Mobile USB-C Adapter', 'Briefcase', 'Backpack', 'Notebook Sleeve',\
            'Alienware 13', 'Alienware 15', 'Alienware 17', 'Inspiron Gaming Desktop', 'Alienware Aurora', 'Alienware Area-51', 'Alienware 34 Curved Gaming Monitor', 'Dell 27 Gaming Monitor',\
            'Alienware 25 Gaming Monitor', 'HTC Hive', 'Alienware headset', 'Gaming Keyboard', 'Alienware Mouse','UltraSharp 24', 'UltraSharp 27', 'UltraSharp 38 Curved', 'UltraSharp 34 Curved', 'UltraSharp 32 8K', 'UltraSharp 27 4K']}

lengths = []        
for key,value in segments.items():
    lengths.append(len(value))

# Build random dataframes based on each customer type
def build_df():
    enthusiast =  np.random.choice([0, 1], size=(5000,lengths[0]), p=[8./10, 2./10])
    enthusiast_df = pd.DataFrame(enthusiast, columns=segments['Enthusiast'])
    for item in list(set(products)-set(segments['Enthusiast'])):
        enthusiast_df[item]=0
        
    business =  np.random.choice([0, 1, 2, 3, 4], size=(1000,lengths[1]), p=[2./10, 2./10, 3./10, 2./10, 1./10])
    business_df = pd.DataFrame(business, columns=segments['Business'])
    for item in list(set(products)-set(segments['Business'])):
        business_df[item]=0
    
    home_office =  np.random.choice([0, 1, 2], size=(3000,lengths[2]), p=[8./10, 1.5/10, 0.5/10])
    home_office_df = pd.DataFrame(home_office, columns=segments['Home_Office'])
    for item in list(set(products)-set(segments['Home_Office'])):
        home_office_df[item]=0
    
    gaming =  np.random.choice([0, 1], size=(1500,lengths[3]), p=[9./10, 1./10])
    gaming_df = pd.DataFrame(gaming, columns=segments['Gaming'])
    for item in list(set(products)-set(segments['Gaming'])):
        gaming_df[item]=0
        
    big_df = pd.concat([enthusiast_df, business_df, home_office_df, gaming_df], sort=True)
    
    return big_df

df = build_df()

samples = df.iloc[[3,5500,8100,9999],:]
# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2).fit(df)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(df)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))

# Create a biplot
# vs.biplot(df, reduced_data, pca)

# Apply clustering algorithm of choice to the reduced data 
clusterer = GaussianMixture(n_components=3).fit(reduced_data)

# TODO: Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

# TODO: Find the cluster centers
centers = clusterer.means_

# TODO: Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(reduced_data, preds)

vs.cluster_results(reduced_data, preds, centers, pca_samples)