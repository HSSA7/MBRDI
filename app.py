import pandas as pd
from flask import Flask, jsonify, request
import os
import numpy as np
import torch
from flask_cors import CORS
from double_hat_solver import *
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt






app = Flask(__name__)
# CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})

global df
global df_cp
global data
df = pd.read_csv("./new_generated_filtered.csv")
df_cp=df
data=df_cp


global chosen_clusters 
chosen_clusters = []
global cluster_indices 
cluster_indices = {}
global random_elements 
random_elements = []

pt_orig=[]
ar_orig=[]
dp_orig=[]
wd_orig=[]
sz=8741
for i in range(sz):
    L, A, R, t = get_features_from_json('./JSON/N/neigh_dh_'+str(i)+'.json')
    P, C = solve_cross_section(L, A, R)
    arr=[]
    for x in P:
        arr.append(x)
    for x in C:
        arr.append(x)
    pt_orig.append(arr)
    area,_=get_cross_section_area_volume(L,A,R,t)
    ar_orig.append(area)
    x_max = max(P, key=lambda p: p[0])[0]
    x_min = min(P, key=lambda p: p[0])[0]
    y_max = max(P, key=lambda p: p[1])[1]
    y_min = min(P, key=lambda p: p[1])[1]
    wd_orig.append(y_max-y_min)
    dp_orig.append(x_max-x_min)


for i, row in df.iterrows():
    df.at[i, 'Area'] = ar_orig[i]
    df.at[i, 'Width'] = wd_orig[i]
    df.at[i, 'Depth'] = dp_orig[i]




@app.route('/preprocess', methods=['POST'])

def preprocess():

    req = request.get_json()
    
    
    deflection = req['deflection']
    width = req['Width']
    depth = req['Depth']
    length = req['L']

    header_range = {
    'Width': (0.9*width, 1.1*width),
    'Depth': (0.9*depth, 1.1*depth),
    'L': (0.9*length, 1.1*length),
    'deflection': (1.1*deflection, 0.9*deflection)
}

    print(header_range)
    
    global df_cp
    df_cp=df
    for header in header_range:
        df_cp = df_cp.loc[(df_cp[header] >= header_range[header][0]) & (df_cp[header] <= header_range[header][1])]
    
    x=len(df_cp)
    #sort according to absolute value of deflection
    df_cp['abs_deflection'] = df_cp['deflection'].abs()
    df_cp = df_cp.sort_values(by=['abs_deflection'],ascending=True)

    resarr=[]
    # // append only the corresponding indices in csv file
    for i in range(x):
        resarr.append(df_cp.index[i])

    values = {i: {'deflection': df_cp.loc[i, 'deflection'],
                             't': df_cp.loc[i, 't'],
                             'L': df_cp.loc[i, 'L'],
                             'Width': df_cp.loc[i, 'Width']} 
                          for i in resarr}


    resarr = [int(x) for x in resarr]
    values = {int(key): value for key, value in values.items()}


    print("RESARR:\n",resarr)
    print("VALUES:\n",values)

    df_cp=df_cp.drop(columns=['abs_deflection'])

    return jsonify({"len": x, "resarr": resarr, "values": values})

@app.route('/display', methods=['POST'])

def display():

    req = request.get_json()
    
    k = req['k']
    # convert k to int
    k = int(k)
    global data
    data=df_cp
    z=len(data)
    data.material = data.material.replace({'Usibor 2000':0, 'Usibor 1500':1 , 'CP 900':2 ,  'DP 1000':3})

    # Extract the feature columns from the data
    features = data.iloc[:, :-1].values

    import random

    # Generate a random integer between 10 and 100, inclusive
    # rand_num = random.randint(k, 4*k)


    # n_clusters = 0

    # if (z<rand_num):
    #     n_clusters=k
    # else:
    #     n_clusters=rand_num

    n_clusters = k
    

    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=n_clusters)

    # Fit the model to the data
    kmeans.fit(features)

    # Predict the cluster labels for each data point
    labels = kmeans.predict(features)

    # Add the cluster labels as a new column in the data
    data['Cluster'] = labels

    # Calculate the centroid of each cluster
    centroids = kmeans.cluster_centers_

    # Use hierarchical clustering to group the centroids into k clusters
    linkage_matrix = linkage(centroids, method='ward')
    clusters = fcluster(linkage_matrix, k, criterion='maxclust')

    # Calculate the distances between each cluster and the others
    cluster_distances = cdist(centroids, centroids, metric='euclidean')
    # print("Yaha tak hua\n")
    # Choose the k clusters that are most distant from one another
    global chosen_clusters
    chosen_clusters=[] 
    for i in range(k):
        # Find the cluster with the largest distance that is not already in the list of chosen clusters
        max_distance = -1
        max_cluster = None
        for j in range(len(cluster_distances)):
            if j not in chosen_clusters and cluster_distances[j].max() > max_distance:
                max_distance = cluster_distances[j].max()
                max_cluster =j
        chosen_clusters.append(max_cluster)

    global cluster_indices
    cluster_indices = {}
        # Loop through the rows of the CSV file
    for i, row in data.iterrows():
        # print(i)
            cluster = data['Cluster'][i]
            if cluster not in chosen_clusters:
                continue
            if cluster not in cluster_indices:
                # If the cluster is not already in the dictionary, add the index of a random element
                cluster_indices[cluster] = [i]
            else:
                # If the cluster is already in the dictionary, append the index to the list of indices
                cluster_indices[cluster].append(i)

    # Print the resulting dictionary
    # print(cluster_indices)
    global random_elements 
    random_elements = []
    # print(len(cluster_indices))
    # Loop through the keys of the dictionary
    # for cluster, indices in cluster_indices.items():
    for key in cluster_indices:
        # Choose a random index from the list of indices
        random_index = random.choice(cluster_indices[key])
        # print(random_index)
        # Append the corresponding element to the list of random elements
        
        # print(cluster, indices)
        # cluster_indices[i]=sort_by_order(cluster_indices[i],res)
        random_elements.append(random_index)

    # Print the resulting list of random elements
    # print(random_elements)

    # convert numpy.int64 objects to regular Python integers

    print(chosen_clusters)
    print(cluster_indices)
    print(random_elements)

    

    print("DATA BEFORE SHIZ: ",data.head(1000))

    values = {i: {'deflection': data.loc[i, 'deflection'],
                             't': data.loc[i, 't'],
                             'L': data.loc[i, 'L'],
                             'Width': data.loc[i, 'Width']} 
                          for i in random_elements}

    print("HERE ARE EXTRACTED VALUES: ", values)

    chosen_clusters = [int(x) for x in chosen_clusters]
    cluster_indices = {int(key): value for key, value in cluster_indices.items()}
    random_elements = [int(x) for x in random_elements]
    values = {int(key): value for key, value in values.items()}

    print("Current values are: ", values)

    return jsonify({"cluster": chosen_clusters, "cluster indices":cluster_indices,"random": random_elements, "values":values})

@app.route('/sort', methods=['POST'])

def sort():

    req = request.get_json()
    boo=req['sel']
    # clus=0
    # if(boo==1):
    #     clus=req['clus']
    p1=req['p1']
    p2=req['p2']
    # print(clus)
    print(p1)
    print(p2)

    global data
    print("DATA AFTER SHIZZ: ",data.head(1000))
    cluster_values = data['Cluster'].unique()
    print("Cluster values present in the selected dataframe:", cluster_values)
    df_cp2 = data
    if(boo==1):
        clus=req['clus']
        df_cp2=data.loc[data['Cluster'].astype(int) == int(clus)]
   
    print(df_cp2.head(10))
    df_cp2['abs_deflection'] = abs(df_cp2['deflection'])

    sort_by=[]

    if(p1==0):
        sort_by.append('abs_deflection')
       
    
    elif (p1==1):
        sort_by.append('Area')
       

    elif (p1==2):
        sort_by.append('t')
        
    
    elif (p1==3):
        sort_by.append('Depth')



    if(p2==0):
       sort_by.append('abs_deflection')
       
    
    elif (p2==1):
        sort_by.append('Area')
       

    elif (p2==2):
       sort_by.append('t')
        
    
    elif (p2==3):
        sort_by.append('Depth')
        


    
    

    df_cp2.sort_values(by=sort_by, ascending=[True]*len(sort_by), inplace=True)
    



    

    
    resar=[]
    for i, row in df_cp2.iterrows():
        resar.append(i)
    

    values = {i: {'deflection': df_cp2.loc[i, 'deflection'],
                             't': df_cp2.loc[i, 't'],
                             'L': df_cp2.loc[i, 'L'],
                             'Width': df_cp2.loc[i, 'Width']} 
                          for i in resar}

    print("HERE ARE EXTRACTED VALUES: ", values)

    values={int(key): value for key, value in values.items()}
    
    return jsonify({"resar": resar,"values":values})


@app.route('/')
def home():
    return "Welcome to Mercedez-Benz Working Unit! \n Please try a different route as this one is blocked."


if __name__=='__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', '3000'))