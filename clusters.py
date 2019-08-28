#!/usr/bin/env python3

from datasets import *
from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import skfuzzy as fuzz

def ComponentsPlot(x,y,categories,df):
    sns.lmplot(x=x, y=y, data=df, hue=categories, fit_reg=False)
    plt.show()

def calculate_scores(data):
    wcss = []
    silhouette = []
    fpcs = [] 
    for n in range(2, 15):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(data)
        y = kmeans.predict(data)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, n, 2, error=0.005, maxiter=1000, init=None)

        wcss.append(kmeans.inertia_)
        silhouette.append(silhouette_score(data,y))
        fpcs.append(fpc)

    return wcss, silhouette, fpcs

def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = len(wcss), wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2

def distribuition(fingerprints_pca, n_groups):
    
    df = pd.DataFrame(index=fingerprints_pca['category'].drop_duplicates())

    for i in range(n_groups):
        df['Cluster '+str(i)] = fingerprints_pca[fingerprints_pca['cluster']==i].groupby('category').count()['cluster']
    df = df.fillna(0)
    for i in range(n_groups):
        df['Cluster '+str(i)] = df['Cluster '+str(i)].astype(int)

    return df


if __name__ == '__main__':

    print("Carregando dataset de fingerprints")
    fingerprints = pd.read_csv("brown_fingerprints.csv",index_col=0)
    y = get_brown_categories()

    print("Extraíndo as duas Componentes Principais")
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(fingerprints)
    pca_df = pd.DataFrame(principalComponents, columns=['PC 1', 'PC 2'], index=fingerprints.index)

    print("Obtendo agrupamentes de 2 à 15 clusters para PC")
    wcss_pca, silhouette_pca, fpcs = calculate_scores(pca_df)
    plt.plot(range(2,15),fpcs,label='Fingerprints PCA')
    plt.show()

    print("Classificando documentos pelo cluster obtido")
    
    kmeans_pca = KMeans(n_clusters=2)
    kmeans_pca.fit(pca_df)
    y_cluster = kmeans_pca.predict(pca_df)
    y_cluster = pd.DataFrame({'cluster':y_cluster},index=pca_df.index)

    print("Índice de Silhueta para 2 grupos:")
    print(silhouette_score(pca_df, y_cluster['cluster']))

    # Reduzir categorias para 2 grupos:
    print("Índice de Rand Ajustado para 2 grupos:")
    y['reduced'] = y['category'].apply(lambda c: 0 if c in ['fiction','mystery','science_fiction','adventure','romance','humor'] else 1)
    # Calcular Rand Ajusted Index
    print(adjusted_rand_score(y['reduced'], y_cluster['cluster']))
    # Para 3 grupos
    print("Índice de Rand Ajustado para 3 grupos:")
    y['reduced'] = y['category'].apply(lambda c: 2 if c in ['goverment','learned'] else 1 if c in ['humor', 'romance','adventure','science_fiction','mystery','fiction'] else 0)
    print(adjusted_rand_score(y['reduced'], y_cluster['cluster']))

    # Agrupar com cfuzzy
    #cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(pca_df, 2, 2, error=0.005, maxiter=1000, init=None)

    #print("Plotando resultados")
    fingerprints_pca = pd.concat([pca_df,y,y_cluster], axis=1)
    #fingerprints = pd.concat([fingerprints,y,y_cluster], axis=1)
    #distribuition(fingerprints_pca, 2).to_csv('category_distribution_2.csv')
    #ComponentsPlot('ang','lat', 'category', fingerprints)
    ComponentsPlot('PC 1','PC 2', 'cluster', fingerprints_pca)
