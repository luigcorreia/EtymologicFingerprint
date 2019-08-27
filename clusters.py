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

def calculate_wcss(data):
    wcss = []
    for n in range(2, 15):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)

    return wcss

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

if __name__ == '__main__':

    print("Carregando dataset de fingerprints")
    fingerprints = pd.read_csv("brown_fingerprints.csv",index_col=0)
    y = get_brown_categories()

    print("Extraíndo as duas Componentes Principais")
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(fingerprints)
    pca_df = pd.DataFrame(principalComponents, columns=['PC 1', 'PC 2'], index=fingerprints.index)

    print("Obtendo agrupamentes de 2 à 15 clusters para PC")
    wcss_pca = calculate_wcss(pca_df)
    plt.plot(range(2,15),wcss_pca,label='Fingerprints PCA')
    
    print("Calculando o número ótimo de grupos para PC")
    optimal_n_pca = optimal_number_of_clusters(wcss)
    print(optimal_n_pca)

    print("Classficando documentos pelo cluster obtido")
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pca_df)
    y_cluster = kmeans.predict(pca_df)
    y_cluster = pd.DataFrame({'cluster':y_cluster},index=pca_df.index)

    # Reduzir categorias em número ótimo de grupos:
    y_categories = y.apply(lambda c: 0 if c in ['goverment','learned'] else 1 if c in ['romance','adventure','science_fiction','mystery','fiction'] else 2)

    # Calcular shilluette e Rand Ajusted Index
    print(ajusted_rand_score(y_categories, y_cluster))
    print(silhouette_score(pca_df, y_cluster))

    # Agruppar com cfuzzy
    # Calcular fuzzy shilluete 

    #print("Plotando resultados")
    #fingerprints_pca = pd.concat([pca_df,y,y_cluster], axis=1)
    #ComponentsPlot('PC 1','PC 2', 'category', fingerprints_pca)
    #ComponentsPlot('PC 1','PC 2', 'cluster', fingerprints_pca)
