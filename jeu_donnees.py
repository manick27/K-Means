# Importer le package scikit-learn
from sklearn.datasets import load_iris

# Charger les données
iris = load_iris()

# Importer le package pandas
import pandas as pd

# Convertir les données en dataframe pandas
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Ajouter une colonne avec les noms des espèces
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Afficher les premières lignes du dataframe
# df.head()
print(df.head())

# ACP

# Importer le package numpy
import numpy as np

# Sélectionner les colonnes numériques
X = df.iloc[:, 0:4]

# Standardiser les données
X = (X - np.mean(X)) / np.std(X)

# Importer le package statsmodels
import statsmodels.api as sm

# Réaliser une ACP
pca = sm.PCA(X)

# Afficher le pourcentage de variance expliquée par chaque composante principale
print(pca.eigenvals / sum(pca.eigenvals))



# Méthode de la sihouette

# Importer le package sklearn.cluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Définir une liste de nombres de clusters à tester
k_range = range(2, 11)

# Initialiser une liste vide pour stocker les scores de silhouette
silhouette_scores = []

# Appliquer l'algorithme de K-means et calculer le score de silhouette pour chaque nombre de clusters
for k in k_range:
    # Appliquer l'algorithme de K-means avec k clusters
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)  # Définir explicitement n_init=10
    cluster_labels = kmeans.fit_predict(X)
    
    # Calculer le score de silhouette pour les clusters obtenus
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Trouver le nombre de clusters avec le score de silhouette le plus élevé
best_k = k_range[np.argmax(silhouette_scores)]

# Afficher les scores de silhouette pour chaque nombre de clusters
for k, score in zip(k_range, silhouette_scores):
    print(f"Nombre de clusters : {k}, Score de silhouette : {score}")

# Afficher le nombre optimal de clusters
print(f"Le nombre optimal de clusters selon la méthode de la silhouette est : {best_k}")



# Méthode de la densité (Density-based methods)

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Sélectionner les colonnes numériques
X = df.iloc[:, 0:4]

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Instancier et ajuster le modèle DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)

# Obtenir les étiquettes de cluster attribuées à chaque point
labels = dbscan.labels_

# Nombre de clusters (ignorer les points considérés comme du bruit, étiquetés -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Afficher le nombre de clusters
print("Nombre de clusters selon la méthode de dbscan: ", n_clusters)




# Algorithme du coude

# Importer le package matplotlib
import matplotlib.pyplot as plt
from scipy import stats

# Importer le sous-module scipy.cluster.vq
from scipy.cluster import vq

# Définir une liste de nombres de clusters à tester
k_range = range(1, 11)

# Initialiser une liste vide pour stocker les valeurs d'inertie totale
tot_withinss = []

# Calculer l'inertie totale pour chaque nombre de clusters
for k in k_range:
  # Appliquer l'algorithme de k-means avec k clusters
  km = vq.kmeans(X, k)
  # Ajouter la valeur de l'inertie totale à la liste
  tot_withinss.append(km[1])

# Tracer le graphique de l'inertie totale en fonction du nombre de clusters
plt.plot(k_range, tot_withinss, 'bx-')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie totale')
plt.title('Méthode du coude pour le jeu de données Iris')
plt.show()

# Algorithme de K-means

# Importer le package sklearn.cluster
from sklearn.cluster import KMeans

# Appliquer l'algorithme de k-means avec le nombre optimal de clusters
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

# Ajouter les étiquettes de cluster au dataframe
df['cluster'] = kmeans.labels_

# Obtenir les centres des clusters
cluster_centers = kmeans.cluster_centers_

# Afficher les premières lignes du dataframe avec les clusters
print(df.head())

# Importer le package seaborn
import seaborn as sns

# Visualiser les clusters avec la variable 'sepal length (cm)' en abscisse et 'sepal width (cm)' en ordonnée
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='cluster')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=100, label='Centres des clusters')
plt.title('Clusters des fleurs d\'iris')
plt.legend()
plt.show()

# Visualiser les clusters avec la variable 'petal length (cm)' en abscisse et 'petal width (cm)' en ordonnée
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='cluster')
plt.scatter(cluster_centers[:, 2], cluster_centers[:, 3], c='red', marker='X', s=100, label='Centres des clusters')
plt.title('Clusters des fleurs d\'iris')
plt.legend()
plt.show()


# Interpretation

# Sélectionner les colonnes numériques (à l'exclusion de 'species')
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_cols]

# Calculer les moyennes des caractéristiques pour chaque cluster
cluster_means = df_numeric.groupby('cluster').mean()

# Afficher les moyennes des caractéristiques pour chaque cluster
print(cluster_means)

# Entrainement du modèle pour prédictions

# Importer le modèle de régression logistique
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Diviser les données en variables prédictives et variable cible
X = df.iloc[:, 0:4]
y = df['cluster']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Créer une instance du modèle de régression logistique
model = LogisticRegression()

# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Prédire les clusters sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer les performances du modèle
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Exactitude : ", accuracy)
print("Matrice de confusion : ")
print(confusion_mat)



# Rapport de classification (Classification Report)


from sklearn.metrics import classification_report

# Calculer le rapport de classification
classification_report = classification_report(y_test, y_pred)
print("Rapport de classification : ")
print(classification_report)



# Exemple de prédiction

from sklearn.linear_model import LogisticRegression

# Créer une instance du modèle LogisticRegression
logistic_regression = LogisticRegression()

# Entraîner le modèle sur vos données d'entraînement
logistic_regression.fit(X_train, y_train)

# Caractéristiques de la fleur que vous souhaitez classifier
new_flower = [5.8, 2, 4, 1]

# Standardiser les caractéristiques de la nouvelle fleur
new_flower = (new_flower - np.mean(X)) / np.std(X)

# Prédire le cluster auquel la nouvelle fleur appartient
predicted_cluster = logistic_regression.predict([new_flower])

# Afficher le cluster prédit
print("Le cluster prédit pour la nouvelle fleur est :", predicted_cluster)
