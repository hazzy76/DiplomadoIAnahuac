import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataset = pd.read_csv('../marketing_campaign.csv')

print("shape\n", dataset.shape)
print("head 10\n", dataset.head(10))
print("info\n", dataset.info())
print("isnull Sunm\n", dataset.isnull().sum())

print("Empezamos con limpieza de datos\nEliminación de valores nulos")
dataset.dropna(inplace=True)
dataset.reset_index(drop=True, inplace=True)
print("Verificamos que se hayan eliminado los nulos\n", dataset.isnull().sum())

print("Reducción de dimencionalidad con algoritmo PCA")
###################################################################Aqui empieza el PCA

customer_X = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
print("Se crea nuevo data frame target para saber si toma promociones")
promos = dataset.iloc[:, [20, 21, 22, 23, 24]]
print("Este es el dataset de promos\n", promos.shape, '\n', promos.head())
customer_Y = pd.DataFrame(promos.sum(axis=1))
customer_Y.rename(columns={0: 'class'}, inplace=True)
print("\nImprimimos X y Y(Suma de las promos que ahn tomado lso customers)")
print(customer_X.shape, '\n', customer_X)
print(customer_Y.shape, 'n', customer_Y)

customer_promos = pd.concat([customer_X, customer_Y], axis=1)
print("\nDataframe ordenado, primero con los atributos y luego las clases")
print(customer_promos.shape, '\n', customer_promos)
print("Pasamos a enteros los datos")
customer_promos['class'] = customer_promos['class'].replace(0, 'No promo')
customer_promos['class'] = customer_promos['class'].replace(1, '1 promo')
customer_promos['class'] = customer_promos['class'].replace(2, '2 promos')
customer_promos['class'] = customer_promos['class'].replace(3, '3 promos')
customer_promos['class'] = customer_promos['class'].replace(4, '4 promos')
customer_promos['class'] = customer_promos['class'].replace(5, '5 promos')
customer_promos['Marital_Status'] = customer_promos['Marital_Status'].replace('Single', 0)
customer_promos['Marital_Status'] = customer_promos['Marital_Status'].replace('Absurd', 1)
customer_promos['Marital_Status'] = customer_promos['Marital_Status'].replace('Alone', 2,)
customer_promos['Marital_Status'] = customer_promos['Marital_Status'].replace('Divorced', 3)
customer_promos['Marital_Status'] = customer_promos['Marital_Status'].replace('Married', 4)
customer_promos['Marital_Status'] = customer_promos['Marital_Status'].replace('Together', 5)
customer_promos['Marital_Status'] = customer_promos['Marital_Status'].replace('Widow', 6)
customer_promos['Marital_Status'] = customer_promos['Marital_Status'].replace('YOLO', 7)
customer_promos['Education'] = customer_promos['Education'].replace('2n Cycle', 0)
customer_promos['Education'] = customer_promos['Education'].replace('Graduation', 1)
customer_promos['Education'] = customer_promos['Education'].replace('Master', 2)
customer_promos['Education'] = customer_promos['Education'].replace('PhD', 3)
customer_promos['Education'] = customer_promos['Education'].replace('Basic', 4)
customer_promos['Dt_Customer'] = pd.to_datetime(customer_promos['Dt_Customer'], format='%d-%m-%Y')
for i in customer_promos.iterrows():
    customer_promos.loc[i[0],'Dt_Customer'] = customer_promos.loc[i[0], 'Dt_Customer'].toordinal()
print(customer_promos.shape, '\n', customer_promos.head())
print("\nSe noamlizan los datos")
x = customer_promos.iloc[:, 0:19].values
x = StandardScaler().fit_transform(x)
print((np.mean(x), np.std(x)))
feat_cols = ['feature' + str(i) for i in range(x.shape[1])]
data_normalized = pd.DataFrame(x, columns=feat_cols)
print(data_normalized.shape,'\n', data_normalized.to_string(max_cols=20, max_rows=10))
print("Empieza algoritmo componentes principales")
# Empieza algoritmo
pca_promos = PCA(n_components=2)
componentes_principales = pca_promos.fit_transform(x)
componentes_principales_DF = pd.DataFrame(data=componentes_principales, columns=['Componente_1', 'Componente_2'])
print(componentes_principales_DF.shape,'\n',componentes_principales_DF)
componentes_principales_DF.plot()
plt.show()
# plt.scatter(x=componentes_principales_DF['Componente_1'], y =componentes_principales_DF['Componente_2'])
# plt.show()

plt.figure()
plt.figure(figsize=(10, 10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Componente Principal - 1', fontsize=20)
plt.ylabel('Componente Principal - 2', fontsize=20)
plt.title("PCA del conjunto de datos de cancer de mama", fontsize=20)
targets = ['No promo', '1 promo', '2 promos', '3 promos', '4 promos', '5 promos']
colors = ['black', 'r', 'g', 'b', 'orange', 'violet']
for target, color in zip(targets, colors):
    print(customer_promos.shape, '\n', customer_promos)
    indicesToKeep = customer_promos['class'] == target
    plt.scatter(componentes_principales_DF.loc[indicesToKeep, 'Componente_1']
                , componentes_principales_DF.loc[indicesToKeep, 'Componente_2'], c=color, s=50)

plt.legend(targets, prop={'size': 15})
plt.show()

###################################################################Aquí termina el PCA
print("Empezamos algoritmo del clustering")
X = componentes_principales_DF.iloc[:, :].values

# Within Clusters Summed Squares
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Visualizing the ELBOW method to get the optimal value of K
plt.plot(range(1, 11), wcss)
plt.title('Método del codo')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()

print("Aquí ya debió imprimir")

# Model Build
kmeansmodel = KMeans(n_clusters=3, init='k-means++', random_state=0)
y_kmeans = kmeansmodel.fit_predict(X)

# Visualizing all the clusters
_ = plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=5, c='red', label='Cluster 1')
_ = plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=5, c='green', label='Cluster 2')
_ = plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=5, c='blue', label='Cluster 3')
# _ = plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
# _ = plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
_ = plt.scatter(kmeansmodel.cluster_centers_[:, 0], kmeansmodel.cluster_centers_[:, 1], s=15, c='black',
                label='Centroides')
_ = plt.title('Clusters de clientes')
_ = plt.xlabel('Año de nacimiento')
_ = plt.ylabel('Gato en vino en los últimos 2 años ($)')
_ = plt.legend()
_ = plt.show()
