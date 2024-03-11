import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, Birch, DBSCAN
import seaborn as sns
from matplotlib import colors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

pd.options.display.max_columns = None
pd.options.display.width = 1000

dataset = pd.read_csv('../marketing_campaign.csv')

print("shape\n", dataset.shape)
print("head 10\n", dataset.head(10))
print("info\n", dataset.info())

print("\n\nEmpezamos con limpieza de datos\nEliminación de valores nulos")
dataset.dropna(inplace=True)
dataset.reset_index(drop=True, inplace=True)
print("Verificamos que se hayan eliminado los nulos\n", dataset.isnull().any())

print(
    "\n\nCambiamos agregamos nuevos atributos para solventar los que no son númericos o nos funcioonan mejor en otro formato")
print("Creamos el atributo Custoemr_For que indica lso días que ha sido cliente desde el último cliente registrado")
dataset['Dt_Customer'] = pd.to_datetime(dataset['Dt_Customer'], format='%d-%m-%Y')
dates = []
for i in dataset['Dt_Customer']:
    i = i.date()
    dates.append(i)
days = []
d1 = max(dates)
for i in dates:
    dif = d1 - i
    days.append(dif)
dataset['Customer_For'] = days
dataset['Customer_For'] = pd.to_numeric(dataset['Customer_For'], errors='coerce')

print("Creamos atrinuto Age")
dataset['Age'] = 2024 - dataset['Year_Birth']

print("Creamos atributo spend para detonar cuanto ha gastado en los últiomos 2 años")
dataset['Spent'] = (dataset['MntWines'] + dataset['MntSweetProducts'] + dataset['MntFruits']
                    + dataset['MntMeatProducts'] + dataset['MntFishProducts'] + dataset['MntGoldProds'])

print("Creamos atributo Living_with para reducir el marital status")
dataset['Living_with'] = dataset['Marital_Status'].replace({'Married': 'Partner', 'Together': 'Partner'
                                                               , 'Absurd': 'Alone', 'Widow': 'Alone', 'YOLO': 'Alone',
                                                            'Divorced': 'Alone', 'Single': 'Alone'})

print("Creamoos atributo Children para reducir los atributos de kids y teen en uno")
dataset['Children'] = dataset['Kidhome'] + dataset['Teenhome']

print("Creamos atributo Family_Size que contiene los datos de los antribtos anteriores")
print(dataset['Children'])
print(dataset['Living_with'].replace({'Alone': 1, 'Partner': 2}))
dataset['Family_Size'] = dataset['Living_with'].replace({'Alone': 1, 'Partner': 2}) + dataset['Children']

print("Creamos atributo Is_Parent para indicarnos si es padre o no")
dataset['Is_Parent'] = np.where(dataset['Children'] > 0, 1, 0)

print('Modificamos atributo Education para reducir las categorías que ay tiene a 3')
dataset['Education'] = dataset['Education'].replace({'Basic': 'Undergraduate', '2n Cycle': 'Undergraduate'
                                                    , 'Graduation': 'Graduate', 'Master': 'Postgraduate',
                                                     'PhD': 'Postgraduate'})

print('Eliminamos atributo redundantes o inutilos')
dataset = dataset.drop(['Marital_Status', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue', 'Year_Birth', 'ID']
                       , axis=1)

print(dataset.describe())


print("Se limpian los datos ouliers")
dataset = dataset[dataset['Age'] < 90]
dataset = dataset[dataset['Income'] < 600000]
print(dataset.describe())

#To plot some selected features
#Setting up colors prefrences
sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"w"})
#Plotting following features
To_Plot = [ "Income", "Recency", "Customer_For", "NumCatalogPurchases", "Spent", "Is_Parent"]
print("Reletive Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(dataset[To_Plot], hue= "Is_Parent",palette= (["g", "r"]))
#Taking hue
plt.show()

dataset = dataset[(dataset["Age"]<90)]
dataset = dataset[(dataset["Income"]<600000)]
dataset = dataset.reset_index(drop=True)

print("\n\nProcesamiento de datos")
print("Se codifican als etiquetas de os atributos categoricos:")
s = (dataset.dtypes == 'object')
object_cols = list(s[s].index)
print(object_cols)
LE=LabelEncoder()
for i in object_cols:
    dataset[i]=dataset[[i]].apply(LE.fit_transform)

print("Se realizará la escala de los datos no booleanos")
data = dataset.copy()
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
data = data.drop(cols_del, axis=1)
#scalacion
scaler = StandardScaler()
scaler.fit(data)
data_scalada = pd.DataFrame(scaler.transform(data), columns=data.columns)
print(data_scalada.head())

print("Reducción de dimencionalidad con PCA")
pca = PCA(n_components=3)
pca.fit(data_scalada)
PCA_df = pd.DataFrame(pca.transform(data_scalada), columns=["Componente_1", "Componente_2", "Componente_3"])
print(PCA_df.describe().T)

x = PCA_df['Componente_1']
y = PCA_df['Componente_2']
z = PCA_df['Componente_3']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x,y,z, c='r', marker='o')
ax.set_title("Proyeción de los datos en dimenciones reducidas")
plt.show()

print("\n\nEmpezamos algoritmo del clustering con K-means")
X = PCA_df.iloc[:, :].values

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
kmeansmodel = KMeans(n_clusters=4, init='k-means++', random_state=0)
y_kmeans = kmeansmodel.fit_predict(X)
Ppca_kmeans = PCA_df.copy()
Ppca_kmeans['ClustersPCA'] = y_kmeans
#dataset['ClustersPCA'] = y_kmeans


fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=Ppca_kmeans["ClustersPCA"], marker='o', cmap=colors.ListedColormap(['r','g','b','c']) )
ax.set_title("Clustering K-beans")
plt.show()

# Visualizing all the clusters
_ = plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=5, c='red', label='Cluster 1')
_ = plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=5, c='green', label='Cluster 2')
_ = plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=5, c='blue', label='Cluster 3')
_ = plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=5, c='cyan', label='Cluster 4')
# _ = plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
_ = plt.scatter(kmeansmodel.cluster_centers_[:, 0], kmeansmodel.cluster_centers_[:, 1], s=15, c='black',
                label='Centroides')
_ = plt.title('Clusters de clientes')
_ = plt.xlabel('Año de nacimiento')
_ = plt.ylabel('Gato en vino en los últimos 2 años ($)')
_ = plt.legend()
plt.show()


print("Calculamos clusters con algoritmo de Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH) ")
# Model
brc = Birch(n_clusters=None, threshold=0.2)
brc.fit(PCA_df)
clusters = brc.predict(PCA_df)

#plt.scatter(PCA_df['Componente_1'], PCA_df['Componente_2'], c=clusters, cmap='rainbow', alpha=0.7, edgecolors='b')
#plt.show()

# 3D

figura = plt.figure(1, figsize=(10, 8))
# ax = Axes3D(figura, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax = figura.add_subplot(111, projection='3d')
ax.scatter(PCA_df['Componente_1'], PCA_df['Componente_2'], PCA_df['Componente_3'], c=clusters, edgecolor="k")
ax.set_xlabel("Componente_1")
ax.set_ylabel("Componente_2")
ax.set_zlabel("Componente_3")
ax.set_title('Clustering BIRCH')
ax.dist = 12
figura.show()



print("Empieaz el clustering por DBSCAN")
DBS_clustering = DBSCAN(eps=0.8, min_samples=4).fit(PCA_df)
DBSCAN_clustered = PCA_df.copy()
DBSCAN_clustered['Cluster'] = DBS_clustering.labels_
print(DBSCAN_clustered['Cluster'].sum)
DBS_clust_size = DBSCAN_clustered.groupby(['Cluster']).size().to_frame()
DBS_clust_size.columns = ['Cluster_size']
print(DBS_clust_size)

atipicos = DBSCAN_clustered[DBSCAN_clustered['Cluster'] == -1]
normales = DBSCAN_clustered[DBSCAN_clustered['Cluster'] != -1]

figu = plt.figure(1, figsize=(10,8))
ax = figu.add_subplot(111, projection='3d')
ax.scatter(normales['Componente_1'], normales['Componente_2'], normales['Componente_3'], c=normales['Cluster'])
ax.scatter(atipicos['Componente_1'], atipicos['Componente_2'], atipicos['Componente_3'], c='r')
ax.set_xlabel('Componente_1')
ax.set_ylabel('Componente_2')
ax.set_zlabel('Componente_3')
ax.dist = 12
figu.show()

print("Se evalúan los resultados del clustering con silhouette_score")
score_kmeans = silhouette_score(PCA_df, y_kmeans)
score_birch = silhouette_score(PCA_df, clusters)
score_BSCAN = silhouette_score(PCA_df, DBS_clustering.labels_)

print("K-means score:", score_kmeans,'\nBIRCH score:', score_birch,'\nBSCAN score:', score_BSCAN)