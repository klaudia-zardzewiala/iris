from sklearn.datasets import load_iris
iris_dataset=load_iris()

print('Klucze z dataset: \n{}'.format(iris_dataset.keys())) #dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
print(iris_dataset['DESCR'][:193]+'\n...') #Number of Instances: 150 (50 in each of three classes), Number of Attributes: 4 numeric, pre

print('Gatunki: {}'.format(iris_dataset['target_names'])) #Gatunki: ['setosa' 'versicolor' 'virginica']
print('Nazwa cech: \n{}'.format(iris_dataset['feature_names'])) #Nazwa cech: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print('Typ danych: {}'.format(type(iris_dataset['data']))) #Typ danych: <class 'numpy.ndarray'>
print('Ksztalt danych: {}'.format(iris_dataset['data'].shape)) #Ksztalt danych: (150, 4)
print('\n{}'.format(iris_dataset['data'][:5])) #Pierwsze 5 probek
print('Typ obiektu target: {}'.format(type(iris_dataset['target']))) #Typ obiektu target: <class 'numpy.ndarray'>
print('Kształt danych w targecie: {}'.format(iris_dataset['target'].shape)) #Kształt danych w targecie: (150,)
print('Target:\n{}'.format(iris_dataset['target'])) 
#Target: (0 gatunek setosa; 1 versicolor; 2 virginia)
#[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
# 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
# 2 2]

#Wywolanie funkcji train_test_split; tasujemy zestaw danych za pomocą generatora liczb pseudolosowych
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print('X_train shape: {}'.format(X_train.shape)) #X_train shape: (112, 4)
print('y_train shape: {}'.format(y_train.shape)) #y_train shape: (112,)
print('X_test shape: {}'.format(X_test.shape)) #X_test shape: (38, 4)
print('y_test shape: {}'.format(y_test.shape)) #y_test shape: (38,)

import pandas as pd 
import matplotlib.pyplot as plt

#Utworzenie ramki danych z danych X_trian, oznaczenie kolumny przy uzyciu ciagu znakow, utworzenie macierzy rozproszonej z ramki danych i pokolorowanie wg y_train
iris_dataframe=pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr=pd.plotting.scatter_matrix(iris_dataframe,c=y_train, figsize=(15,15), marker='o',hist_kwds={'bins':20}, s=60, alpha=0.8)
plt.show()

#Klasyfikator k-najbliyszch sąsiadów
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train) #fit() - zwarca obiekt knn i modyfikuje go w miejscu; pokazuje jakie parametry zostały uzyte podczas tworzenia modelu
print('Repreznetacja: {}'.format(knn.fit(X_train, y_train))) #KNeighborsClassifier(n_neighbors=1)

#Przewidywania
import numpy as np
X_new = np.array([[5, 2.9, 1, 0.2]])
print('X_new.shape: {}'.format(X_new)) #X_new.shape: (1, 4)

prediction = knn.predict(X_new)
print('Prognoza: {}'.format(prediction)) #Prognoza: [0]
print('Typ kwiatu: {}'.format(iris_dataset['target_names'][prediction])) #Typ kwiatu: ['setosa']
#model przewiduje, ze kwiat nalezy do klasy 0 - czyli gatunku setosa

#Ocenu modelu
#W danych testowych mozemy dokonać predykcji, a potem porównać jej wynik z etykietą (nazwą gatunku)
y_pred = knn.predict(X_test)
print('Prognozy danych z zestawu: \n {}'.format(y_pred))
#Prognozy danych z zestawu: 
#[2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0 2]
print('Wyniki dla zestawu danych: {:.2f}'.format(np.mean(y_pred==y_test))) #Wyniki dla zestawu danych: 0.97
print('Wynik dla zestawu danych: {:.2f}'.format(knn.score(X_test, y_test))) #metoda score obiektu kknn 0 oblicza dokładność zestawu danych, Wyniki dla zestawu danych: 0.97

