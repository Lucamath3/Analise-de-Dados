

# Importação de Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import GridSearchCV

# Divisão do dataset em treino e teste
X, y = datasets.load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=.30)

# Criação do Modelo
model_knn = KNeighborsClassifier()

# Treinando o modelo
model_knn.fit(x_train,y_train)

# Acurácia do Modelo
print("Acurácia do Modelo KNN: ", model_knn.score(x_test,y_test)*100)


# Implementação do Grid Search

knn_param_grid = {
    'n_neighbors': [5, 25],  # Number of neighbors to consider
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']  # Algorithm used for nearest neighbors search
}

classificador_knn = GridSearchCV(model_knn, knn_param_grid, cv=10, verbose=True,
                                 scoring='accuracy')

# Fit GridSearchCV com os dados de treino
classificador_knn.fit(x_train,y_train)

# Avaliação de Resultados
print("Tuned hpyerparameters of KNN :(best parameters) ", classificador_knn.best_params_)
print("Accuracy score of KNN :",(classificador_knn.best_score_)*100)
print(" ")