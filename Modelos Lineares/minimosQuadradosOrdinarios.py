# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

## Etapa 1 - Preparação dos dados
X, y = load_diabetes(return_X_y=True)
X = X[:,[2]] # Utiliza apenas uma característica
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=20, shuffle=False)

## Etapa 2 - Modelo de Regressão

# Obs: por padrão uma interceptação (w0) é adicionado no modelo. No entanto, é possível controlar esse valor definindo
# o parâmetro "fit_interpect". Por exemplo: regressor = LinearRegression(fit_intercept=5).fit(X_train, y_train)

regressor = LinearRegression().fit(X_train, y_train)

## Etapa 3 - Avaliação do Modelo

# Para avaliar o modelo podemos utilizar o erro quadratico médio e o coeficiente de determinação
y_pred = regressor.predict(X_test)

print(f"Erro Quadrático Médio: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coeficiente de determinação: {r2_score(y_test, y_pred):.2f}")


## Etapa 4 - Plotando os resultados

import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)

ax[0].scatter(X_train, y_train, label="Pontos dos dados de Treino")
ax[0].plot(
    X_train,
    regressor.predict(X_train),
    linewidth=3,
    color="tab:orange",
    label="Model predictions",
)
ax[0].set(xlabel="Características", ylabel="Alvo", title="Conjunto de Treino")
ax[0].legend()

ax[1].scatter(X_test, y_test, label="Pontos dos dados de Teste")
ax[1].plot(X_test, y_pred, linewidth=3, color="tab:orange", label="Predições do Modelo")
ax[1].set(xlabel="Características", ylabel="Target", title="Conjunto de Teste")
ax[1].legend()

fig.suptitle("Regressão Linear")

plt.show()





