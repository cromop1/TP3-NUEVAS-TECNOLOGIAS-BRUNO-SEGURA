# ============================================
# TP - California Housing Dataset
# Regresión lineal vs regresión polinómica
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# ============================================
# 1. Cargar dataset California Housing
# ============================================
california = fetch_california_housing()

X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name="Price")

df = X.copy()
df["Price"] = y

print("=== Primeras filas ===")
print(df.head())

print("\n=== Dimensiones del dataset ===")
print(df.shape)

print("\n=== Información general ===")
print(df.info())

print("\n=== Valores nulos ===")
print(df.isnull().sum())

print("\n=== Estadísticas descriptivas ===")
print(df.describe())


# ============================================
# 2. Explorar correlaciones
# ============================================
print("\n=== Correlación con la variable objetivo (Price) ===")
corr_price = df.corr(numeric_only=True)["Price"].sort_values(ascending=False)
print(corr_price)


# ============================================
# 3. Dividir en train/test (80/20)
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n=== Tamaños de train y test ===")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================
# 4. Modelo 1: Regresión lineal
# ============================================
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train, y_train)

y_train_pred_lineal = modelo_lineal.predict(X_train)
y_test_pred_lineal = modelo_lineal.predict(X_test)

mse_train_lineal = mean_squared_error(y_train, y_train_pred_lineal)
mse_test_lineal = mean_squared_error(y_test, y_test_pred_lineal)

r2_train_lineal = r2_score(y_train, y_train_pred_lineal)
r2_test_lineal = r2_score(y_test, y_test_pred_lineal)

print("\n=== Regresión Lineal ===")
print(f"MSE Train: {mse_train_lineal:.4f}")
print(f"MSE Test : {mse_test_lineal:.4f}")
print(f"R² Train : {r2_train_lineal:.4f}")
print(f"R² Test  : {r2_test_lineal:.4f}")


# ============================================
# 5. Gráfico: predicciones vs valores reales
#    (modelo lineal)
# ============================================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred_lineal, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Regresión Lineal - Predicciones vs Valores Reales")
plt.grid(True)
plt.show()


# ============================================
# 6. Modelo 2: Regresión polinómica grado 2
# ============================================
modelo_polinomico = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression())
])

modelo_polinomico.fit(X_train, y_train)

y_train_pred_poly = modelo_polinomico.predict(X_train)
y_test_pred_poly = modelo_polinomico.predict(X_test)

mse_train_poly = mean_squared_error(y_train, y_train_pred_poly)
mse_test_poly = mean_squared_error(y_test, y_test_pred_poly)

r2_train_poly = r2_score(y_train, y_train_pred_poly)
r2_test_poly = r2_score(y_test, y_test_pred_poly)

print("\n=== Regresión Polinómica Grado 2 ===")
print(f"MSE Train: {mse_train_poly:.4f}")
print(f"MSE Test : {mse_test_poly:.4f}")
print(f"R² Train : {r2_train_poly:.4f}")
print(f"R² Test  : {r2_test_poly:.4f}")


# ============================================
# 7. Gráfico: predicciones vs valores reales
#    (modelo polinómico)
# ============================================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred_poly, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Regresión Polinómica Grado 2 - Predicciones vs Valores Reales")
plt.grid(True)
plt.show()


# ============================================
# 8. Comparar métricas
# ============================================
comparacion = pd.DataFrame({
    "Modelo": ["Regresión Lineal", "Regresión Polinómica Grado 2"],
    "MSE Train": [mse_train_lineal, mse_train_poly],
    "MSE Test": [mse_test_lineal, mse_test_poly],
    "R² Train": [r2_train_lineal, r2_train_poly],
    "R² Test": [r2_test_lineal, r2_test_poly]
})

print("\n=== Comparación de modelos ===")
print(comparacion)


# ============================================
# 9. Feature con más impacto
#    usando coeficientes del modelo lineal
# ============================================
coeficientes = pd.DataFrame({
    "Feature": X.columns,
    "Coeficiente": modelo_lineal.coef_
})

coeficientes["Impacto_absoluto"] = coeficientes["Coeficiente"].abs()
coeficientes = coeficientes.sort_values(by="Impacto_absoluto", ascending=False)

print("\n=== Coeficientes del modelo lineal ===")
print(coeficientes)


# ============================================
# 10. Respuestas automáticas orientativas
# ============================================
mejor_modelo = comparacion.sort_values(
    by=["R² Test", "MSE Test"], ascending=[False, True]
).iloc[0]["Modelo"]

feature_mas_importante = coeficientes.iloc[0]["Feature"]

print("\n=== RESPUESTAS SUGERIDAS ===")

print("\n1) ¿Cuál modelo es mejor? ¿Cómo lo determinaron?")
print(
    f"El modelo con mejor rendimiento fue: {mejor_modelo}. "
    "Se determinó comparando las métricas en test: "
    "un mejor modelo debe tener menor MSE y mayor R²."
)

print("\n2) ¿Hay señales de overfitting en alguno? ¿Cómo se dan cuenta?")
print(
    "Se detecta overfitting comparando train y test. "
    "Si un modelo tiene métricas mucho mejores en train que en test, "
    "entonces hay señales de sobreajuste. "
    "La regresión polinómica suele tener más riesgo de overfitting."
)

print("\n3) ¿Qué feature tiene más impacto en el precio? ¿Cómo lo averiguaron?")
print(
    f"La feature con mayor impacto en el modelo lineal fue: {feature_mas_importante}. "
    "Se averiguó observando el valor absoluto de los coeficientes del modelo lineal."
)
