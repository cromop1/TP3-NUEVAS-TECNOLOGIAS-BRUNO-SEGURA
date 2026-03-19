
<p align="center">
  <img src="https://i.imgur.com/EjIOCZB.png" width="100%">
</p>
<p align="center">
  <img src="https://i.imgur.com/RVGaecC.png" width="100%">
</p>

<p align="center">
<img src="https://img.shields.io/badge/Trabajo%20Academico-ITS%20Villada-blue?style=for-the-badge">
</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Machine%20Learning-Regresión-green?logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/Dataset-California%20Housing-orange?logo=databricks&logoColor=white">
<img src="https://img.shields.io/badge/Modelo-Regresión%20Lineal%20%2B%20Polinómica-purple?logo=scikitlearn&logoColor=white">
<img src="https://img.shields.io/badge/Métricas-MSE%20%7C%20R²-yellow?logo=chartdotjs&logoColor=white">
<img src="https://img.shields.io/badge/Plataforma-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white">

</p>










<p align="center">
  <img src="https://i.imgur.com/RVGaecC.png" width="100%">
</p>

<div align="center">

# 📘 Índice

[Introducción](#-introduccion)

[Objetivo](#-objetivo)

[Dataset Analizado](#-dataset-analizado)

[¿Qué predeciría el modelo?](#-que-predeciria-el-modelo)

[Análisis de Resultados](#-analisis-de-resultados)

[¿Cuál modelo es mejor?](#-cual-modelo-es-mejor-como-lo-determinaron)

[Overfitting](#-hay-senales-de-overfitting-en-alguno-como-se-dan-cuenta)

[Feature más importante](#-que-feature-tiene-mas-impacto-en-el-precio-como-lo-averiguaron)

[Gráficos](#-graficos)

</div>
<p align="center">
  <img src="https://i.imgur.com/RVGaecC.png" width="100%">
</p>

# 🔸 Introducción

En el campo de la **ciencia de datos**, los datasets representan una herramienta fundamental para entrenar modelos de **Machine Learning** capaces de identificar relaciones entre variables y realizar predicciones sobre datos reales.

Uno de los problemas más comunes dentro del análisis de datos es la **predicción de valores numéricos**, como por ejemplo precios, costos o tendencias del mercado.

En esta actividad se analiza el dataset **California Housing**, el cual contiene información sobre viviendas en distintas zonas de California.

A partir de estos datos es posible entrenar modelos de aprendizaje automático capaces de **predecir el precio de las viviendas**, identificando cómo influyen diferentes variables socioeconómicas y geográficas.



<p align="center">
  <img src="https://i.imgur.com/zDTIHyR.png" width="100%">
</p>

# 🔸 Objetivo

El objetivo de este trabajo es aplicar técnicas de **Machine Learning** para resolver un problema de regresión, utilizando el dataset California Housing.

Entre los puntos principales se encuentran:

* Analizar y explorar el dataset
* Entrenar modelos de regresión
* Evaluar el rendimiento mediante métricas
* Comparar distintos modelos

Este análisis permite comprender cómo los modelos pueden utilizarse para **predecir valores continuos en base a datos reales**, como el precio de viviendas.



<p align="center">
  <img src="https://i.imgur.com/zDTIHyR.png" width="100%">
</p>

# 🔹 Dataset Analizado

El dataset utilizado es **California Housing**, disponible en la librería *scikit-learn*.

🔗 Documentación oficial  
https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

Este dataset contiene información sobre viviendas en distintas regiones de California.

Cada registro incluye variables como:

* ingreso medio de la zona (MedInc)
* edad promedio de las viviendas
* número promedio de habitaciones
* población de la zona
* ubicación geográfica (latitud y longitud)

La variable objetivo es el **precio medio de la vivienda**, expresado como un valor numérico.

Esto permite analizar cómo diferentes factores influyen en el valor de una propiedad.



<p align="center">
  <img src="https://i.imgur.com/zDTIHyR.png" width="100%">
</p>

# 🔸 ¿Qué predeciría el modelo?

El modelo tiene como objetivo **predecir el precio de las viviendas** a partir de las características del dataset.

Dado que el valor a predecir es numérico, este problema se clasifica como un **problema de regresión**.

El modelo analiza las variables de entrada para estimar un valor continuo que represente el precio de una vivienda en función de sus características.

<p align="center">
  <img src="https://i.imgur.com/zDTIHyR.png" width="100%">
</p>
<p align="center">
  <img src="https://i.imgur.com/RVGaecC.png" width="100%">
</p>


# 📊 Análisis de Resultados

## ¿Cuál modelo es mejor? ¿Cómo lo determinaron?

El modelo que presenta mejor rendimiento es la **Regresión Polinómica de grado 2**.

Esto se determinó comparando las métricas obtenidas en el conjunto de test:

- Regresión Lineal:
  - MSE: 0.5559
  - R²: 0.5758

- Regresión Polinómica:
  - MSE: 0.4643
  - R²: 0.6457

El modelo polinómico tiene un **menor error (MSE)** y un **mayor coeficiente de determinación (R²)**, lo que indica que explica mejor la variabilidad del precio de las viviendas.

Por lo tanto, se concluye que el modelo polinómico es superior en este caso.

<p align="center">
  <img src="https://i.imgur.com/zDTIHyR.png" width="100%">
</p>

## ¿Hay señales de overfitting en alguno? ¿Cómo se dan cuenta?

No hay un overfitting grave, pero sí se observa una leve tendencia en la regresión polinómica.

Comparación:

- Regresión Lineal:
  - R² Train: 0.6126
  - R² Test:  0.5758

- Regresión Polinómica:
  - R² Train: 0.6853
  - R² Test:  0.6457

En ambos modelos, el rendimiento en entrenamiento es mejor que en test, lo cual es normal.

Sin embargo, en la regresión polinómica la diferencia es un poco mayor, lo que indica que el modelo es más complejo y comienza a ajustarse más a los datos de entrenamiento.

Aun así, como el rendimiento en test mejora, no se considera un overfitting problemático.

<p align="center">
  <img src="https://i.imgur.com/zDTIHyR.png" width="100%">
</p>

## ¿Qué feature tiene más impacto en el precio? ¿Cómo lo averiguaron?

La variable con mayor impacto en el modelo de regresión lineal es:

👉 **AveBedrms (promedio de dormitorios)**

Esto se determinó analizando los coeficientes del modelo:

- AveBedrms: 0.7831 (mayor impacto)
- MedInc: 0.4487
- Longitude: -0.4337
- Latitude: -0.4198

Se tomó el valor absoluto de los coeficientes para medir su influencia.

La variable **AveBedrms** es la que más influye en la predicción del precio según el modelo lineal.

Sin embargo, es importante destacar que variables como **MedInc (ingreso medio)** también tienen una alta correlación con el precio (0.688), por lo que su impacto es muy relevante en términos reales.

<p align="center">
  <img src="https://i.imgur.com/zDTIHyR.png" width="100%">
</p>
# 📈 GRAFICOS 
<p align="center">
  <img src="https://i.imgur.com/eqtSura.png" width="100%">
</p>
<p align="center">
  <img src="https://i.imgur.com/PCmq9Ol.png" width="100%">
</p>

<p align="center">
<img src="https://img.shields.io/badge/By_Bruno_Segura-blue?style=for-the-badge">
</p>
