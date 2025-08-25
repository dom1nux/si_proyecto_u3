
# TAREA - PROYECTO DE UNIDAD 3

## Trabajo de fin de unidad: Comparación de Algoritmos de Clasificación con Fashion-MNIST

### Objetivo General

Comparar el desempeño de Redes Neuronales con algoritmos de Machine Learning clásico (Naive Bayes, KNN, SVM, Random Forest) en la clasificación de imágenes de prendas de vestir del dataset Fashion-MNIST.

### Objetivos Específicos

1. Implementar y entrenar diferentes algoritmos de clasificación
2. Evaluar y comparar métricas de desempeño (accuracy, precision, recall, F1-score)
3. Analizar tiempos de entrenamiento y predicción
4. Generar visualizaciones comparativas de resultados
5. Crear un reporte con conclusiones sobre la efectividad de cada algoritmo

---

#### Dataset: Fashion-MNIST (Utilizar el que viene en keras)

Fashion-MNIST es un dataset de imágenes en escala de grises de 28x28 píxeles que contiene 70,000 imágenes de 10 categorías de prendas:

- 0: Camiseta/Top
- 1: Pantalón
- 2: Suéter
- 3: Vestido
- 4: Abrigo
- 5: Sandalia
- 6: Camisa
- 7: Zapatilla deportiva
- 8: Bolso
- 9: Botín

**Distribución del dataset:**

- Entrenamiento: 60,000 imágenes
- Prueba: 10,000 imágenes

---

## Parte 1: Preparación y Exploración de Datos (20 puntos)

### Actividades:

#### 1. Carga y exploración inicial (10 puntos)

```python
# Cargar Fashion-MNIST
from tensorflow.keras.datasets import fashion_mnist

# Mostrar información básica del dataset
# Visualizar muestras de cada clase
# Analizar distribución de clases
```

#### 2. Preprocesamiento de datos (10 puntos)

```python
# Normalización de píxeles (0-1)
# Aplanamiento para algoritmos clásicos (28x28 → 784)
# One-hot encoding para redes neuronales
# División train/validation para redes neuronales
```

**Entregables:**

- Gráficos de muestras del dataset
- Análisis estadístico básico
- Código de preprocesamiento documentado

---

## Parte 2: Implementación de Algoritmos Clásicos (30 puntos)

### 2.1 Naive Bayes (Grupo 1)

```python
from sklearn.naive_bayes import GaussianNB
# Implementar y entrenar modelo
# Evaluar desempeño
```

### 2.2 K-Nearest Neighbors (Grupo 2)

```python
from sklearn.neighbors import KNeighborsClassifier
# Probar diferentes valores de k (3, 5, 7, 9)
# Evaluar impacto del parámetro k
```

### 2.3 Support Vector Machine (Grupo 3)

```python
from sklearn.svm import SVC
# Implementar con kernel RBF y lineal
# Comparar kernels
```

### 2.4 Random Forest (Grupo 4)

```python
from sklearn.ensemble import RandomForestClassifier
# Experimentar con diferentes números de árboles
# Analizar importancia de características
```

**Entregables:**

- Código implementado para cada algoritmo
- Matrices de confusión para cada modelo
- Métricas de evaluación (accuracy, precision, recall, F1)
- Análisis de hiperparámetros

---

## Parte 3: Implementación de Redes Neuronales (30 puntos)

### 3.1 Red Neuronal Simple (Dense) (10 puntos)

```python
import tensorflow as tf
from tensorflow import keras

# Arquitectura sugerida:
# - Input: 784 neuronas
# - Hidden 1: 128 neuronas + ReLU
# - Hidden 2: 64 neuronas + ReLU
# - Output: 10 neuronas + Softmax
```

### 3.2 Red Neuronal Convolucional (CNN) (20 puntos)

```python
# Arquitectura sugerida:
# - Conv2D(32, 3x3) + ReLU + MaxPooling
# - Conv2D(64, 3x3) + ReLU + MaxPooling
# - Flatten
# - Dense(128) + ReLU + Dropout(0.5)
# - Dense(10) + Softmax
```

**Entregables:**

- Arquitecturas de las redes implementadas
- Gráficos de loss y accuracy durante entrenamiento
- Evaluación en conjunto de prueba
- Análisis de overfitting/underfitting

---

## Parte 4: Análisis Comparativo (15 puntos)

### 4.1 Métricas de Desempeño (8 puntos)

Crear tabla comparativa con:

- Accuracy
- Precision (macro/micro)
- Recall (macro/micro)
- F1-score (macro/micro)
- Tiempo de entrenamiento
- Tiempo de predicción

### 4.2 Análisis por Clase (7 puntos)

Identificar qué clases son más difíciles de clasificar

- Analizar patrones de errores comunes
- Comparar desempeño por algoritmo y por clase

**Entregables:**

- Tabla comparativa completa
- Gráficos de barras comparativos
- Heatmaps de matrices de confusión
- Análisis de errores por clase

---

## Parte 5: Reporte Final (5 puntos)

### Estructura del reporte:

1. **Introducción (0.5 puntos)**
	- Descripción del problema
	- Objetivos del estudio
2. **Metodología (1 punto)**
	- Descripción del dataset
	- Preprocesamiento aplicado
	- Algoritmos implementados
3. **Resultados (2 puntos)**
	- Presentación de métricas
	- Visualizaciones comparativas
	- Análisis de tiempos de ejecución
4. **Discusión (1 punto)**
	- Interpretación de resultados
	- Ventajas y desventajas de cada algoritmo
	- Casos de uso recomendados
5. **Conclusiones (0.5 puntos)**
	- Resumen de hallazgos principales
	- Recomendaciones

---

## Código Base para Iniciar

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Configuración de visualizaciones
plt.style.use('seaborn')
sns.set_palette("husl")

# Función para medir tiempo de ejecución
def measure_time(func):
	 def wrapper(*args, **kwargs):
		  start = time.time()
		  result = func(*args, **kwargs)
		  end = time.time()
		  return result, end - start
	 return wrapper

# Función para evaluar modelo
def evaluate_model(name, y_true, y_pred, training_time=None, prediction_time=None):
	 results = {
		  'Model': name,
		  'Accuracy': accuracy_score(y_true, y_pred),
		  'Training_Time': training_time,
		  'Prediction_Time': prediction_time
	 }

	 # Agregar métricas detalladas
	 report = classification_report(y_true, y_pred, output_dict=True)
	 results['Macro_Precision'] = report['macro avg']['precision']
	 results['Macro_Recall'] = report['macro avg']['recall']
	 results['Macro_F1'] = report['macro avg']['f1-score']

	 return results

# Labels de Fashion-MNIST
class_names = ['Camiseta/Top', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
					'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']
```

---

## Criterios de Evaluación

### Excelente (90-100%)

- Implementación correcta de todos los algoritmos
- Análisis profundo y crítico de resultados
- Visualizaciones claras y profesionales
- Reporte bien estructurado con conclusiones sólidas
- Código limpio y bien documentado

### Bueno (80-89%)

- Implementación correcta de la mayoría de algoritmos
- Análisis adecuado de resultados
- Visualizaciones apropiadas
- Reporte completo con algunas conclusiones
- Código funcional con documentación básica

### Satisfactorio (70-79%)

- Implementación básica de algoritmos principales
- Análisis superficial de resultados
- Visualizaciones básicas
- Reporte incompleto o con errores menores
- Código funcional pero poco documentado

### Insuficiente (<70%)

- Implementación incompleta o incorrecta
- Análisis ausente o erróneo
- Visualizaciones inexistentes o incorrectas
- Reporte ausente o muy deficiente
- Código no funcional

---

## Fechas Importantes

- **Entrega Final (Completa):** 24 agosto 11:55pm (incluye código e informe)
- **Presentaciones:** 25 agosto (hora de clase)

---

## Preguntas Reflexivas para el Reporte

1. ¿Por qué crees que las redes neuronales superan a los algoritmos clásicos en este dataset?
2. ¿En qué escenarios preferirías usar KNN sobre una red neuronal?
3. ¿Cómo afecta la dimensionalidad de las imágenes al desempeño de cada algoritmo?
4. ¿Qué trade-offs observas entre tiempo de entrenamiento y precisión?
5. ¿Cómo se compara Fashion-MNIST con MNIST en términos de dificultad de clasificación?

---

## Bonus (Puntos Extra) (Indicar en el informe para su evaluación)

- **+5 puntos:** Implementar técnicas de data augmentation para CNN
- **+3 puntos:** Analizar el impacto de diferentes optimizadores en redes neuronales
- **+3 puntos:** Implementar validación cruzada para algoritmos clásicos
