from utils import db_connect
engine = db_connect()

# your code here
# Explore here
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
# VER BASE DE DATOS:

df_costo_seguro = pd.read_csv('../data/raw/medical_insurance_cost.csv')
print(df_costo_seguro.head())
# VER INFO DE LA BASE DE DATOS:

print(f"Información General: {df_costo_seguro.info()}")

print(f"Estadísticas: {df_costo_seguro.describe()}")

print(f"Comprobar valores nulos: {df_costo_seguro.isnull().sum()}")
# PRIMER VISTAZO GRÁFICO:

fig, axis = plt.subplots(2, 3, figsize = (20, 15))

# Historiograma de los costes del seguro:

sns.histplot(ax=axis[0, 0], data=df_costo_seguro, x='charges', kde=True, color='indigo', edgecolor='black')
axis[0, 0].set_title("Distribución del Coste del Seguro", fontsize=14)
axis[0, 0].set_xlabel("Coste del seguro", fontsize=12)
axis[0, 0].set_ylabel("Frecuencia", fontsize=12)

# Gráfico pastel de distribución de fumadores:

fumadores = df_costo_seguro['smoker'].value_counts()
plt.pie(fumadores,
        labels=fumadores.index,
        autopct='%1.1f%%',
        colors=sns.color_palette('dark'))
axis[0, 1].set_title("Fumadores", fontsize=14)
axis[0, 1].axis('equal')

# Gráfico de barras con número de hijos:

sns.countplot(ax=axis[0, 2], data=df_costo_seguro, x='children', color='indigo', edgecolor='black')
axis[0, 2].set_title("Cantidad de hijos", fontsize=14)
axis[0, 2].set_xlabel("Numero de Hijos", fontsize=12)
axis[0, 2].set_ylabel("Frecuencia", fontsize=12)

# Boxplot coste por fumador:

sns.boxplot(ax=axis[1, 0], data=df_costo_seguro, x='smoker', y='charges', palette='dark')
axis[1, 0].set_title("Coste del Seguro para Fumadores", fontsize=14)
axis[1, 0].set_xlabel("Fumador", fontsize=12)
axis[1, 0].set_ylabel("Coste del seguro", fontsize=12)

# Boxplot coste por género:

sns.boxplot(ax=axis[1, 1], data=df_costo_seguro, x='sex', y='charges', palette='dark')
axis[1, 1].set_title("Coste del Seguro por Género", fontsize=14)
axis[1, 1].set_xlabel("Género", fontsize=12)
axis[1, 1].set_ylabel("Coste del seguro", fontsize=12)

# Heatmap de correlación:

correlacion = df_costo_seguro.corr(numeric_only=True)
sns.heatmap(ax=axis[1, 2], data=correlacion, annot=True, cmap='coolwarm')
axis[1, 2].set_title("Correlación entre las variables", fontsize=14)

plt.tight_layout()
plt.show()
# PREPARACIÓN DE DATOS:

df_seguro = pd.get_dummies(df_costo_seguro, drop_first=True) 
print(df_seguro)
# CREAR MODELO DE REGRESIÓN LINEAL:

X = df_seguro.drop('charges', axis=1)
y = df_seguro['charges']

# División entre entrenamiento y test:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Crear y entrenar modelo:

model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones:

y_pred = model.predict(X_test)

# Evaluar modelo:

print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))