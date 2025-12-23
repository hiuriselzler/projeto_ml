#linear regression
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("taxi.csv")
print(df.head())

def outlier_columns_iqr(df):
    outlier_cols = []
    for col in df.select_dtypes(include="number"):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        if ((df[col] < lower) | (df[col] > upper)).any():
            outlier_cols.append(col)
    return outlier_cols



# Entradas
X = df[["trip_distance", "passenger_count"]]



# Saída
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



outlier_cols = outlier_columns_iqr(X_train)
print("Colunas com outliers:", outlier_cols)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)



plt.scatter(df["trip_distance"], df["fare_amount"], s=5)
plt.xlabel("Distance")
plt.ylabel("Fare")
plt.title("Preço da corrida x Distância")
plt.show()
