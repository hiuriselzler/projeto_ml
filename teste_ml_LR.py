#linear regression
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


df = pd.read_csv("taxi.csv")
print(df.head())

# Entradas
X = df[["trip_distance", "passenger_count"]]



# Saída
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)


plt.scatter(df["trip_distance"], df["fare_amount"], s=5)
plt.xlabel("Distance")
plt.ylabel("Fare")
plt.title("Preço da corrida x Distância")
plt.show()
