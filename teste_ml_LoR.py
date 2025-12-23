import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


df = pd.read_csv("taxi_class_large.csv")


def outlier_columns_iqr(df):#verifica se existe outliers e gera uma matriz 1d com as colunas com outliers
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



X = df[["passenger_count", "trip_distance", "fare_amount"]]
y = df["long_trip"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

outlier_cols = outlier_columns_iqr(X_train)
print("Colunas com outliers:", outlier_cols)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Se você não Padronizar, o modelo pode dar mais peso às features com valores maiores, mesmo que elas não sejam mais importantes.


model = LogisticRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("Confusion Matrix:\n", cm)


y_proba = model.predict_proba(X_test_scaled)[:,1]
print("Exemplo de probabilidades:", y_proba[:5])
