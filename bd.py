import requests

url  = "https://raw.githubusercontent.com/google/eng-edu/main/ml/cc/exercises/taxi.csv"
r = requests.get(url)
with open("taxi.csv", "wb") as f:
    f.write(r.content)
print("Dataset baixado com sucesso!")