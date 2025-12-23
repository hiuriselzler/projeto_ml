import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

ride_id = np.arange(1, n+1)
passenger_count = np.random.randint(1, 5, n)
trip_distance = np.round(np.random.uniform(0.5, 5.0, n), 2)
fare_amount = np.round(5 + trip_distance * np.random.uniform(1.5, 4.0, n), 2)
long_trip = (trip_distance > 2.0).astype(int)

df = pd.DataFrame({
    "ride_id": ride_id,
    "passenger_count": passenger_count,
    "trip_distance": trip_distance,
    "fare_amount": fare_amount,
    "long_trip": long_trip
})

df.to_csv("taxi_class_large.csv", index=False)
print(df.head())
