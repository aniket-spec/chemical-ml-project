import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Fake dataset (same as before)
X = np.random.rand(100, 4)

y = np.column_stack([
    X[:,0]*10 + X[:,1]*5,
    X[:,2]*7 + X[:,3]*3,
    X[:,0]*2 - X[:,3]*4
])

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained and ready to predict inputs")