import joblib
import numpy as np

model = joblib.load("model.pkl")

temp = float(input("Enter temperature: "))
pressure = float(input("Enter pressure: "))
flow = float(input("Enter flow rate: "))
conc = float(input("Enter concentration: "))

test = [[temp, pressure, flow, conc]]
prediction = model.predict(test)

print("\nPredicted Outputs:")
print("Yield:", prediction[0][0])
print("Efficiency:", prediction[0][1])
print("Purity:", prediction[0][2])


print("\n--- Optimization Mode ---")

target_yield = float(input("Target Yield: "))
target_eff = float(input("Target Efficiency: "))
target_purity = float(input("Target Purity: "))

best_input = None
best_error = float("inf")

for _ in range(1000):
    t = np.random.uniform(200, 500)
    p = np.random.uniform(1, 10)
    f = np.random.uniform(10, 100)
    c = np.random.uniform(0.1, 1.0)

    pred = model.predict([[t, p, f, c]])[0]

    error = abs(pred[0] - target_yield) + \
            abs(pred[1] - target_eff) + \
            abs(pred[2] - target_purity)

    if error < best_error:
        best_error = error
        best_input = [t, p, f, c]

print("\nBest Input Found:")
print("Temperature:", best_input[0])
print("Pressure:", best_input[1])
print("Flow Rate:", best_input[2])
print("Concentration:", best_input[3])