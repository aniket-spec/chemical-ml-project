from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    temp = float(request.form["temp"])
    pressure = float(request.form["pressure"])
    flow = float(request.form["flow"])
    conc = float(request.form["conc"])

    prediction = model.predict([[temp, pressure, flow, conc]])[0]

    results = {
        "yield": round(prediction[0], 4),
        "efficiency": round(prediction[1], 4),
        "purity": round(prediction[2], 4),
    }
    return render_template("index.html", mode="predict", results=results,
                           inputs={"temp": temp, "pressure": pressure, "flow": flow, "conc": conc})

@app.route("/optimize", methods=["POST"])
def optimize():
    target_yield = float(request.form["target_yield"])
    target_eff = float(request.form["target_eff"])
    target_purity = float(request.form["target_purity"])

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

    results = {
        "temp": round(best_input[0], 2),
        "pressure": round(best_input[1], 4),
        "flow": round(best_input[2], 2),
        "conc": round(best_input[3], 4),
    }
    return render_template("index.html", mode="optimize", results=results,
                           targets={"yield": target_yield, "efficiency": target_eff, "purity": target_purity})

if __name__ == "__main__":
    app.run(debug=True)
