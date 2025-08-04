from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model and scaler
with open("kmeans_model.pickle", "rb") as f:
    kmeans = pickle.load(f)
with open("scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

# Updated 5-cluster labels (update these based on your cluster center analysis)
cluster_labels = {
    0: "Likely to Churn (Low Income & High Spending)",
    1: "Premium Customer (High Income & High Spending)",
    2: "Needs Engagement (High Income & Low Spending)",
    3: "Balanced Buyer (Moderate Income & Spending)",
    4: "Budget-Conscious (Low Income & Low Spending)"
}



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        income = float(request.form.get('income', 0))
        score = float(request.form.get('score', 0))

        if not (0 <= income <= 200):
            raise ValueError("Income must be between 0 and 200")
        if not (1 <= score <= 100):
            raise ValueError("Spending score must be between 1 and 100")

        input_data = pd.DataFrame([[income, score]], columns=['Annual Income (k$)', 'Spending Score (1-100)'])
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]
        result = cluster_labels.get(cluster, f"Cluster {cluster}")

        return render_template('index.html', prediction_text=f"Predicted Segment: {result}", success=True)

    except ValueError as ve:
        return render_template('index.html', prediction_text=f"Validation Error: {str(ve)}", success=False)
    except Exception as e:
        return render_template('index.html', prediction_text=f"System Error: {str(e)}", success=False)

if __name__ == '__main__':
    app.run(debug=True)
