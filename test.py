from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
import openai  # ✅ Correct import for latest SDK

# ✅ Set your OpenAI API key (keep this secret in production)
openai.api_key = ""

app = Flask(__name__)
CORS(app)

# File paths
data_path = "C:/Users/hp/OneDrive/Desktop/SRM/PROJECT"
history_path = os.path.join(data_path, "Backend")
os.makedirs(history_path, exist_ok=True)

# Product mapping
product_files = {
    "rice": os.path.join(data_path, "rice.csv"),
    "wheat": os.path.join(data_path, "wheat.csv"),
    "maize": os.path.join(data_path, "maize.csv"),
    "barley": os.path.join(data_path, "barley.csv"),
    "bajra": os.path.join(data_path, "bajra.csv"),
    "jowar": os.path.join(data_path, "jowar.csv"),
    "sugarcane": os.path.join(data_path, "sugarcane.csv"),
    "cotton": os.path.join(data_path, "cotton.csv"),
    "soybean": os.path.join(data_path, "soybean.csv"),
    "mustard": os.path.join(data_path, "mustard.csv"),
    "gram": os.path.join(data_path, "gram.csv"),
    "tur": os.path.join(data_path, "tur.csv"),
    "onion": os.path.join(data_path, "onion.csv"),
    "potato": os.path.join(data_path, "potato.csv"),
    "tomato": os.path.join(data_path, "tomato.csv"),
    "cauliflower": os.path.join(data_path, "cauliflower.csv"),
    "cabbage": os.path.join(data_path, "cabbage.csv"),
    "peas": os.path.join(data_path, "peas.csv"),
    "brinjal": os.path.join(data_path, "brinjal.csv"),
    "carrot": os.path.join(data_path, "carrot.csv"),
    "avocados": os.path.join(data_path, "avocados.csv"),
    "strawberries": os.path.join(data_path, "strawberries.csv")
}

usd_products = {"strawberries", "cauliflower", "carrots", "avocados"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        product = data["product"].lower()
        future_dates = [d.strip() for d in data["dates"].split(",")]

        if product not in product_files:
            return jsonify({"error": "Invalid product name"}), 400

        df = pd.read_csv(product_files[product])
        df.columns = df.columns.str.lower()

        if 'date' not in df.columns or 'value' not in df.columns:
            return jsonify({"error": "CSV must contain 'date' and 'value' columns"}), 400

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)

        if product in usd_products:
            df['value'] = df['value'].astype(str).str.replace('$', '', regex=False)
            df['value'] = df['value'].astype(float) * 85

        df = df.sort_values(by='date')

        # Anomaly Detection
        mean_price = df['value'].mean()
        std_dev = df['value'].std()
        latest_price = df['value'].iloc[-1]

        if latest_price > mean_price + 2 * std_dev:
            anomaly_note = "⚠️ Warning: Sudden price spike detected!"
        elif latest_price < mean_price - 2 * std_dev:
            anomaly_note = "⚠️ Notice: Price has dropped unusually low."
        else:
            anomaly_note = "✅ Prices are within normal range."

        sequence_length = 5
        for i in range(1, sequence_length + 1):
            df[f'lag_{i}'] = df['value'].shift(i)

        df.dropna(inplace=True)
        X = df.drop(columns=['date', 'value'])
        y = df['value']

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train({
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'seed': 42
        }, dtrain, num_boost_round=100)

        def forecast(model, recent_vals, future_dates):
            predictions = []
            last_sequence = recent_vals[-sequence_length:].tolist()
            for _ in future_dates:
                features = pd.DataFrame([last_sequence], columns=X.columns)
                scaled = scaler.transform(features)
                pred = model.predict(xgb.DMatrix(scaled))[0]
                predictions.append(pred)
                last_sequence.pop(0)
                last_sequence.append(pred)
            return predictions

        predicted_vals = forecast(model, df['value'].values, future_dates)
        predicted_prices = [round(float(p), 2) for p in predicted_vals]

        prediction_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price (INR)": predicted_prices
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{product}_{timestamp}.csv"
        file_path = os.path.join(history_path, filename)
        prediction_df.to_csv(file_path, index=False)

        return jsonify({
            "table": prediction_df.to_dict(orient="records"),
            "plot_data": {
                "actual": {
                    "x": df['date'].dt.strftime('%Y-%m-%d').tolist(),
                    "y": [float(v) for v in df['value']]
                },
                "predicted": {
                    "x": future_dates,
                    "y": predicted_prices
                }
            },
            "anomaly": anomaly_note
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chatbot", methods=["POST"])
def chatbot():
    question = request.json.get("question", "").strip()

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers agricultural questions, crop pricing, and recommendations."},
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"❌ Error: {str(e)}"})

@app.route("/recommend", methods=["POST"])
def recommend_crop():
    month = request.json.get("month", "").lower()
    crop_map = {
        "january": ["Mustard", "Potato"],
        "february": ["Mustard", "Wheat"],
        "march": ["Maize", "Sugarcane"],
        "april": ["Rice", "Maize"],
        "may": ["Maize", "Jowar"],
        "june": ["Cotton", "Soybean"],
        "july": ["Paddy", "Brinjal"],
        "august": ["Onion", "Tomato"],
        "september": ["Tomato", "Potato"],
        "october": ["Wheat", "Gram"],
        "november": ["Mustard", "Peas"],
        "december": ["Carrot", "Cabbage"]
    }
    return jsonify({"recommendation": crop_map.get(month, ["No recommendation available"])})

@app.route("/api/history", methods=["GET"])
def get_history():
    try:
        files = [f for f in os.listdir(history_path) if f.endswith('.csv')]
        return jsonify(sorted(files))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/static/<path:filename>")
def serve_file(filename):
    return send_from_directory(history_path, filename)

if __name__ == "__main__":
    app.run(debug=True)
