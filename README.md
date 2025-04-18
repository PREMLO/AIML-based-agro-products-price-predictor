# AIML-based-agro-products-price-predictor
Description

This is an AI-powered web application that forecasts the prices of agricultural products using historical data and machine learning. Designed with farmers and agri-analysts in mind, the tool leverages the XGBoost regression model to provide accurate future price predictions.

The system includes:

Crop price prediction

Crop recommendation engine based on the month

Anomaly detection for price spikes or drops

Chatbot assistant powered by OpenAI (GPT-3.5)

Features

Predict prices for 20+ agro products

Plotly.js graphs to visualize trends

Downloadable prediction history in CSV format

Intelligent chatbot using GPT-3.5

Crop recommendation by season

Admin and user-specific dashboards

Tech Stack

Backend: Python, Flask, Flask-CORS

Machine Learning: XGBoost, Scikit-Learn

Frontend: HTML, CSS, JavaScript, Plotly.js

External API: OpenAI GPT-3.5

📁 Project Structure

/PROJECT
├── test.py               # Flask API backend
├── /frontend             # All HTML files
│   ├── login.html
│   ├── user_landing.html
│   ├── admin_landing.html
│   ├── user_profile.html
│   └── admin_profile.html
├── /Backend              # Folder to store prediction CSVs
├── /datasets             # All crop CSV files

⚙️ Installation & Setup

Clone the repository:

git clone https://github.com/your-username/agro-price-predictor.git
cd agro-price-predictor

Install required packages:

pip install flask flask-cors pandas numpy xgboost scikit-learn openai

Set your OpenAI API Key inside test.py:

openai.api_key = "your-api-key"

Run the backend server:

python test.py

Open frontend/login.html in your browser.

How to Use

Use login.html to login as either:

User: Prem / user123

Admin: Anitej / admin123

Navigate between prediction, history, chatbot, and profile.

 Screenshots
![Screenshot 2025-04-14 192203](https://github.com/user-attachments/assets/8214610b-bfe1-40f8-8beb-7fc6517cb86a)
![Screenshot 2025-04-14 192233](https://github.com/user-attachments/assets/4881c73c-94cd-4afa-af9f-093a13a9bc61)
![Screenshot 2025-04-14 192249](https://github.com/user-attachments/assets/51ea5657-1cc8-4710-9aa7-12de39970de4)
![Screenshot 2025-04-14 192304](https://github.com/user-attachments/assets/6a3b404b-1a6b-42a6-9c97-bb9830375862)
![Screenshot 2025-04-14 192334](https://github.com/user-attachments/assets/b31fe857-fdb0-416b-9eec-384234d4bfd2)
![Screenshot 2025-04-14 192434](https://github.com/user-attachments/assets/c435ed88-4a9b-436f-9a57-cdb01c1ba9e5)
![Screenshot 2025-04-14 192422](https://github.com/user-attachments/assets/3b2ce5ab-c3b5-4582-8715-563c1eaa0479)
![Screenshot 2025-04-14 192449](https://github.com/user-attachments/assets/ebf6beb6-d5e4-48dc-b828-948ed8b5fa0d)
![Screenshot 2025-04-14 192545](https://github.com/user-attachments/assets/ce5736d5-666d-4963-b485-6db4d4434cc8)



📝 License

This project is licensed under the MIT License, which is a permissive open-source license. It allows anyone to use, modify, distribute, and sublicense your project, as long as they include the original license and copyright notice.

MIT License Summary:
 Can be used for commercial and private purposes
 Can be modified and redistributed

 No warranty provided by the original authors

 Authors
Prem Lohia
Anitej Mishra


For queries, please contact via the GitHub issues section.
