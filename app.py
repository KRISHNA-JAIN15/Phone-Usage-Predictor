from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


app = Flask(__name__)


model = joblib.load('model.pkl')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    age = int(request.form['Age'])
    gender = request.form['Gender']
    location = request.form['Location']
    phone_brand = request.form['Phone Brand']
    os = request.form['OS']
    screen_time = float(request.form['Screen Time'])
    data_usage = float(request.form['Data Usage'])
    num_apps = int(request.form['Number of Apps Installed'])
    social_media_time = float(request.form['Social Media Time'])
    ecommerce_spend = float(request.form['E-commerce Spend'])
    streaming_time = float(request.form['Streaming Time'])
    gaming_time = float(request.form['Gaming Time'])
    recharge_cost = float(request.form['Monthly Recharge Cost'])
    primary_use = request.form['Primary Use']
    calls_duration = float(request.form['Calls Duration'])

    
    predict_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Location": [location],
        "Phone Brand": [phone_brand],
        "OS": [os],
        "Screen Time (hrs/day)": [screen_time],
        "Data Usage (GB/month)": [data_usage],
        "Number of Apps Installed": [num_apps],
        "Social Media Time (hrs/day)": [social_media_time],
        "E-commerce Spend (INR/month)": [ecommerce_spend],
        "Streaming Time (hrs/day)": [streaming_time],
        "Gaming Time (hrs/day)": [gaming_time],
        "Monthly Recharge Cost (INR)": [recharge_cost],
        "Primary Use": [primary_use],
        "Calls Duration (hrs/day)": [calls_duration],
    })

    
    predicted_time = round(model.predict(predict_data)[0], 2)

    return render_template('result.html', predicted_time=predicted_time[0])

if __name__ == '__main__':
    app.run(debug=True)
