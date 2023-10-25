import pickle
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    url_for,
)
from flasgger import Swagger
import pandas as pd
import numpy as np

app = Flask(__name__)
swagger = Swagger(app)

# Customize Swagger UI configuration
app.config['SWAGGER'] = {
    'title': 'ChurchML',  
    'description': 'An API for predicting Church Average Monthly Attendance and Income', 
    'uiversion': 3,
    'favicon': 'https://churchml.onrender.com/favicon.png'
}


with open('./predict_income.pkl', 'rb') as f:
  double_exp = pickle.load(f)

with open('./predict_attendance.pkl', 'rb') as f:
  ar = pickle.load(f)

with open('./scaler.pkl', 'rb') as f:
  scaler = pickle.load(f)
  
# fitted or previous
forecast_steps = 37  # Adjust the number of steps you want to forecast backward
ar_forecast_backwards = ar.predict(start=-forecast_steps, end=-1)

forecast_steps = 936  # Adjust the number of steps you want to forecast backward
ar_forecast_forewards = ar.forecast(forecast_steps)   

@app.route('/predict-income', methods=['POST'])
def predict_income():
    """
    Predicts income for a given month and year.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          properties:
            month:
              type: integer
              description: Month (1-12)
            year:
              type: integer
              description: Year (e.g., 2023)
    responses:
      200:
        description: Successful prediction
        schema:
          properties:
            status:
              type: string
              description: Status of the prediction
            date:
              type: string
              description: Date of the prediction
            value:
              type: float
              description: Predicted income value
      404:
        description: Value not found for the specified month and year.
    """
    data = request.get_json()
    month = data['month']
    year = data['year']

    # fitted or previous
    forecast_steps = 37  # Adjust the number of steps you want to forecast backward
    forecast_backwards = double_exp.predict(start=-forecast_steps, end=-1)
    forecast_backwards = scaler.inverse_transform(forecast_backwards.values.reshape(-1, 1))

    forecast_steps = 936  # Adjust the number of steps you want to forecast backward
    forecast_forewards = double_exp.forecast(forecast_steps)
    forecast_forewards = scaler.inverse_transform(forecast_forewards.values.reshape(-1, 1))

    values = np.concatenate((forecast_backwards, forecast_forewards)).flatten()
    index = np.concatenate((ar_forecast_backwards.index, ar_forecast_forewards.index))
    
    time_series = pd.Series(values, index=index)

    date = f'{year}-{month}-01'
    try:
      value = time_series[date]
    except KeyError:
      return jsonify(
        {
            'status': "ok",
            "date": date,
            'value': f"Not trained"
        }
      )

    if value is None:
        return jsonify({'error': 'Value not found for the specified month and year.'}), 404

    return jsonify(
        {
            'status': "ok",
            "date": date,
            'value': f"{value:.4f}"
        }
    )


@app.route('/predict-attendance', methods=['POST'])
def predict_attendance():
    """
    Predicts attendance for a given month and year.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          properties:
            month:
              type: integer
              description: Month (1-12)
            year:
              type: integer
              description: Year (e.g., 2023)
    responses:
      200:
        description: Successful prediction
        schema:
          properties:
            status:
              type: string
              description: Status of the prediction
            date:
              type: string
              description: Date of the prediction
            value:
              type: float
              description: Predicted attendance value
      404:
        description: Value not found for the specified month and year.
    """
    data = request.get_json()
    month = data['month']
    year = data['year']

    values = np.concatenate((ar_forecast_backwards, ar_forecast_forewards))
    index = np.concatenate((ar_forecast_backwards.index, ar_forecast_forewards.index))
    
    time_series = pd.Series(values, index=index)

    date = f'{year}-{month}-01'
    try:
      value = time_series[date]
    except KeyError:
      return jsonify(
        {
            'status': "ok",
            "date": date,
            'value': f"Not trained"
        }
      )

    if value is None:
        return jsonify({'error': 'Value not found for the specified month and year.'}), 404

    return jsonify(
        {
            'status': "ok",
            "date": date,
            'value': f"{int(value)}"
        }
    )


@app.route('/')
def index():
    return render_template('index.html', url_for=url_for)


@app.route('/predictIncome')
def income():
    return render_template(
      "income.html",
      url_for=url_for,
      years=range(2019, 2100),
      months=range(1, 13)
    )
  

@app.route('/predictAttendance')
def attendance():
    return render_template(
      "attendance.html",
      url_for=url_for,
      years=range(2019, 2100),
      months=range(1, 13)
    )


if __name__ == '__main__':
    app.run()
