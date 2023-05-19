import pickle
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd

app = Flask(__name__)
swagger = Swagger(app)


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

    with open('./predict_income.pkl', 'rb') as f:
        double_exp = pickle.load(f)

    time_series = double_exp.forecast(100)

    date = f'{year}-{month}-01'
    value = time_series[date]

    if value is None:
        return jsonify({'error': 'Value not found for the specified month and year.'}), 404

    return jsonify(
        {
            'status': "ok",
            "date": date,
            'value': value
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

    with open('./predict_attendance.pkl', 'rb') as f:
        ar = pickle.load(f)

    time_series = ar.forecast(100)

    date = f'{year}-{month}-01'
    value = time_series[date]

    if value is None:
        return jsonify({'error': 'Value not found for the specified month and year.'}), 404

    return jsonify(
        {
            'status': "ok",
            "date": date,
            'value': value
        }
    )


if __name__ == '__main__':
    app.run(debug=True)
