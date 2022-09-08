
import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

## Loading the model
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # getting data in json type
    data = request.json['data']
    # printing the data
    print(data)
    # converting the data values in array and list and reshaping it to form a matrix
    print(np.array(list(data.values())).reshape(1,-1))
    # transforming the data and storing in new_data using standard scaler
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    # predicting the whole data and saving in output
    output = model.predict(new_data)
    # printing the first data
    print(output[0])
    return jsonify(output[0])


if __name__ == "__main__":
    app.run(debug=True)