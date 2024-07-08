from flask import Flask, request, jsonify
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from flask_cors import CORS
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/ML')
db = client.material_forecasting

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify("Hello darling")

@app.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        filepath = os.path.join("./tmp", file.filename)
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        file.save(filepath)
        
        data = pd.read_csv(filepath)
        db.data.insert_many(data.to_dict('records'))
        
        return jsonify({"message": "File successfully uploaded", "data": data.to_dict()}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/forecast', methods=['POST'])
def perform_forecast():
    try:
        records = list(db.data.find({}, {'_id': 0}).limit(10))
        
        # Assuming you have a function to process 'records' and return a forecast
        result = forecast(records)
        
        return jsonify({'forecast': result}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/data', methods=['GET'])
def get_data():
    try:
        records = list(db.data.find({}, {'_id': 0}))
        return jsonify(records), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def forecast(data):
    try:
        df = pd.DataFrame(data)
        df.set_index('Month', inplace=True)
        train = df.drop(columns=[' \"2005\"']).values.astype('float32')
        train = train.reshape(train.shape[0], 1, train.shape[1])
        model = Sequential()
        model.add(LSTM(50, input_shape=(1, train.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(train[:-1], train[1:], epochs=10, batch_size=1, verbose=2)
        prediction = model.predict(train[-1].reshape(1, 1, train.shape[2]))

        return float(prediction[0][0])

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
