import numpy as np
from flask import Flask, request, jsonify
import joblib
import pandas as pd
#from apiflask import APIFlask
app = Flask(__name__)

model = joblib.load(open('model.sav','rb'))
predict_data=pd.read_json(r'predict_df.json')
predict_data=predict_data.set_index('index')

@app.route('/api',methods=['POST','GET'])
def predict(): 
    # Get the data from the POST request.
    client_id = request.get_json()
    #text = request.get_json()
    #print(request.data)
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict_proba([predict_data.loc[client_id['client_num'],:]])
    # Take the first value of prediction
    
    return jsonify(prediction[0][0])
    #return text
if __name__ == '__main__':
    # Load  model and data
   
    app.run(debug=True,port=1238)
   