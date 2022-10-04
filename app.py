import numpy as np
from flask import Flask, request, jsonify,render_template
import joblib
import pandas as pd
#from apiflask import APIFlask
app = Flask(__name__)

model = joblib.load(open('model.sav','rb'))
predict_data=pd.read_json(r'predict_df.json')
predict_data=predict_data.set_index('index')
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict(): 
    # Get the data from the POST request.
    client_id =[float(x) for x in request.form.values()]
    #client_id=request.get_json()
    #print(request.data)
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict_proba([predict_data.loc[client_id[0],:]])
    # Take the first value of prediction
    
    #output=jsonify(prediction[0][0])
    output=prediction[0][0]
    
    return render_template('index.html', Probability_of_payment='The Customer will pay loans with probability {}'.format(output))
    #return text

@app.route('/process_json',methods=['POST','GET'])
def predict_json():   # Get the data from the POST request.
    client_id=request.get_json()
    #print(request.data)
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict_proba([predict_data.loc[client_id[0],:]])
    # Take the first value of prediction
    
    output=jsonify(prediction[0][0])
   
    
    return output
    #return text

if __name__ == '__main__':
    # Load  model and data
   
    app.run(debug=True,port=8082)
   