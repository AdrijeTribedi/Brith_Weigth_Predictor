from flask import Flask,request,jsonify,render_template

import pandas as pd

import pickle

app = Flask(__name__)

def get_cleared_data(form_data):
    gestation = float(form_data["gestation"])
    parity = int(form_data["parity"])
    age = float(form_data["age"])
    height = float(form_data["height"])
    weight = float(form_data["weight"])
    smoke = float(form_data["smoke"])
    
    cleared_data = {
        "gestation" : [gestation],
        "parity" : [parity],
        "age" : [age],
        "height" : [height],
        "weight" : [weight],
        "smoke" : [smoke]
    }
    return cleared_data

@app.route("/",methods=["GET"])
def home():
    return render_template("index.html")

#creating the endpoint
@app.route("/predict",methods=["POST"])
def get_prediction():
    # get data from user
    baby_data_form = request.form
    
    baby_data_cleaned = get_cleared_data(baby_data_form)
    #convert into dataframe
    baby_df = pd.DataFrame(baby_data_cleaned)
    
    # load machine trained model
    with open("model/model.pkl","rb") as obj:
        model = pickle.load(obj)
        
    # make prediction on user data
    prediction = model.predict(baby_df)
    prediction = round(float(prediction),2)
    
    # return prediction in a json format
    response = {"Prediction" : prediction}
    return render_template("index.html",prediction=prediction)
    
if(__name__) == "__main__":
    app.run(debug=True)
    
