# Creating Flask app for model deployment
# Mode of Deployment - Web based (on Heroku cloud platform)
import numpy as np
from flask import Flask, request, render_template
import pickle 

app_web = Flask(__name__)

# Root endpoint
@app_web.route('/', methods = ['GET', 'POST'])
def home():
    return render_template('index.html')

# Predict endpoint
@app_web.route('/predict', methods = ['POST']) 
def predict():
    model = pickle.load(open('model.pickle', 'rb'))

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction_lep = model.predict(final_features)
    output_lep = round(prediction_lep[0], 2)
    return render_template('index.html', prediction_text = 'Life Expectancy is {} yrs'.format(output_lep))
 
if __name__ == "__main__":
    app_web.run(port = 5000, debug = True)