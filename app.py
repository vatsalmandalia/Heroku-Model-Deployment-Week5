# Creating Flask app for model deployment
# Mode of Deployment - Web based (on Heroku cloud platform)
import numpy as np
from flask import Flask, request, render_template 
import pickle 

app = Flask(__name__)

# Root endpoint
@app.route('/', methods = ['GET', 'POST'])
def home():
    return render_template('index.html')

# Predict endpoint
@app.route('/predict', methods = ['POST']) 
def predict():
    model = pickle.load(open('model.pickle', 'rb'))

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction_lep = model.predict(final_features)
    output_lep = round(prediction_lep[0], 2)
    return render_template('index.html', prediction_text = 'Life Expectancy is {} yrs'.format(output_lep))
 
if __name__ == "__main__":
    app.run(port = 5000, debug = True)

# For testing the flask app api's on Postman
# Root endpoint
# @app.route('/', methods = ['GET', 'POST'])
# def home():
#     data = 'Testing the flask app APIs on Postman' 
#     return jsonify({'Its time!!':data})

# # Predict endpoint
# @app.route('/predict') 
# def predict():
#     model = pickle.load(open('model.pickle', 'rb'))

#     AdultMortality = request.args.get('AdultMortality')
#     infantdeaths = request.args.get('infantdeaths')
#     Alcohol = request.args.get('Alcohol')
#     BMI = request.args.get('BMI')

#     test_df = pd.DataFrame({'Adult_Mortality':[AdultMortality], 'infant_deaths':[infantdeaths], 'Alcohol_c':[Alcohol], 'BMI':[BMI]})
#     print('Values inputed by the user:\n')
#     print(test_df)

#     prediction = model.predict(test_df)   
#     return jsonify({'Life Expectancy':(str(prediction[0]) + "yrs")})

# if __name__ == "__main__":
#     app.run(port = 5000, debug = True)