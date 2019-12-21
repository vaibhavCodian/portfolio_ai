import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('./model_dev/model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    l_year_a = request.form["l_y_a"]
    day = request.form.get("day")
    religion = request.form.get("religion")
    gpa = request.form["gpa"]
    weather = request.form["weather"]

    test = np.array([[int(l_year_a), int(day), int(religion), int(gpa), int(weather)]])
    test.reshape(1, -1)
    res = model.predict(test)
    # output = round(prediction[0], 2)
    # return(str(religion))
    if int(res):
        return render_template('index.html', prediction_text='The Student Will Be Present On The Given Day 🥳')
    else:
        return render_template('index.html', prediction_text='The Student Will Not Be Present On The Given Day 😥')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

    


if __name__ == "__main__":
    app.run(debug=True)