from flask import Flask, redirect, url_for, render_template, request
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal-length'])
    sepal_width = float(request.form['sepal-width'])
    petal_length = float(request.form['petal-length'])
    petal_width = float(request.form['petal-width'])

    # Model prediction
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)[0]

    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)