# Creating web app using flask for making classifier into production


from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/check1',methods = ['Post'])
def result1():
    review = request.form['review']
    from classifier_deployments.Afinn_Testing import classify
    pred, score = classify(review)
    return render_template('homepage.html', reviewtext = review, prediction = pred, active="active", score=score,
                           scoretext="Normalized Score")

@app.route('/check2',methods = ['Post'])
def result2():
    review = request.form['review']
    from classifier_deployments.MultinomialNB_Testing import classify
    pred, score = classify(review, 1)
    return render_template('homepage.html', reviewtext=review, prediction=pred, active="active", score=round(score,2),
                           scoretext="Probability")

@app.route('/check3',methods = ['Post'])
def result3():
    review = request.form['review']
    from classifier_deployments.MultinomialNB_Testing import classify
    pred, score = classify(review, 2)
    return render_template('homepage.html', reviewtext=review, prediction=pred, active="active", score=round(score,2),
                           scoretext="Probability")

@app.route('/check4',methods = ['Post'])
def result4():
    review = request.form['review']
    from classifier_deployments.RNN_Custom_Testing import classify
    pred, score = classify(review)
    return render_template('homepage.html', reviewtext=review, prediction=pred, active="active", score=round(score,2),
                           scoretext="Probability")

@app.route('/check5',methods = ['Post'])
def result5():
    review = request.form['review']
    from classifier_deployments.RNN_GloVe_Testing import classify
    pred, score = classify(review)
    return render_template('homepage.html', reviewtext=review, prediction=pred, active="active", score=round(score,2),
                           scoretext="Probability")

@app.route('/check6',methods = ['Post'])
def result6():
    review = request.form['review']
    from classifier_deployments.CNN_testing import classify
    pred, score = classify(review)
    return render_template('homepage.html', reviewtext=review, prediction=pred, active="active")

if __name__ == '__main__':
    app.run(debug=True)
