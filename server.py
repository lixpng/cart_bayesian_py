from flask import Flask, url_for, request, Response
import bayesian
import cart
import time
import os
import json

app = Flask(__name__)

bayesian.main()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/upload-data', methods=['POST'])
def upload_data():
    ticks = time.time()
    training_file = './training.' + str(ticks) + '.data'
    test_file = './test.' + str(ticks) + '.data'

    training_data = request.files['training_data']
    training_data.save(training_file)
    test_data = request.files['test_data']
    test_data.save(test_file)


    cart.read_data(training_file)
    cart_res = cart.test(test_file)
    bayesian.read_data(training_file)
    bayesian_res = bayesian.test(test_file)

    res = {
        'bayesian': bayesian_res,
        'cart': cart_res
    }

    os.remove(training_file)
    os.remove(test_file)
    return Response(json.dumps(res), mimetype='application/json')

if __name__ == '__main__':
    app.run()