from flask import Flask, make_response,request
from flask_cors import CORS, cross_origin

import modules.hello_world as hello_world
import modules.operations as operations
import modules.formating as formating

from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
#app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/api/hello_world', methods=['GET'])
@cross_origin()
def return_hello_world():
    return hello_world.hello_world()


@app.route('/api/operations/multiplicate/<parameters>', methods=['GET'])
@cross_origin()
def return_multiplication(parameters):
    numbers=parameters.split(',')
    return operations.multiplicate(numbers)

@app.route('/api/formating/lowercase_string/<string>', methods=['GET'])
@cross_origin()
def return_lowercase_string(string):
    return formating.lowercase_string(string)
