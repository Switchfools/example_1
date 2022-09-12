from urllib import response
import orjson
from flask import Response


def hello_world():
    return Response("hello world", mimetype='application/json')

