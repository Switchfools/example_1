from functools import reduce
import json 
from flask import Response


def multiplicate(numbers):
    return Response( json.dumps({'result':reduce(lambda x, y: x * y,map(int,numbers))}), mimetype='application/json')

