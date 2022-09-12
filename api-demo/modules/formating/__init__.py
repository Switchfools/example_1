from flask import Response


def lowercase_string(string):
    return Response(string.lower(), mimetype='application/json')

