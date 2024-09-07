from flask import Flask, request, jsonify
import recc_mk1 as recc
from waitress import serve

app = Flask(__name__)

# Define a route that handles POST requests
@app.route('/api', methods=['POST'])
def echo():
    data = request.json
    skills = data["skills"]
    skills = ' '.join(skills)
    loc = data["location"]
    resp = recc.recc(str(skills),str(loc))
    int_list = [int(x) for x in resp]
    return jsonify({"indices": int_list})

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
