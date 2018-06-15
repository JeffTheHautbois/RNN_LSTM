import csv
import json
from flask import Flask
from flask import Response
app = Flask(__name__)


def csv_to_json(filename):
    json_data = []
    for row in csv.DictReader(open(filename)):
        data = {}
        for field in row:
            key, _, sub_key = field.partition('.')
            if not sub_key:
                data[key] = row[field]
            else:
                if key not in data:
                    data[key] = [{}]
                data[key][0][sub_key] = row[field]

        #print(json.dumps(data, indent=True))
        #print('---------------------------')
        #json_data.append(json.dumps(data))
        json_data.append(data)
    return json.dumps(json_data)


breakfast_path = "spending_breakfast.csv"
breakfast_json = csv_to_json(breakfast_path)
print(breakfast_json)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/summary')
def summary():
    data = breakfast_json

    resp = Response(
        response=data,
        status=200,
        mimetype='application/json'
    )

    return resp