import flask
import os
import csv
import torch
import torchvision
from torchvision import transforms
import sys
import numpy as np
import json
from flask import Response


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

@app.route('/api/v2/symptoms', methods=['GET'])
def get_symptoms():
    curr = os.path.dirname(__file__)
    path = os.path.join(curr, r'data/csv-data/Training.csv')
    raw = []

    with open(path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for idx, rows in enumerate(csv_reader):
            if idx == 0:
                raw = rows
    
    mapped = []
    
    for idx, symp in enumerate(raw):
        print(symp)
        if '_' in symp:
            new_symp = ' '.join(symp.split('_'))
        else:
            new_symp = symp
        mapped.append({"id": idx, "symptom": new_symp})
    
    return Response(json.dumps(mapped), mimetype='application/json')


@app.route('/api/v2/predict', methods=['POST'])
def predict():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

app.run()