import flask
import os
import csv
import torch
import torchvision
from torchvision import transforms
import sys
import numpy as np
import json
from flask import Response, request
from flask_cors import CORS, cross_origin
from net import Net

FILE = 'model.pth'

app = flask.Flask(__name__)
CORS(app)
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
                raw = rows[0:len(rows)-1]
    
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
    ids = request.get_json()['ids']
    req_symtpoms = request.get_json()['symptoms']
    print(req_symtpoms)

    net = Net()
    net.load_state_dict(torch.load(FILE))
    net.eval()
    input = []

    curr = os.path.dirname(__file__)
    path = os.path.join(curr, r'data/csv-data/Training.csv')

    with open(path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for idx, rows in enumerate(csv_reader):
            if idx == 0:
                symptoms = rows[0:len(rows)-1]
            
    for idx, symptom in enumerate(symptoms):
        for id in ids:
            if id == idx:
                input.append(1)
            else:
                input.append(0)

    print(input)
    print(len(input))
            
    # data = torch.from_numpy(np.array(ids, dtype='float32'))
    # print(data)
    # X, y = data
    # output = net(X)
    return Response(json.dumps({'success': 'ok'}), mimetype='application/json')

app.run()