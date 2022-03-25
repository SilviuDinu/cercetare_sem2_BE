import os
import csv
import torch
import torchvision
from torchvision import transforms
import sys
import numpy as np

curr = os.path.dirname(__file__)

depression_simptoms = [
    'sadness',
    'anger',
    'irritability',
    'sleep_disturbances',
    'tiredness',
    'loss_of_appetite',
    'increased_appetite',
    'weight_loss',
    'weight gain',
    'anxiety',
    'guilt', 
    'lack_of_concentration',
    'trouble_remembering_things',
    'thoughts_of_death',
    'back_pain',
    'headache'
]
class TrainingData(torch.utils.data.Dataset):

    def __init__(self):

        curr = os.path.dirname(__file__)

        trainCSVPath = os.path.join(curr, r'data/csv-data/Training_despression_processed.csv')

        training_csv_output = os.path.join(curr, r'data/csv-data/processed_training_disertatie.csv')

        diseases_path = os.path.join(curr, r'data/csv-data/diseases_disertatie.csv')

        if not os.path.exists(training_csv_output):
            self.build_processed(trainCSVPath, training_csv_output, diseases_path)

        xy = np.loadtxt(training_csv_output, delimiter=',', dtype=np.float32, skiprows=1)

        self.x = torch.from_numpy(xy[:, 0:-1])
        self.y = torch.from_numpy(xy[:, [-1]])
        # print('self.x, self.y = ', self.x, self.y)
        self.n_samples = xy.shape[0]
        
        # print(self.X)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
    def get_diseases(self):
        return self.diseases

    def get_raw_data(self):
        return self.raw

    def get_disease_name(self, index):
        for disease in self.diseases:
            if disease[0] == index:
                return disease[1]
    
    def build_processed(self, trainCSVPath, training_csv_output, diseases_path):
        raw = []
        headers = []

        with open(trainCSVPath, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for idx, rows in enumerate(csv_reader):
                if idx > 0:
                    raw.append(rows)
                else:
                    headers.append(rows)

        raw_copy = raw
        self.raw = raw
        raw_diseases = []
    
        for data in raw_copy:
            curr = data[len(data) - 1]
            raw_diseases.append(curr)

        filtered_diseases = list(set(raw_diseases))

        mapped_diseases = []
        for idx, disease in enumerate(filtered_diseases):
            mapped_diseases.append([idx, disease])

        self.diseases = mapped_diseases

        if not os.path.exists(diseases_path):
            with open(diseases_path, 'w', newline='') as file:
                write = csv.writer(file)
                write.writerows(self.diseases)

        for idx, data in enumerate(raw_copy):
            for index, elem in enumerate(data):
                for disease in mapped_diseases:
                    if disease[1] == elem:
                        data[index] = disease[0]
                if elem == '1' or elem == '0' or elem == 1 or elem == 0:
                    data[index] = int(elem)
        
        with open(training_csv_output, 'w', newline='') as file:
            write = csv.writer(file)
            write.writerows(headers)
            write.writerows(raw_copy)
        



