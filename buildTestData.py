import os
import csv
import torch
import torchvision
from torchvision import transforms
import sys
import numpy as np

curr = os.path.dirname(__file__)

class TestingData(torch.utils.data.Dataset):

    def __init__(self):

        curr = os.path.dirname(__file__)
        testCSVPath = input("Path to testing data: ")

        if not testCSVPath:
            testCSVPath = os.path.join(curr, r'data/csv-data/Testing.csv')

        testing_csv_output = os.path.join(curr, r'data/csv-data/processed_testing.csv')

        diseases_path = os.path.join(curr, r'data/csv-data/diseases.csv')

        if not os.path.exists(testing_csv_output):
            raw = []
            headers = []

            with open(testCSVPath, "r") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for idx, rows in enumerate(csv_reader):
                    if idx > 0:
                        raw.append(rows)
                    else:
                        headers.append(rows)

            raw_copy = raw
            self.raw = raw
            mapped_diseases = []
            with open(diseases_path, "r") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for idx, rows in enumerate(csv_reader):
                    mapped_diseases.append(rows)

            self.diseases = mapped_diseases

            for idx, data in enumerate(raw_copy):
                for index, elem in enumerate(data):
                    for disease in mapped_diseases:
                        if disease[1] == elem:
                            data[index] = disease[0]
                    if elem == '1' or elem == '0':
                        data[index] = int(elem)
          
            with open(testing_csv_output, 'w', newline='') as file:
                write = csv.writer(file)
                write.writerows(headers)
                write.writerows(raw_copy)
        

        xy = np.loadtxt(testing_csv_output, delimiter=',', dtype=np.float32, skiprows=1)

        self.x = torch.from_numpy(xy[:, 0:-1])
        self.y = torch.from_numpy(xy[:, [-1]])
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



