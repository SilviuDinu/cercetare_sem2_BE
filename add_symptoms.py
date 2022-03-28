import os
import csv
import random
import math
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
    'weight_gain',
    'anxiety',
    'guilt',
    'lack_of_concentration',
    'trouble_remembering_things',
    'thoughts_of_death',
    'back_pain',
    'headache',
    'loss_of_interest'
]

difference = [
    'sadness',
    'anger',
    'sleep_disturbances',
    'tiredness',
    'guilt',
    'trouble_remembering_things',
    'thoughts_of_death',
    'loss_of_interest'
]


def get_random_value(max):
    return random.random() * 100 < max


def build_processed_file_for_depression(input_path, output_path):
    raw = []
    headers = []
    depression_patients = []
    total_depression_pacients = 0

    with open(input_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for idx, rows in enumerate(csv_reader):
            if idx > 0:
                original = list(map(int, rows[:-1]))
                original.extend([0] * 8)
                original.extend([rows[len(rows)-1]])
                raw.append(original)
            else:
                headers = rows[:-1]
                for idx1, symptom in enumerate(rows):
                    for idx2, depression_symptom in enumerate(depression_simptoms):
                        if depression_symptom not in headers:
                            headers.append(depression_symptom)
                headers.append(rows[len(rows)-1])

    random_pacients = random.sample(raw, math.floor(0.5 * len(raw)))
    for i, pacient in enumerate(random_pacients):
        total = 0
        total_not_depression = 0
        if get_random_value(25):
            for j, symptom in enumerate(pacient):
                symptom_name = headers[j]
                if symptom_name not in depression_simptoms and symptom == 1:
                    total_not_depression += 1
                    if get_random_value(85):
                        pacient[j] = 0
                if symptom_name in depression_simptoms and symptom == 0:
                    if get_random_value(65):
                        pacient[j] = 1
                        total += 1
                        # print(symptom_name)
            pacient[-1] = 'Depression'
            total_depression_pacients += 1
            depression_patients.append(pacient)

            # total_positive_depression_symptoms = total / \
            #     len(depression_simptoms)
            # if total_positive_depression_symptoms > 0.65:
            #     pacient[-1] = 'Depression'
            # print(pacient[-1])

        # print('How many depression from list of depression: {}'.format(total / len(depression_simptoms)))
        # print('How many positive symptoms that are not depression related: {}'.format(total_not_depression / (len(pacient) - 1)))
        # print('How many non-depression symptoms compared to total number of depression symptoms: {}'.format(total_not_depression / len(depression_simptoms)))

    print(total_depression_pacients)
    raw.extend(depression_patients)
    print(len(raw), len(headers))
    with open(output_path, 'w', newline='') as file:
        write = csv.writer(file)
        write.writerows([headers])
        # unique = [list(item) for item in set(tuple(row) for row in raw)]
        write.writerows(raw)
        # write.writerows(raw)


trainCSVPath = os.path.join(curr, r'data/csv-data/Training_depression.csv')
training_csv_output = os.path.join(
    curr, r'data/csv-data/Training_despression_processed.csv')

build_processed_file_for_depression(trainCSVPath, training_csv_output)

testCSVPath = os.path.join(curr, r'data/csv-data/Testing_depression.csv')
testing_csv_output = os.path.join(
    curr, r'data/csv-data/Testing_despression_processed.csv')

build_processed_file_for_depression(testCSVPath, testing_csv_output)

# STATS

total_training_depression = 0
total_training = 0
total_testing_depression = 0
total_testing = 0
total_diseases_train = []
total_diseases_test = []

with open(training_csv_output, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx, rows in enumerate(csv_reader):
        if idx > 0:
            disease_name = rows[-1]
            total_diseases_train.append(disease_name)
            if disease_name == 'Depression':
                total_training_depression += 1
            total_training += 1
        else:
            print('-')

with open(testing_csv_output, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx, rows in enumerate(csv_reader):
        if idx > 0:
            disease_name = rows[-1]
            total_diseases_test.append(disease_name)
            if disease_name == 'Depression':
                total_testing_depression += 1
            total_testing += 1
        else:
            print('-')

print('Training total depression pacients: {} -> {} \nTesting total depression pacients: {} -> {}'.format(total_training_depression,
      total_training_depression / total_training, total_testing_depression, total_testing_depression / total_testing))

print(len(list(set(total_diseases_train))),
      len(list(set(total_diseases_test))))
