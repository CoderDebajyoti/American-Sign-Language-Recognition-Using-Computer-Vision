import pickle
import os

if os.path.isfile('dataset.pkl'):
    with open('dataset.pkl', 'rb') as file:
        dataset = pickle.load(file)
    print(dataset)
else:
    print("The 'dataset.pkl' file doesn't exist. Create one by running 'create_dataset.py' file.")