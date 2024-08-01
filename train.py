import pickle
import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib

if os.path.isfile('dataset.pkl'):
    with open('dataset.pkl', 'rb') as file:
        dataset = pickle.load(file)
        # Split the dataset into features and target
        X = dataset['data']  # Features
        y = dataset['labels']  # Target variable (Outcome)

        # Split the data into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        # Train the MLP classifier
        mlp = MLPClassifier(hidden_layer_sizes=(26, 26, 26, 26), max_iter=80000, random_state=1)
        # mlp.fit(X_train, y_train)
        mlp.fit(X, y)
        
        # Evaluate the model on the train set
        # accuracy = mlp.score(X_train, y_train)
        # print(f"Train accuracy: {accuracy * 100:.2f}%")
        
        # Evaluate the model on the test set
        # accuracy = mlp.score(X_test, y_test)
        # print(f"Test accuracy: {accuracy * 100:.2f}%")

        # Evaluate the model on the test set
        accuracy = mlp.score(X, y)
        print(f"Total accuracy: {accuracy * 100:.2f}%")

        # Save the trained model
        joblib.dump(mlp, 'trained_model.joblib')
        
        # Make predictions on new data
        new_data = [[1.4169514530034282, 0.4660353065263077, -2.5079897955519486, -0.17304476253401682, -2.69208497104052, -1.9822531062442406, 3.0240048088590283, 0.09944239421821577, -1.2862565618154265, 3.138588834559428, 1.3037546177892818, -0.3751314149357923, 0.9230696092310282, -1.707921687769879, -3.046223965243571, 2.8976168662306576, 0.044288038268602115, 3.0095935163499052, 1.0723097952051035, -3.1199038796922123, 0.05700664264814726]]  # Replace with your data
        prediction = mlp.predict(new_data)
        
        print(f'The character is likely {prediction[0]}.')
else:
    print("The 'dataset.pkl' file doesn't exist. Create one by running 'create_dataset.py' file.")