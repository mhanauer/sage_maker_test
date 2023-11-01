import argparse
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described by adding arguments
    parser.add_argument('--max_iter', type=int, default=1000)

    # Parse arguments
    args = parser.parse_args()

    # Load synthetic data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train a logistic regression model
    model = LogisticRegression(max_iter=args.max_iter)
    model.fit(X_train_scaled, y_train)

    # Save the model
    model_path = os.path.join('./', 'model.joblib')  # Changed directory to current directory
    joblib.dump(model, model_path)

    print(f'Saved model to {model_path}')
