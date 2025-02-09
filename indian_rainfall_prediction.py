import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load dataset
df = pd.read_csv("rainfall in india 1901-2015.csv")
df.fillna(value=0, inplace=True)

def rainfall_prediction():
    states = df.SUBDIVISION.unique()
    state_models = {}
    
    for state in states:
        print(f"Training model for {state}...")
        grouped = df.groupby(df.SUBDIVISION)
        state_data = grouped.get_group(state)
        data = np.asarray(state_data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
                                    'AUG', 'SEP', 'OCT', 'NOV', 'DEC']])
        
        X, y = None, None
        for i in range(data.shape[1] - 3):
            if X is None:
                X = data[:, i:i+3]
                y = data[:, i+3]
            else:
                X = np.concatenate((X, data[:, i:i+3]), axis=0)
                y = np.concatenate((y, data[:, i+3]), axis=0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=1)
        model.fit(X_train, y_train)
        
        # Save model
        model_filename = f"models/{state.replace(' ', '_').lower()}_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    rainfall_prediction()