
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load model
with open('random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

print(loaded_model.predict([[1, 2]]))
