import pickle
from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit([[1, 2], [3, 4], [5, 6]], [0, 1, 0])

# Save model in pickle format
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)


##
## Save model in joblib format
##
from joblib import dump, load

# Save model
dump(model, 'random_forest_model.joblib')

# Load model
loaded_model = load('random_forest_model.joblib')
print(loaded_model.predict([[1, 2]]))
