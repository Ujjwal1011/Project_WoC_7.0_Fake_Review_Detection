import pickle

with open('best_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

print(f"Loaded model type: {type(loaded_model)}")
