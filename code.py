import pickle

file = open('hindi_data.pkl', 'rb')
data = pickle.load(file)

print(data)