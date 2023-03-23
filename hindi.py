import json
from googletrans import Translator
import pickle

f = open('./snli_1.0/snli_1.0_train_format.jsonl')
data = json.load(f)

f.close()

hindi_data = []
translator = Translator()

for i in range(len(data)):
    print(i/50*100,'%')
    if i == 50:
        break
    _ = data[i]
    h1 = translator.translate(_['sentence1'], src='en', dest ='hi').text
    h2 = translator.translate(_['sentence2'], src='en', dest ='hi').text
    
    hindi_data.append((_['annotator_labels'][0], h1, h2))

file = open('hindi_data.pkl', 'wb')
pickle.dump(hindi_data, file)

file.close()
    