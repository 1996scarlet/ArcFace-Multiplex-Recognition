import pickle
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
from helper import get_dataset

data_train = './Temp/train_data'
dataset_train = get_dataset(data_train)

class_names = [cls.name.replace('_', ' ') for cls in dataset_train]
print(class_names)

with open('./zoo/model-mlp/updated2.pkl', 'rb') as infile:
    (mlp, class_names) = pickle.load(infile)

with open('./zoo/model-mlp/updated.pkl', 'wb') as outfile:
    pickle.dump((mlp, class_names), outfile)

