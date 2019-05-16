import cv2
import sys
import numpy as np
import time
from termcolor import colored
import matplotlib.pyplot as plt
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn import metrics

from helper import read_pkl_model, start_up_init, get_dataset, get_image_paths_and_labels
import face_embedding

# =================== ARGS ====================
args = start_up_init()

# =================== MODEL CLASS ====================
arcface = face_embedding.EmbeddingModel(args)

# =================== LOAD DATASET ====================.
dir_train = './Embedding/train.npy'
data_train = './Temp/train_data'
dataset_train = get_dataset(data_train)
paths_train, labels_train = get_image_paths_and_labels(dataset_train)

dir_safe = './Embedding/safe.npy'
data_safe = './Temp/safe_data'
dataset_safe = get_dataset(data_safe)
paths_safe, labels_safe = get_image_paths_and_labels(dataset_safe)

dir_danger = './Embedding/danger.npy'
data_danger = './Temp/danger_data'
dataset_danger = get_dataset(data_danger)
paths_danger, labels_danger = get_image_paths_and_labels(dataset_danger)

try:
    train_emb_array = np.load(dir_train)
except OSError:
    train_emb_array = arcface.get_features_from_path(paths_train)
    np.save(dir_train, train_emb_array)

print('Train dataset reloaded: ', len(train_emb_array))

try:
    safe_emb_array = np.load(dir_safe)
except OSError:
    safe_emb_array = arcface.get_features_from_path(paths_safe)
    np.save(dir_safe, safe_emb_array)

print('Safe dataset reloaded: ', len(safe_emb_array))

try:
    danger_emb_array = np.load(dir_danger)
except OSError:
    danger_emb_array = arcface.get_features_from_path(paths_danger)
    np.save(dir_danger, danger_emb_array)

print('Danger dataset reloaded: ', len(danger_emb_array))

# print(labels)

class_names = [cls.name.replace('_', ' ') for cls in dataset_train]
print(class_names)

# X_train, X_test, y_train, y_test = train_test_split(train_emb_array, labels)

mlp = MLPClassifier(hidden_layer_sizes=(550, ), verbose=True,
                    activation='relu', solver='adam', tol=10e-7, n_iter_no_change=100,
                    learning_rate_init=10e-4, max_iter=50000)
# model.fit(X_train, y_train)
# # model.fit(train_emb_array, labels)
# print('train dataset accuracy: %.2f%%' %
#       (100 * metrics.accuracy_score(y_test, model.predict(X_test))))

mlp.fit(train_emb_array, labels_train)

with open('./zoo/model-mlp/mlp.pkl', 'wb') as outfile:
    pickle.dump((mlp, class_names), outfile)

with open('./zoo/model-mlp/mlp.pkl', 'rb') as infile:
    (mlp, class_names) = pickle.load(infile)

safe_prob = mlp.predict_proba(safe_emb_array)
danger_prob = mlp.predict_proba(danger_emb_array)


plt.hist(safe_prob.max(1), bins=200,
         facecolor='blue', alpha=0.5, label='safe-arcface')
plt.hist(danger_prob.max(1), bins=200,
         facecolor='red', alpha=0.5, label='danger-arcface')

with open('./model-mlp/mlp-facenet.pkl', 'rb') as infile:
    (mlp, class_names) = pickle.load(infile)

safe_prob = mlp.predict_proba(safe_emb_array)
danger_prob = mlp.predict_proba(danger_emb_array)

plt.hist(safe_prob.max(1), bins=200,
         facecolor='yellow', alpha=0.5, label='safe-facenet')
plt.hist(danger_prob.max(1), bins=200,
         facecolor='green', alpha=0.5, label='danger-facenet')

plt.legend()
plt.title("Split into %d parts of the training sample" % len(class_names))
plt.show()
