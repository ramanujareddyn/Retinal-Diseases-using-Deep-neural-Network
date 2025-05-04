import cv2
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
import pickle
from sklearn.metrics import confusion_matrix #class to calculate accuracy and other metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
import keras
from keras import Model, layers
import pandas as pd
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout


X = []
Y = []
path = "SelectedImages"
labels = []

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index
print(labels)

'''
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (32, 32))
            X.append(img)
            label = getLabel(name)
            Y.append(label)
            print(name+" "+str(label))
            

X = np.asarray(X)
Y = np.asarray(Y)

np.save('model/X.txt',X)
np.save('model/Y.txt',Y)
'''
X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')
print(Y)
print(Y.shape)
print(np.unique(Y, return_counts=True))

X = X.astype('float32')
X = X/255

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

if os.path.exists('model/aug_X.txt.npy'):
    X = np.load('model/aug_X.txt.npy')
    Y = np.load('model/aug_Y.txt.npy')
else:
    print("no")
    aug = ImageDataGenerator(rotation_range=15, shear_range=0.8, horizontal_flip=True)
    data = aug.flow(X_train, y_train, 1)
    X = []
    Y = []  
    for x, y in data:
        x = x[0]
        y = y[0]
        X.append(x)
        Y.append(y)
        if len(Y) > 30000:
            break    
    X = np.asarray(X)
    Y = np.asarray(Y)
    np.save('model/aug_X.txt',X)
    np.save('model/aug_Y.txt',Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1) #split dataset into train and test
print(X.shape)
print(Y.shape)
print(np.unique(np.argmax(Y, axis=1), return_counts=True))
#now train eyenet with CNN 12 layers

eyenet_model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Conv2D(filters=16, kernel_size=(9,9), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Conv2D(filters=8, kernel_size=(7,7), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Conv2D(filters=8, kernel_size=(6,6), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Conv2D(filters=8, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(2,2), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(y_train.shape[1], activation='softmax')
])

opt = SGD(lr=0.001)
#opt = Adam(lr=0.0001)
eyenet_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])#compiling the model
#training and loading the model
if os.path.exists("model/sgd_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/sgd_weights.hdf5', verbose = 1, save_best_only = True)
    hist = eyenet_model.fit(X_train, y_train, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/sgd_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    eyenet_model.load_weights("model/sgd_weights.hdf5")

predict = eyenet_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)

opt = Adam(lr=0.001)
eyenet_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])#compiling the model
#training and loading the model
if os.path.exists("model/adam_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/adam_weights.hdf5', verbose = 1, save_best_only = True)
    hist = eyenet_model.fit(X_train, y_train, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/adam_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    eyenet_model.load_weights("model/adam_weights.hdf5")
   
predict = eyenet_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)


model = Sequential()
model.add(InputLayer(input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])))
model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
if os.path.exists("model/extension_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
    hist = model.fit(X_train, y_train, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/extension_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    model.load_weights("model/extension_weights.hdf5")

predict = model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)

