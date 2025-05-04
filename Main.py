#importing require python packages and classes
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
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
from keras.callbacks import ModelCheckpoint
import keras
from keras import Model, layers
import pandas as pd
from keras.optimizers import SGD #import SGD class
from keras.optimizers import Adam #import Adam class optimizer
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout



main = tkinter.Tk()
main.title("EyeDeep-Net")
main.geometry("1300x1200")

#define global variables to calculate and store accuracy and other metrics
precision = []
recall = []
fscore = []
accuracy = []
global filename, dataset,testImages,testLabels
global data,labels,xlabel,ylabel,extension_gru,extension_model
global X, Y,data
global X_train, X_test, y_train, y_test,eyenet_model
global X_train, X_val, y_train, y_val

#define global variables
X = []
Y = []
path = "SelectedImages"
labels = []


#define function to load class labels
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
print("Retinal Diseases Class Labels found in dataset : "+str(labels))



def UploadDataset():
    global filename, dataset, labels, X_train, Y_train, text,X,Y
    text.delete('1.0', END)

    filename = filedialog.askdirectory(initialdir='Dataset')
    
    text.insert(END,filename+" loaded\n\n")
    
    if os.path.exists('model/X.txt.npy'):#if images already processed then load all images
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:#if not processed then read and process each image
        X = []
        Y = []
        for root, dirs, directory in os.walk(path):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])#read image
                    img = cv2.resize(img, (32, 32))#resize image
                    X.append(img)#addin images features to training array
                    label = getLabel(name)
                    Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)            
    text.insert(END,"Dataset Loading Completed"+"\n")
    text.insert(END,"Total images found in dataset Before Augmentation : "+str(X.shape[0])+"\n")      

    #visualizing class labels count found in dataset
    names, count = np.unique(Y, return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (5, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph Before Augmentation")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.show()

    

def PreprocessDataset():
    global filename, dataset,labels,vectorizer
    global X, Y
    global X_train, X_test, y_train, y_test
    global X_train, X_val, y_train, y_val
    #dataset preprocessing such as shuffling and normalization
    text.delete('1.0', END)
    #preprocess images like shuffling and normalization
    X = X.astype('float32')
    X = X/255 #normalized pixel values between 0 and 1
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) #shuffle all images
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    #now apply augmentation on splitted train images
    if os.path.exists('model/aug_X.txt.npy'):
        X = np.load('model/aug_X.txt.npy')
        Y = np.load('model/aug_Y.txt.npy')
    else:
        aug = ImageDataGenerator(rotation_range=15, shear_range=0.8, horizontal_flip=True)#apply augmentation to increase images
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
    text.insert(END,"Image Augmentation Completed")
    text.insert(END,"Total images found in dataset After Augmentation : "+str(X.shape[0])+"\n")

    #split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    text.insert(END,"Training Images Size   : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Validation Images Size : "+str(X_val.shape[0])+"\n")
    text.insert(END,"Testing Images Size    : "+str(X_test.shape[0])+"\n")
    
    #plotting graph of augmented images class labels
    names, count = np.unique(np.argmax(Y, axis=1), return_counts=True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (5, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph After Augmentation")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.show()


        
#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    text.delete('1.0', END)
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100   
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision   : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall      : '+str(r)+"\n")
    text.insert(END,algorithm+' FMeasure    : '+str(f))    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.xticks(rotation=90)
    plt.show()
 
def traineyenet():
    global filename, dataset
    global X, Y
    global X_train, X_test, y_train, y_test
    global X_train, X_val, y_train, y_val,eyenet_model
    #dataset preprocessing such as shuffling and normalization
    text.delete('1.0', END)
    #train eyenet model using SGD optimizer fixed learning rate
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
    #compiling, training and loading the model
    opt = SGD(lr=0.001)
    eyenet_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])#compiling the model
    if os.path.exists("model/sgd_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/sgd_weights.hdf5', verbose = 1, save_best_only = True)
        hist = eyenet_model.fit(X_train, y_train, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/sgd_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        eyenet_model.load_weights("model/sgd_weights.hdf5")
    #perfrom prediction on test data
    predict = eyenet_model.predict(X_val)
    predict = np.argmax(predict, axis=1)
    y_val1 = np.argmax(y_val, axis=1)
    acc = accuracy_score(y_val1, predict) * 100
    print("EyeNet SGD Validation Accuracy  :  "+str(acc))
    predict = eyenet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("EyeNet SGD Testing", predict, y_test1)


def traineyenet1():
    global filename, dataset
    global X, Y
    global X_train, X_test, y_train, y_test
    global X_train, X_val, y_train, y_val,eyenet_model
    #dataset preprocessing such as shuffling and normalization
    text.delete('1.0', END)
    #training EyeNet with Adam Optimizer
    opt = Adam(lr=0.001)#defining Adam
    eyenet_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])#compiling the model
    #compiling, training and loading the model
    if os.path.exists("model/adam_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/adam_weights.hdf5', verbose = 1, save_best_only = True)
        hist = eyenet_model.fit(X_train, y_train, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/adam_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        eyenet_model.load_weights("model/adam_weights.hdf5")
    #perfrom prediction on test data
    predict = eyenet_model.predict(X_val)
    predict = np.argmax(predict, axis=1)
    y_val1 = np.argmax(y_val, axis=1)
    acc = accuracy_score(y_val1, predict) * 100
    print("EyeNet Adam Validation Accuracy  :  "+str(acc))
    predict = eyenet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("EyeNet Adam Testing", predict, y_test1)    

def traineyenet2():
    global filename, dataset,extension_model
    global X, Y
    global X_train, X_test, y_train, y_test
    global X_train, X_val, y_train, y_val
    #dataset preprocessing such as shuffling and normalization
    text.delete('1.0', END)

    #train extension model with padding as 'same and valid'
    extension_model = Sequential()
    extension_model.add(InputLayer(input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])))
    extension_model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
    extension_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    extension_model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    extension_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    extension_model.add(BatchNormalization())
    extension_model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    extension_model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    extension_model.add(BatchNormalization())
    extension_model.add(Flatten())
    extension_model.add(Dense(units=100, activation='relu'))
    extension_model.add(Dense(units=100, activation='relu'))
    extension_model.add(Dropout(0.25))
    extension_model.add(Dense(units=y_train.shape[1], activation='softmax'))
    extension_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    if os.path.exists("model/extension_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
        hist = extension_model.fit(X_train, y_train, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/extension_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        extension_model.load_weights("model/extension_weights.hdf5")
    #perfrom prediction on test data
    predict = extension_model.predict(X_val)
    predict = np.argmax(predict, axis=1)
    y_val1 = np.argmax(y_val, axis=1)
    acc = accuracy_score(y_val1, predict) * 100
    print("Extension with Adam & Valid Padding Validation Accuracy  :  "+str(acc))
    predict = extension_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("Extension with Adam & Valid Padding Testing", predict, y_test1)     

def AccuracyGraph():
    global filename, dataset,labels
    global X, Y
    global X_train, X_test, y_train, y_test
    global X_train1,X_test1,y_train1,y_test1,rf_cls
    #dataset preprocessing such as shuffling and normalization
    text.delete('1.0', END)
    sgd_acc, sgd_loss = values("model/sgd_history.pckl", "accuracy", "loss")
    adam_acc, adam_loss = values("model/adam_history.pckl", "accuracy", "loss")
    extension_acc, extension_loss = values("model/extension_history.pckl", "accuracy", "loss")
        
    plt.figure(figsize=(6,4))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(sgd_acc, 'ro-', color = 'green')
    plt.plot(adam_acc, 'ro-', color = 'blue')
    plt.plot(extension_acc, 'ro-', color = 'black')
    plt.legend(['EyeDeepNet SGD', 'EyeDeepNet Adam', 'Extension with Valid & Same padding'], loc='lower right')
    plt.title('All Algorithm Training Accuracy Graph')
    plt.show()


def graph():
    text.delete('1.0', END)
    #display all algorithm performnace
    algorithms = ['EyeNet with SGD', 'EyeNet with Adam', 'Extension with Adam & Valid']
    data = []
    for i in range(len(accuracy)):
        data.append([algorithms[i], accuracy[i], precision[i], recall[i], fscore[i]])
    data = pd.DataFrame(data, columns=['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'FSCORE'])
    text.insert(END,str(data)+"\n")

    df = pd.DataFrame([['EyeNet with SGD','Accuracy',accuracy[0]],['EyeNet with SGD','Precision',precision[0]],['EyeNet with SGD','Recall',recall[0]],['EyeNet with SGD','FSCORE',fscore[0]],
                       ['EyeNet with Adam','Accuracy',accuracy[1]],['EyeNet with Adam','Precision',precision[1]],['EyeNet with Adam','Recall',recall[1]],['EyeNet with Adam','FSCORE',fscore[1]],
                       ['Extension with Adam & Valid','Accuracy',accuracy[2]],['Extension with Adam & Valid','Precision',precision[2]],['Extension with Adam & Valid','Recall',recall[2]],['Extension with Adam & Valid','FSCORE',fscore[2]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(8, 2))
    plt.title("All Algorithms Performance Graph")
    plt.show()

def values(filename, acc, loss):
    f = open(filename, 'rb')
    train_values = pickle.load(f)
    f.close()
    accuracy_value = train_values[acc][0:20]
    loss_value = train_values[loss][0:20]
    return accuracy_value, loss_value

def predict():
    global labels
    global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
    global accuracy, precision, recall, fscore, values, text,vectorizer
    global extension_model

    text.delete('1.0', END)
    image_path = filedialog.askopenfilename(initialdir='.')
    image = cv2.imread(image_path)#read test image
    img = cv2.resize(image, (32,32))#resize image
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)#convert image as 4 dimension
    img = np.asarray(im2arr)
    img = img.astype('float32')#convert image features as float
    img = img/255 #normalized image
    predict = extension_model.predict(img)#perform prediction on test image
    predict = np.argmax(predict)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (400,300))#display image with predicted output
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.putText(img, 'Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    plt.figure(figsize=(4,3))
    plt.imshow(img)
    plt.show()
    
        


font = ('times', 14, 'bold')
title = Label(main, text='EyeDeep-Net: a multi-class diagnosis of retinal diseases using deep neural network')
title.config(bg='dim gray', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", bg='dim gray', fg='white',command=UploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)

font1 = ('times', 13, 'bold')
preprocessButton = Button(main, text="Preprocess Dataset", bg='dim gray', fg='white',command=PreprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

font1 = ('times', 13, 'bold')
RunCNNButton = Button(main, text="EyeNet with SGD", bg='dim gray', fg='white',command=traineyenet)
RunCNNButton.place(x=50,y=200)
RunCNNButton.config(font=font1)

font1 = ('times', 13, 'bold')
RunCNNButton = Button(main, text="EyeNet with Adam", bg='dim gray', fg='white',command=traineyenet1)
RunCNNButton.place(x=50,y=250)
RunCNNButton.config(font=font1)

font1 = ('times', 13, 'bold')
RunExtensionButton = Button(main, text="Train with Adam & Valid Padding", bg='dim gray', fg='white',command=traineyenet2)
RunExtensionButton.place(x=50,y=300)
RunExtensionButton.config(font=font1)


font1 = ('times', 13, 'bold')
PButton = Button(main, text="Comparision Graph", bg='dim gray', fg='white',command=graph)
PButton.place(x=50,y=350)
PButton.config(font=font1)

font1 = ('times', 13, 'bold')
RunGraphButton = Button(main, text="Accuracy & Loss Graph", bg='dim gray', fg='white',command=AccuracyGraph)
RunGraphButton.place(x=50,y=400)
RunGraphButton.config(font=font1)

font1 = ('times', 13, 'bold')
PdButton = Button(main, text="Predict from Test", bg='dim gray', fg='white',command=predict)
PdButton.place(x=50,y=450)
PdButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=118)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=330,y=100)
text.config(font=font1)

main.config(bg='Maroon1')
main.mainloop()


