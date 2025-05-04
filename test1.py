import pandas as pd
import numpy as np
import cv2

dataset = pd.read_csv("Dataset/RFMiD_Training_Labels.csv", usecols=['ID', 'Disease_Risk', 'DR', 'MH', 'ODC'])
names = dataset['ID'].ravel()
normal = dataset['Disease_Risk'].ravel()
dr = dataset['DR'].ravel()
mh = dataset['MH'].ravel()
odc = dataset['ODC'].ravel()

for i in range(len(normal)):
    if normal[i] == 0:
        img = cv2.imread("Dataset/images/"+str(names[i])+".png")
        cv2.imwrite("SelectedImages/Normal/"+str(names[i])+".png", img)
        print("normal")

for i in range(len(dr)):
    if dr[i] == 1:
        img = cv2.imread("Dataset/images/"+str(names[i])+".png")
        cv2.imwrite("SelectedImages/DR/"+str(names[i])+".png", img)
        print("dr")

for i in range(len(mh)):
    if mh[i] == 1:
        img = cv2.imread("Dataset/images/"+str(names[i])+".png")
        cv2.imwrite("SelectedImages/MH/"+str(names[i])+".png", img)
        print("mh")

for i in range(len(odc)):
    if odc[i] == 1:
        img = cv2.imread("Dataset/images/"+str(names[i])+".png")
        cv2.imwrite("SelectedImages/ODC/"+str(names[i])+".png", img)
        print("odc")
