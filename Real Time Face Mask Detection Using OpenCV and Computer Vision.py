#%%
#import dependencies
import cv2 as cv
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

#%%
#access pretrain model
pretrained = cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_frontalface_default.xml')

#%%
#capture video from computer camera
capture = cv.VideoCapture(0)

data = []

while True:
    #keep capture the video
    ret, img = capture.read()
    
    if ret:
        #when something is captured, the pretrained model use to detect the face
        faces = pretrained.detectMultiScale(img)
        for (x,y,w,h) in faces:
            #draw rectangle
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
            #extract the face from the image
            face = img[y:y+h,x:x+w,:]
            face = cv.resize(face,(50,50))
            print((len(data)))
            if len(data)<400:
                data.append(face)
        cv.imshow('Window',img)
        if cv.waitKey(2)==27 or len(data)>=200:
            break

capture.release()
cv.destroyAllWindows()

#%%
np.save("mask.npy",data)

# %%
plt.figure(figsize=(5,5))
plt.imshow(data[0])
plt.axis(False)
plt.show()

# %%
capture = capture = cv.VideoCapture(0)

data = []

while True:
    #keep capture the video
    ret, img = capture.read()
    
    if ret:
        #when something is captured, the pretrained model use to detect the face
        faces = pretrained.detectMultiScale(img)
        for (x,y,w,h) in faces:
            #draw rectangle
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
            #extract the face from the image
            face = img[y:y+h,x:x+w,:]
            face = cv.resize(face,(50,50))
            print((len(data)))
            if len(data)<400:
                data.append(face)
        cv.imshow('Window',img)
        if cv.waitKey(2)==27 or len(data)>=200:
            break

capture.release()
cv.destroyAllWindows()

# %%
np.save('no_mask.npy',data)

# %%
plt.figure(figsize=(5,5))
plt.imshow(data[0])
plt.axis(False)
plt.show()

# %%
mask = np.load('mask.npy')
no_mask = np.load('no_mask.npy')

# %%
print(mask.shape)
print(no_mask.shape)

# %%
#convert 4D data to 2D
mask = mask.reshape(200,50*50*3)
no_mask = no_mask.reshape(200,50*50*3)

print(mask.shape)
print(no_mask.shape)

# %%
X = np.concatenate((no_mask,mask),axis=0)
print(X.shape)

# %%
Y = np.zeros(X.shape[0])
Y.shape

# %%
#0 to 199: no mask(0)
#other with  mask
Y[200:]=1.0

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

# %%
print(len(X_train))
print(len(X_test))
print(len(Y_train))
print(len(Y_test))

# %%
#7500 will slow the ML process, so dimensionality reduction
pca = PCA(n_components=3)
X_train=pca.fit_transform(X_train)
X_test=pca.fit_transform(X_test)
print(X_train.shape)
print(X_test.shape)

# %%
print(X_train[0])

# %%
model = SVC()
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

# %%
Y_pred = model.predict(X_test)
Y_pred
# %%
Y_test
# %%
accuracy_score(Y_test,Y_pred)
# %%
print(classification_report(Y_test,Y_pred))
# %%
cm = confusion_matrix(Y_test,Y_pred)
plt.figure(figsize=(4,3))
g = sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
g.set_xticklabels(labels=['No Mask(0)','Mask(1)'],rotation=30)
g.set_yticklabels(labels=['No Mask(0)','Mask(1)'],rotation=30)
plt.ylabel('True Label',fontsize=14)
plt.xlabel('Predicted Label',fontsize=14)
plt.title('Confusion Matrix',fontsize=16)
plt.show
# %%
capture = cv.VideoCapture(0)
data = []

names = {
    0:'No Mask',
    1:'Mask'}

font = cv.FONT_HERSHEY_COMPLEX

while True:
    #keep capture the video
    ret, img = capture.read()
    
    if ret:
        #when something is captured, the pretrained model use to detect the face
        faces = pretrained.detectMultiScale(img)
        for (x,y,w,h) in faces:
            #draw rectangle
            cv.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            #extract the face from the image
            face = img[y:y+h,x:x+w,:]
            face = cv.resize(face,(50,50))

            face = face.reshape(1,-1)
            face = pca.transform(face)
            pred = model.predict(face)
            n = names[int(pred)]
            cv.putText(img,n,(x,y),font,1,(244,250,250),2)
    
    cv.imshow('Window',img)
    if cv.waitKey(2)==27:
        break

capture.release()
cv.destroyAllWindows()



# %%
