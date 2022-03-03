from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.metrics import roc_auc_score

img_dims = 64
batch_size = 32

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (img_dims, img_dims, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

input_path = 'chest_xray/'
train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(directory=input_path+'train',
                                            target_size = (img_dims, img_dims),
                                            batch_size = batch_size,
                                            class_mode = 'binary')
test_set = test_datagen.flow_from_directory(directory=input_path+'test',
                                            target_size = (img_dims, img_dims),
                                            batch_size = batch_size,
                                            class_mode = 'binary')

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

epochs = 10

hist = classifier.fit_generator(
           training_set, steps_per_epoch=training_set.samples // batch_size, 
           epochs=epochs, validation_data=test_set, 
           validation_steps= test_set.samples)

img = plt.imread('chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg')
img = cv2.resize(img, (img_dims, img_dims))
img = np.dstack([img, img, img])
img = img.astype('float32') / 255
result = classifier.predict(np.expand_dims(image.img_to_array(img), axis = 0))  

if result[0][0] > 0.5:
  prediction = 'Pnuemonia'
else:
  prediction = 'Normal'
  
print(prediction)


test_data = []
test_labels = []
for cond in ['/NORMAL/', '/PNEUMONIA/']:
        for img in (os.listdir(input_path + 'test' + cond)):
            img = plt.imread(input_path+'test'+cond+img)
            img = cv2.resize(img, (img_dims, img_dims))
            img = np.dstack([img, img, img])
            img = img.astype('float32') / 255
            if cond=='/NORMAL/':
                label = 0
            elif cond=='/PNEUMONIA/':
                label = 1
            test_data.append(img)
            test_labels.append(label)
        
test_data = np.array(test_data)
test_labels = np.array(test_labels)
from sklearn.metrics import accuracy_score, confusion_matrix

preds = classifier.predict(test_data)

acc = accuracy_score(test_labels, np.round(preds))*100
cm = confusion_matrix(test_labels, np.round(preds))

print('CONFUSION MATRIX')
print(cm)

print('\nTEST METRICS')
print('Test Accuracy: {}%'.format(acc))

print('\nTRAIN METRIC')
print('Train Accuracy: {}'.format(np.round((hist.history['accuracy'][-1])*100, 2)))


classifier.save('Assignment 2-pnuemonia-CNN.model')

predict_dir_path='chest_xray/images/'
onlyfiles = [f for f in os.listdir(predict_dir_path) if os.path.isfile(os.path.join(predict_dir_path, f))]

from keras.preprocessing import image
normal_counter = 0 
viral_pneumonia_counter  = 0
for file in onlyfiles:
    img = image.load_img(predict_dir_path+file, target_size=(img_dims, img_dims))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = classifier.predict_classes(images, batch_size=10)
    classes = classes[0][0]
    
    if classes == 0:
        print(file + ": " + 'Normal Pneumonia')
        normal_counter += 1
    else:
        print(file + ": " + 'Viral Pneumonia')
        viral_pneumonia_counter += 1
print("Total Normal Pneumonia:",normal_counter)
print("Total Viral Pneumonia :",viral_pneumonia_counter)

roc_auc_score(test_labels, preds[:,:])


p1 = (cm[0,0] + cm[1,0]) / (cm[0,0]+cm[0,1]+cm[1,0]+cm[0,0])
p2 = (cm[0,0] + cm[0,1]) / (cm[0,0]+cm[0,1]+cm[1,0]+cm[0,0])

randacc = p1*p2 + (1-p1)*(1-p2)

kappa = (acc-randacc)/(1-randacc)

print(kappa)