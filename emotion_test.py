import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
import random
import cv2

import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Input
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l1, l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import regularizers
from keras.utils import img_to_array



def data_read():
    # loading and class renaming
    train_dir = '/Users/arian/Documents/Uni/Done & Gone/Project/archive/train/'
    test_dir = '/Users/arian/Documents/Uni/Done & Gone/Project/archive/test/'

    train_datagen = ImageDataGenerator( rescale=1./255,
                                        rotation_range = 10,
                                        horizontal_flip = True,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        fill_mode = 'nearest')

    training_set = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = batch_size,
                                                    target_size = (image_size, image_size),
                                                    shuffle = True,
                                                    color_mode = 'grayscale',
                                                    class_mode = 'categorical')

    test_datagen = ImageDataGenerator(rescale=1./255)
    testing_set = test_datagen.flow_from_directory(test_dir,
                                                    batch_size=batch_size,
                                                    target_size=(image_size,image_size),
                                                    shuffle=True,
                                                    color_mode='grayscale',
                                                    class_mode='categorical')

    return (training_set, testing_set)

def cnn_model():
    # first input model
    visible = Input(shape=(image_size, image_size, 1), name='input')
    num_classes = 7
    #the 1-st block
    conv1_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_1')(visible)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_2')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(2,2), name = 'pool1_1')(conv1_2)
    drop1_1 = Dropout(0.3, name = 'drop1_1')(pool1_1)#the 2-nd block
    conv2_1 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_1')(drop1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_2')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_3 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_3')(conv2_2)
    conv2_2 = BatchNormalization()(conv2_3)
    pool2_1 = MaxPooling2D(pool_size=(2,2), name = 'pool2_1')(conv2_3)
    drop2_1 = Dropout(0.3, name = 'drop2_1')(pool2_1)#the 3-rd block
    conv3_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_1')(drop2_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_2')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_3')(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_4')(conv3_3)
    conv3_4 = BatchNormalization()(conv3_4)
    pool3_1 = MaxPooling2D(pool_size=(2,2), name = 'pool3_1')(conv3_4)
    drop3_1 = Dropout(0.3, name = 'drop3_1')(pool3_1)#the 4-th block
    conv4_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_1')(drop3_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_2')(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_3')(conv4_2)
    conv4_3 = BatchNormalization()(conv4_3)
    conv4_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_4')(conv4_3)
    conv4_4 = BatchNormalization()(conv4_4)
    pool4_1 = MaxPooling2D(pool_size=(2,2), name = 'pool4_1')(conv4_4)
    drop4_1 = Dropout(0.3, name = 'drop4_1')(pool4_1)

    #the 5-th block
    conv5_1 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_1')(drop4_1)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_2 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_2')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_3 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_3')(conv5_2)
    conv5_3 = BatchNormalization()(conv5_3)
    conv5_4 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_4')(conv5_3)
    conv5_3 = BatchNormalization()(conv5_3)
    pool5_1 = MaxPooling2D(pool_size=(2,2), name = 'pool5_1')(conv5_4)
    drop5_1 = Dropout(0.3, name = 'drop5_1')(pool5_1)#Flatten and output
    flatten = Flatten(name = 'flatten')(drop5_1)
    ouput = Dense(num_classes, activation='softmax', name = 'output')(flatten)# create model
    model = Model(inputs =visible, outputs = ouput)

    optimizer = Adam(lr = 0.0001, decay = 1e-6)
    model.compile(optimizer = optimizer,
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    return model

def evaluate_func(result):
    fig , ax = plt.subplots(1,2)
    train_acc = result.history['accuracy']
    train_loss = result.history['loss']
    fig.set_size_inches(12,4)

    ax[0].plot(result.history['accuracy'])
    ax[0].plot(result.history['val_accuracy'])
    ax[0].set_title('Training Accuracy vs Validation Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='upper left')

    ax[1].plot(result.history['loss'])
    ax[1].plot(result.history['val_loss'])
    ax[1].set_title('Training Loss vs Validation Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper left')

    plt.show()

def train():
    # creating the model and compiling it
    model = cnn_model()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=6,
                                  verbose=1,
                                  min_delta=0.0001)

    checkpoint = ModelCheckpoint(filepath=chk_path,
                                 save_best_only=True,
                                 verbose=1,
                                 mode='min',
                                 moniter='val_loss')

    callbacks = [checkpoint, reduce_lr]

    result = model.fit(x=training_set,
             validation_data=testing_set,
             steps_per_epoch=num_train/batch_size,
             verbose=1,
             callbacks=callbacks,
             epochs=epochs)

    # evaluating the model
    evaluate_func(result)
    # save Q
    s = input("save Model?(Y/N) ")
    if s == 'Y':
        model.save('cnn_emotion')

def show_result():
    model = load_model("cnn_emotion")

    plt.figure(figsize=(20,5))
    i = 1
    for expression in os.listdir(test_dir):
        if expression[0] != '.':
            img = image.imread((test_dir + expression + '/' + os.listdir(test_dir + expression)[random.randint(0, 111)]))
            plt.subplot(2,7,i)
            plt.imshow(img)
            plt.title(expression)
            plt.axis('off')
            imag = img
            img = np.expand_dims(img, axis = 0)
            img = np.expand_dims(img, axis = -1)
            p = model.predict(img)
            p = p[0]
            pred_label = np.argsort(-p)[:3]
            pred_prob = [p[l] for l in pred_label]
            pred_label = [class_names[l] for l in pred_label]
            plt.subplot(2, 7, i+7)
            plt.bar(range(3), pred_prob)
            plt.xticks(range(3), pred_label)
            i += 1
    plt.show()


train_dir = '/Users/arian/Documents/Uni/Done & Gone/Project/archive/train/'
test_dir = '/Users/arian/Documents/Uni/Done & Gone/Project/archive/test/'
image_size = 48 # x, y
num_classes = 7
num_train = 28709
epochs = 100
batch_size = 256
chk_path = 'check'
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral', 'Surprise']

training_set, testing_set = data_read()
#train()
#show_result()

model = load_model("cnn_emotion")
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
while cap.isOpened():
    res,frame=cap.read()
    height, width , channel = frame.shape
    #---------------------------------------------------------------------------
    # Creating an Overlay window to write prediction and cofidence
    sub_img = frame[0:int(height/6),0:int(width)]
    black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0
    res = cv2.addWeighted(sub_img, 0.77, black_rect,0.23, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.8
    FONT_THICKNESS = 2
    lable_color = (10, 10, 255)
    lable = "Emotion Detection"
    lable_dimension = cv2.getTextSize(lable,FONT ,FONT_SCALE,FONT_THICKNESS)[0]
    textX = int((res.shape[1] - lable_dimension[0]) / 2)
    textY = int((res.shape[0] + lable_dimension[1]) / 2)
    cv2.putText(res, lable, (textX,textY), FONT, FONT_SCALE, (0,0,0), FONT_THICKNESS)
    # prediction part --------------------------------------------------------------------------
    gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_image )
    try:
        for (x,y, w, h) in faces:
            cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  2)
            roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
            roi_gray=cv2.resize(roi_gray,(48,48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis = 0)
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            cv2.putText(res, "Sentiment: {}".format(emotion_prediction), (0,textY+22+5), FONT,0.7, lable_color,2)
            lable_violation = 'Confidence: {}'.format(str(np.round(np.max(predictions[0])*100,1))+ "%")
            violation_text_dimension = cv2.getTextSize(lable_violation,FONT,FONT_SCALE,FONT_THICKNESS )[0]
            violation_x_axis = int(res.shape[1]- violation_text_dimension[0])
            cv2.putText(res, lable_violation, (violation_x_axis,textY+22+5), FONT,0.7, lable_color,2)
    except :
        pass
    frame[0:int(height/6),0:int(width)] = res
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        breakcap.release()
cv2.destroyAllWindows

# predecting new data
#y_test = np.argmax(y_test, axis=1)
#plt.figure(figsize=(10, 20))
