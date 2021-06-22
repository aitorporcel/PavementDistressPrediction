# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 13:15:16 2021

@author: Aitor
"""
#Import the dataset----------------------------------------------------
import os
train_dir = os.path.join('./distress_recognition/')
test_dir = os.path.join('./distress_recognition_test/')


#Image augmentation function-------------------------------------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def image_gen_w_aug(train_parent_directory, test_parent_directory):
    
    train_datagen = ImageDataGenerator(rescale=1/255,
                                      rotation_range = 30,  
                                      zoom_range = 0.2, 
                                      width_shift_range=0.1,  
                                      height_shift_range=0.1,
                                      validation_split = 0.15)
    
  
    
    test_datagen = ImageDataGenerator(rescale=1/255)
    
    train_generator =          train_datagen.flow_from_directory(train_parent_directory,
                                  target_size = (75,75),
                                  batch_size = 40,
                                  class_mode = 'categorical',
                                  subset='training')
    
    val_generator = train_datagen.flow_from_directory(train_parent_directory,
                                  target_size = (75,75),
                                  batch_size = 10,
                                  class_mode = 'categorical',
                                  subset = 'validation')
    
    test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                 target_size=(75,75),
                                 batch_size = 10,
                                 class_mode = 'categorical')
    return train_generator, val_generator, test_generator

#Split the sets---------------------------------------------------------------
train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, test_dir)


#Inception-------------------------------------------------------------------
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.models import save_model
from tensorflow import keras

def model_output_for_TL (pre_trained_model, last_output):    
    x = Flatten()(last_output)
    
    # Dense hidden layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    
    # Output neuron. 
    x = Dense(4, activation='softmax')(x)
    
    model = Model(pre_trained_model.input, x)
    
    return model
pre_trained_model = InceptionV3(input_shape = (75, 75, 3), 
                                include_top = False, 
                                weights = 'imagenet')
for layer in pre_trained_model.layers:
  layer.trainable = False
last_layer = pre_trained_model.get_layer('mixed5')
last_output = last_layer.output
model_TL = model_output_for_TL(pre_trained_model, last_output)


#train the model--------------------------------------------------------------
model_TL.compile(optimizer=keras.optimizers.Adam(learning_rate=0.5e-4),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

history = model_TL.fit(
      train_generator,
      steps_per_epoch=2,  
      epochs=30,
      verbose=1,
      validation_data = validation_generator)

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#save the model
save_model(model_TL,'my_model.hdf5')

#Evaluate the data in the test set
print("Evaluate on test data")
results = model_TL.evaluate(test_generator, batch_size=10)
print("test loss, test acc:", results)