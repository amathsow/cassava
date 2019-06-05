
#%%
import matplotlib.pyplot as plt
from EgoCNN import EgoCNN, EgoDense, EgoMobile, EgoResNet
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model, Model
from keras.optimizers import SGD, Adam

import numpy as np
import h5py
import os


#%%
from keras.layers import Input, Conv2D, Dense, Dropout, Flatten, BatchNormalization
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator


#%%
train_path = 'train'
valid_path = 'validation'

#%%
model = load_model('model_data/egomob.hdf5')

#%%
model.summary()


#%%
def classifier(model):
    input_img = Input(shape=(224,224,3))
    x = model.get_layer('mobilenetv2_1.00_224')(input_img)
    x = Conv2D(128, kernel_size=(2,2), activation='elu',
               kernel_regularizer=l2(0.0001),
               bias_regularizer=l2(0.0001))(x)
    x = Conv2D(128, kernel_size=(1,1), activation='elu',
               kernel_regularizer=l2(0.0001),
               bias_regularizer=l2(0.0001))(x)
    x = Conv2D(64, kernel_size=(1,1), activation='elu',
               kernel_regularizer=l2(0.0001),
               bias_regularizer=l2(0.0001))(x)
    x = Flatten()(x)
    x = Dropout(rate=.3)(x)
    x = Dense(200, activation='elu')(x)
    x = Dense(5, activation='softmax')(x)     
    return Model(input_img, x)


#%%
model = classifier(model)


#%%
model.summary()


#%%
# callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
cp_filepath = 'model_data/egoclassifier_mobile.hdf5'
checkpointer = ModelCheckpoint(cp_filepath, monitor='val_loss', verbose=0,
                               save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001)


#%%
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['acc'])


#%%
batch_size = 32

# generators
train_generator = ImageDataGenerator(rotation_range=10,
                                     width_shift_range=10,
                                     height_shift_range=10,
                                     fill_mode='reflect',
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     brightness_range=[0.5, 1.5],
                                     channel_shift_range = 0.1,
                                     zoom_range=[0.9, 1.25],
                                     shear_range=0.01).flow_from_directory('gdrive/My Drive/cassava/train',
                                                                             target_size=(224, 224),
                                                                             classes=['cbb','cbsd','cgm','cmd','healthy'],
                                                                             class_mode='categorical',
                                                                             batch_size=batch_size,
                                                                             shuffle=True)
valid_generator = ImageDataGenerator(horizontal_flip=True,
                                     vertical_flip=True).flow_from_directory('gdrive/My Drive/cassava/validation',
                                                                             target_size=(224, 224),
                                                                             classes=['cbb','cbsd','cgm','cmd','healthy'],
                                                                             class_mode='categorical',
                                                                             batch_size=batch_size,
                                                                             shuffle=True)


#%%
if os.path.isfile(cp_filepath):
    print('Loading model ' + cp_filepath)
    model = load_model(cp_filepath)
    history = model.fit_generator(train_generator,
                                  epochs=100,
                                  steps_per_epoch=3958//batch_size,
                                  validation_data=valid_generator,
                                  validation_steps=1698//batch_size,
                                  callbacks=[early_stop, checkpointer, reduce_lr])
else:
    history = model.fit_generator(train_generator,
                                  epochs=100,
                                  steps_per_epoch=3958//batch_size,
                                  validation_data=valid_generator,
                                  validation_steps=1698//batch_size, 
                                  #class_weight={0: 5.7, 1: 1.84, 2: 3.43, 3: 1.0, 4: 8.41},
                                  callbacks=[early_stop, checkpointer, reduce_lr])


#%%
metrics = ['acc', 'val_acc']
plt.figure(figsize=(10,10))
for met in metrics:
    plt.plot(history.history[met])
plt.legend(metrics)
