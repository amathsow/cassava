
#%%
import matplotlib.pyplot as plt
from EgoCNN import EgoCNN, EgoDense, EgoMobile, EgoResNet
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from keras.optimizers import SGD, Adam

import os
import numpy as np
import h5py


#%%
hdf5_path = 'extraimages_224.hdf5'
f = h5py.File(hdf5_path)


#%%
ego_mobile = EgoMobile(in_size=(224,224,3),
                       xrange=(-3,3),
                       yrange=(-3,3),
                       out_size=(20,7,7),
                       batch_size=32, tbins=20)
model = ego_mobile.model


#%%
ego_mobile.batch_size = 32
ego_mobile.lscale = 7


#%%
# callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
cp_filepath = '~/cassava/model_data/egomob.hdf5'
checkpointer = ModelCheckpoint(cp_filepath, monitor='val_loss', verbose=0,
                               save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=15, min_lr=0.0001)


#%%
model.compile(optimizer=Adam(lr=0.001),
              loss={'output0': 'categorical_crossentropy',
                    'output1': 'categorical_crossentropy',
                    'output2': 'categorical_crossentropy'},
              metrics=['acc'])
model.summary()


#%%
# generators
train_generator = ego_mobile.batch_generator(f['train_img'])
val_generator = ego_mobile.batch_generator(f['val_img'])


#%%
if os.path.isfile(cp_filepath):
    print('Loading model')
    model = load_model(cp_filepath)
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=1,
                                  epochs=1,
                                  validation_data=val_generator,
                                  validation_steps=1,
                                  callbacks=[early_stop, checkpointer, reduce_lr])
else:
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=1,
                                  epochs=1,
                                  validation_data=val_generator,
                                  validation_steps=1,
                                  callbacks=[early_stop, checkpointer, reduce_lr])


#%%
mets = ['output0_loss', 'output1_loss', 'output2_loss', 'val_output0_loss', 'val_output1_loss', 'val_output2_loss' ]
plt.figure(figsize=(15,7))
for met in mets:
    plt.plot(history.history[met])

plt.legend(mets)


#%%



