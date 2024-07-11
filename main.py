from models import create_cnn_model
from utils import get_data_generator,multi_output_generator,plot_history
import os
import tensorflow as tf
import gc

real_data_dir = 'C:/Users/akash/webdev/signature_verification/dataset1/real'
forge_data_dir = 'C:/Users/akash/webdev/signature_verification/dataset1/forge'

input_shape = (128,128,1)
num_individuals = len(os.listdir(real_data_dir))
batch_size = 32
epochs = 30

train_real = get_data_generator(real_data_dir,batch_size = batch_size)
train_forged = get_data_generator(forge_data_dir,batch_size = batch_size)
val_real = get_data_generator(real_data_dir, batch_size=batch_size, augment=False)  # No data augmentation
val_forged = get_data_generator(forge_data_dir, batch_size=batch_size, augment=False)  # No data augmentation



train_generators = multi_output_generator(train_real,train_forged)
val_generators = multi_output_generator(val_real,val_forged)

model = create_cnn_model(input_shape,num_individuals)

history = model.fit(
          train_generators,
          steps_per_epoch = 200,
          epochs = epochs,
          validation_data = val_generators,
          validation_steps = 10,
          verbose = 1
          
)
model_save_path = 'C:/Users/akash/webdev/signature_verification/saved_model/my_model.h5'
model.save(model_save_path,include_optimizer = True)
import pickle

# Example using pickle
with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
del model
gc.collect()
tf.keras.backend.clear_session()
