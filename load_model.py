import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.models import Sequential
import pathlib

#loading trained model
new_model = tf.keras.models.load_model('my_model.h5')

#finding validation data for evaluation
#get path of file
path = None #input path
data_dir = tf.keras.utils.get_file("defi1certif-datasets-fire_medium", path, archive_format='tar', untar=True)

batch_size = 32
img_height = 180
img_width = 180


val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#accuracy on validation data
loss, acc = new_model.evaluate(val_ds, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
