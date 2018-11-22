from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.models import Sequential
from keras.layers import Convolution2D, BatchNormalization, Activation, AveragePooling2D, Dropout, GlobalAveragePooling2D

import pandas as pd
import numpy as np
import imageio
import os
from skimage import transform
from skimage.color import rgb2gray
from sklearn.preprocessing import label_binarize

from constants import emotions

class DataManager(object):
	"""docstring for DataManager"""
	def __init__(self, dataset_path, dataset_meta_path, image_size=(48, 48)):
		super(DataManager, self).__init__()
		self.dataset_path = dataset_path
		self.dataset_meta_path = dataset_meta_path
		self.image_size = image_size
		if self.dataset_path is not None:
			self.dataset_path = dataset_path
		else:
			raise Exception('Incorrect dataset path')

	def get_data(self):
		data = self._load_data()
		print("----------Data---------")
		print(data.head(5))
		return data['image'], data['emotion']

	def _load_data(self):
		all_df = pd.DataFrame(columns=['image', 'emotion'])
		data_meta = pd.read_csv(self.dataset_meta_path)
		data_meta = self._process_meta(data_meta)
		img_counter = 0
		for image in os.listdir(self.dataset_path):
			try:
				im = imageio.imread(str(os.path.join(self.dataset_path, image)))
			except:
				print("could not read image {}".format(image))
				continue
			im = self._process_image(im)
			if image in data_meta['image'].unique():
				s = {'image': im, 'emotion': data_meta.loc[data_meta['image'] == image]['emotion_vec'].iloc[0]}
				all_df = all_df.append(s, ignore_index=True)
				img_counter += 1
				if img_counter % 1000 == 0:
					print("{} images done".format(img_counter))
		return all_df


	def _process_image(self, image):
		shape = image.shape
		image = rgb2gray(image)
		image = transform.resize(image, self.image_size, mode='symmetric', preserve_range=True)
		return image

	def _process_meta(self, meta):
		meta['emotion'] = meta.apply(lambda x: x['emotion'].lower(), axis=1)
		meta['emotion_vec'] = meta.apply(lambda x: label_binarize([x['emotion']], classes=emotions), axis=1)
		print("----Image Meta-----")
		print(meta.head(5))
		return meta


def simple_CNN(input_shape, num_classes):
	model = Sequential()
	model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same', name='image_array', input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.5))

	model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
	model.add(BatchNormalization())
	model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.5))

	model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.5))

	model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.5))

	model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
	model.add(GlobalAveragePooling2D())
	model.add(Activation('softmax', name='predictions'))
	return model

def preprocess_input(x):
    x = x / 255.
    return x

def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data


#parameters
batch_size = 32
num_epochs = 10000
input_shape = (64, 64, 1)
validation_split = 0.2
verbose = 1
num_classes = len(emotions)
patience = 50
base_path = './trained_models/emotion_models'

#data generator
data_generator = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)

#model paramteres/compilation
# model = mini_XCEPTION(input_shape, num_classes)
model = simple_CNN(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


#training
dataset_name = 'facial_expression_dataset'
dataset_path = 'images'
dataset_meta_path = 'legend.csv'
print("Training dataset:", dataset_name)
log_file_path = base_path + dataset_name + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)
trained_models_path = base_path + dataset_name + '_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

#loading_dataset
data_loader = DataManager(dataset_path=dataset_path, dataset_meta_path=dataset_meta_path, image_size=input_shape[:2])
faces, emotions = data_loader.get_data()
faces = preprocess_input(faces)
faces = np.array(faces.tolist()).reshape(faces.shape[0], input_shape[0], input_shape[1], input_shape[2])
emotions = np.array(emotions.tolist()).reshape(emotions.shape[0], num_classes)
print("faces shape {}, emotions shape {}".format(faces.shape, emotions.shape))
num_samples, num_classes = emotions.shape
train_data, val_data = split_data(faces, emotions, validation_split)
train_faces, train_emotions = train_data
# model.fit(data_generator.flow(train_faces, train_emotions, batch_size), steps_per_epoch=len(train_faces)/batch_size, epochs=num_epochs, verbose=1, callbacks=callbacks, validation_data=val_data)

model.fit(train_faces, train_emotions, batch_size=batch_size, epochs=num_epochs, validation_data=val_data, shuffle=True)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])