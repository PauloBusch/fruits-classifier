from os import system
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import models, layers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

# parameters
epochs = 500
learning_rate = 0.001
batch_size = 32
dropout = 0.5

num_classes = 3
image_size=[150, 150]
train_dir='./dataset/train'
test_dir='./dataset/test'

# load imagenet model pretrain
base_model=InceptionResNetV2(
  weights='imagenet',
  include_top=False,
  
  classes=num_classes
)

# apply GlobalMaxPooling
pooling=GlobalAveragePooling2D()(base_model.output)

# add new layer with num_classes neurons
vector=layers.Dense(num_classes, activation='softmax')(pooling)

# apply dropout
# vector=Dropout(dropout, noise_shape=None, seed=None)(vector)

# create model with new layer
model=models.Model(inputs=base_model.input, outputs=vector)

# freeze imagenet loaded layers
for layer in model.layers:
  layer.trainable=layer.name == 'dense'

# load images with InceptionResNetV2 function pre processor
train_data_gen=ImageDataGenerator(preprocessing_function=preprocess_input)
test_data_gen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=train_data_gen.flow_from_directory(
  train_dir,
  target_size=image_size,
  batch_size=batch_size,
  shuffle=False
)
test_generator=test_data_gen.flow_from_directory(
  test_dir,
  target_size=image_size,
  batch_size=batch_size,
  shuffle=False
)

# compile the model with SGD optimizer
model.compile(
  optimizer=SGD(learning_rate=learning_rate),
  loss=CategoricalCrossentropy(),
  metrics=['accuracy']
)

# define size steps
step_size_train=train_generator.n//train_generator.batch_size
step_size_test=test_generator.n//test_generator.batch_size

# train and test model
system('cls')
with tf.device("/device:CPU:0"):
  history=model.fit(
    train_generator,
    steps_per_epoch=step_size_train,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=step_size_test
  )

# plot Loss
plt.title('Training and validation loss')
plt.plot(history.history['loss'], label='Training loss', color='red')
plt.plot(history.history['val_loss'], label='Validation loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plot Accuracy
plt.subplots(1, 1)
plt.title('Training and validation accuracy')
plt.plot(history.history['accuracy'], label='Training accuracy', color='red')
plt.plot(history.history['val_accuracy'], label='Validation accuracy', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# run prediction
classes=test_generator.class_indices.keys()
predicted = model.predict(test_generator)
predicted = np.argmax(predicted, axis=1)

# plot confusion matrix
dataFrame=pd.DataFrame(
  confusion_matrix(test_generator.classes, predicted), 
  columns=classes, 
  index=classes
)
plt.subplots(1, 1)
sns.heatmap(dataFrame, annot=True)

# print classification report
system('cls')
classification = classification_report(test_generator.classes, predicted)
print(classification_report(test_generator.classes, predicted))

# evaluate the test accuracy and test loss of the model
loss, accuracy=model.evaluate(test_generator)
print(f'Accuracy: {accuracy:.3}')
print(f'Loss: {loss:.3}')

plt.show(block=True)
print('');