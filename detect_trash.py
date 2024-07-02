import os 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam  

base_dir = './dataset-resized'
train_datagen = ImageDataGenerator(rescale=1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, validation_split = 0.2)
train_generator = train_datagen.flow_from_directory(base_dir, target_size = (224,224), batch_size = 32, class_mode = 'categorical', subset = 'training')
validation_generator = train_datagen.flow_from_directory(base_dir, target_size = (224,224), batch_size = 32, class_mode = 'categorical', subset = 'validation')

base_model = ResNet50(weights = 'imagenet', include_top = False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(train_generator.num_classes, activation = 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

for layer in base_model.layers: 
    layer.trainable = False 

model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
                            




