import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

IMG_SIZE = (112, 112)
BATCH = 16

train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True
)

train = train_gen.flow_from_directory(
    "dataset3",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    subset='training'
)

val = train_gen.flow_from_directory(
    "dataset3",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    subset='validation'
)

base = ResNet50V2(weights='imagenet', include_top=False, input_shape=(112,112,3))

x = Flatten()(base.output)
x = Dense(1000, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(2, activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train, validation_data=val, epochs=10)

model.save("models/drowsy_model3.h5")