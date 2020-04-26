from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

# Layer 1
model.add(Conv2D(32, 3, 3, border_mode="same", input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Conv2D(64, 2, 2, border_mode="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_datagen = datagen.flow_from_directory('./img_data', target_size=(150, 150), batch_size=16)

model.fit_generator(train_datagen, steps_per_epoch=100, epochs=1)

model.save_weights('flower-weight.h5')
