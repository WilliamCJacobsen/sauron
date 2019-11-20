from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#Keras imports
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout,Flatten
import tensorflow as tf
#misc imports 
import numpy as np
import time
#local imports
from Face_trainer import Face_trainer

class ConvolutionNN:
    def __init__(self, cv, cascade, image_size: int, output = 4, epochs : int = 400):
        self.output = output
        self.epochs = epochs
        self.image_size = image_size
        self.model = self.get_model()
        self.cv = cv
        self.cascade = cascade
        self.ft = Face_trainer(cv=self.cv,image_size = image_size, cascade = self.cascade)
        self.generator = ImageDataGenerator(vertical_flip=True,
        horizontal_flip=False, 
        width_shift_range=0.125, 
        height_shift_range=0.125, 
        fill_mode="constant",
        cval=0,
        rotation_range=20)



    def get_model(self):
        try:
            print("returning model...")
            return load_model("my_model1574283011.436102.h5")
            print("fetch complete")
        except:
            print("fetch failed..")
            # lager en ny model file hvis det ikke fantes en annen fra før av
            return Sequential([
                Conv2D(64,(3,3),input_shape=(self.image_size,self.image_size,1), padding='same', activation='relu', name='Conv2D-1'),
                MaxPooling2D(pool_size=2, name='MaxPool'),
                Conv2D(32,(3,3), activation='relu', name='Conv2D-2',padding='same'),
                MaxPooling2D(pool_size=2, name='MaxPool-2'),
                Conv2D(16,(3,3), activation='relu', name='Conv2D-3',padding='same'),
                Flatten(name='flatten'),
                Dense(512, activation='relu', name='Dense600'),
                Dropout(0.1, name='Dropout1'),
                Dense(self.output, activation='softmax', name='Output')
                ], name="Convolution model 2 batches")

    def model_summary(self):
        self.model.summary()

    def recoginze(self, frame):
        names = list(self.ft.get_names())
        values = []
        gray = self.cv.cvtColor(frame, self.cv.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            if roi.size > 0:
                roi = tf.keras.utils.normalize(roi, axis=-1,order=2)
                roi = self.cv.resize(roi, dsize=(self.image_size,self.image_size), interpolation=self.cv.INTER_NEAREST)
                prediction = self.model.predict(roi.reshape(1,self.image_size,self.image_size,1))
                values.append(names[0][np.array(prediction[0]).argmax()])
        return values

    def train(self):
        # get images and reshape them
        es = EarlyStopping(monitor='val_loss',min_delta=1e-4,patience=10, verbose=2, restore_best_weights=True )
        (x_values, y_labels) = self.ft.train_label()
        number_of_images = len(x_values)
        print("number of training images: " + str(number_of_images))

        x_values = np.array(np.stack(x_values, axis=0).reshape(number_of_images,self.image_size,self.image_size,1))
        y_labels = np.stack(y_labels, axis=0)
        y_labels = keras.utils.to_categorical(y_labels,self.output)

        x_values, y_labels = shuffle(x_values, y_labels, random_state = 101);

        X_train, X_test, y_train, y_test = train_test_split(x_values, y_labels, test_size=0.20, random_state=42)

        adam_optimizer = keras.optimizers.Adam(1e-5)

        self.model.compile(optimizer=adam_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

        self.generator.fit(X_train)
        self.model.fit_generator(self.generator.flow(X_train,y_train,batch_size=32),validation_data=(X_test,y_test), epochs=self.epochs, steps_per_epoch=number_of_images//32, callbacks= [es])

        loss, metrics = self.model.evaluate(X_test,y_test, batch_size = 32)

        print(f"loss value : {loss}")
        print(f"\n metrics : {metrics}")
        self.model.save(f"models/my_model{time.time()}.h5")
