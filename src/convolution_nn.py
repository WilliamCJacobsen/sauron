#Keras imports
from keras.models import Sequential
import keras
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout,Flatten

#misc imports 
import numpy as np
#local imports
from Face_trainer import Face_trainer

class ConvolutionNN:
    def __init__(self, cv, cascade):
        self.model = self.make_model()
        self.cv = cv
        self.cascade = cascade

    def make_model(self):
        if False:
            #sjekker om model har en modelFile som er trent opp
            pass
        else:
            # lager en ny model file hvis det ikke fantes en annen fra før av
            return Sequential([
                Conv2D(32,(3,3),input_shape=(124,124,1), activation='relu', name='Conv2D-1'),
                MaxPooling2D(pool_size=2, name='MaxPool'),
                Dropout(0.3, name='Dropout'),
                Flatten(name='flatten'),
                Dense(64, activation='relu', name='Dense'),
                Dense(4, activation='softmax', name='Output')
                ], name="Convolution model 2 batches")

    def model_summary(self):
        self.model.summary()

    # anti-lean development right here
    def data_augmentation(self, x_data, y_data):
        pass

    def training(self):
        # get images and reshape them
        ft = Face_trainer(cv=self.cv, cascade = self.cascade)
        (x_values, y_labels) = ft.train_label()
        number_of_images = len(x_values)
        print("number of training images: " + str(number_of_images))
        x_values = np.array(np.stack(x_values, axis=0).reshape(number_of_images,124,124,1))
        y_labels = np.stack(y_labels, axis=0)
        y_labels = keras.utils.to_categorical(y_labels,4)

        # create a generator 
        model = self.make_model()

        model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

        model.fit(x_values, y_labels, epochs=10, batch_size=32)
