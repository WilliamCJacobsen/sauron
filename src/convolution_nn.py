#Keras imports
from keras.models import Sequential
from keras.models import load_model
import keras
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout,Flatten

#misc imports 
import numpy as np
#local imports
from Face_trainer import Face_trainer

class ConvolutionNN:
    def __init__(self, cv, cascade):
        self.model = self.get_model()
        self.cv = cv
        self.cascade = cascade

    def get_model(self):
        
        if False:
            #sjekker om model har en modelFile som er trent opp
            pass
        else:
            # lager en ny model file hvis det ikke fantes en annen fra før av
            return Sequential([
                Conv2D(32,(3,3),input_shape=(300,300,1), activation='relu', name='Conv2D-1'),
                MaxPooling2D(pool_size=2, name='MaxPool'),
                Dropout(0.3, name='Dropout'),
                Flatten(name='flatten'),
                Dense(64, activation='relu', name='Dense'),
                Dense(4, activation='softmax', name='Output')
                ], name="Convolution model 2 batches")

    def model_summary(self):
        self.model.summary()

    def recoginze(self):
        self.model.predict()
        pass

    def training(self):
        # get images and reshape them
        ft = Face_trainer(cv=self.cv, cascade = self.cascade)
        (x_values, y_labels) = ft.train_label()
        number_of_images = len(x_values)
        print("number of training images: " + str(number_of_images))

        x_values = np.array(np.stack(x_values, axis=0).reshape(number_of_images,300,300,1))
        y_labels = np.stack(y_labels, axis=0)
        y_labels = keras.utils.to_categorical(y_labels,4)

        self.model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

        self.model.fit(x_values, y_labels, epochs=1, batch_size=32)
        self.model.save("src/model/my_model.h5") 
