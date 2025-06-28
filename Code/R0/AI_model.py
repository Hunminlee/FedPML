from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.utils import to_categorical


def build_model(num_classes):

    dr = 0.5  # dropout rate (%)
    model = Sequential()
    model.add(Conv2D(50, (1, 8), padding='same', activation="relu", name="conv1", kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(Conv2D(50, (2, 8), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
    model.add(Dropout(dr))
    model.add(Dense(num_classes, kernel_initializer='he_normal', name="dense2", activation="softmax"))

    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    #model.build((None, 2, 128, 1))

    return model