import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
from sklearn.model_selection import train_test_split
from data_loader import df, resolution

tf.random.set_seed(
            seed=0
)

## importing data
data_train, data_test = train_test_split(df, test_size=0.2)

x_train = list()
y_train = list()
x_test = list()
y_test = list()

for idx in range(len(data_train)):
    x_train.append(data_train[idx][0]) 
    y_train.append(data_train[idx][1]) 
for idx in range(len(data_test)):
    x_test.append(data_test[idx][0]) 
    y_test.append(data_test[idx][1]) 
    
x_train = np.reshape(x_train, (len(data_train), 1, resolution, resolution)).astype('float32')
x_test = np.reshape(x_test, (len(data_test), 1, resolution, resolution)).astype('float32')
y_train = np.reshape(y_train, (len(data_train), 1))
y_test = np.reshape(y_test, (len(data_test), 1))

## Creating CNN model
def cnn_model():

    filters = 50
    kernel = (6,6)
    l1_rate = 1e-4

    model=Sequential()
    model.add(Conv2D(filters,kernel, padding='same',input_shape=(1,resolution,resolution), 
                     activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l1_rate)))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    opt = adam_v2.Adam(learning_rate=1e-3)
    model.compile(loss='BinaryCrossentropy', optimizer=opt, metrics=['AUC'])
    model.summary()
    return model

model = cnn_model() 

## ML model training
batch_size = 10
epochs = 20
history = model.fit(
                    x=x_train,
                    y=y_train,
                    batch_size=batch_size, 
                    epochs=epochs
)

## Displaying results
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Loss: {}, Test Accuracy: {}'.format(test_loss, test_acc))

y_pred = model.predict(x_test)
y_class = np.round(y_pred)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()

confusion_mtx = confusion_matrix(y_test, y_class)
fig, ax = plt.subplots(figsize=(12,8))
ax = heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap='Blues')
ax.set_xlabel('Prediciton Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')
plt.show()



