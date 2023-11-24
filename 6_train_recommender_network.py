import tensorflow as tf
from tensorflow import keras
from keras import layers
import pickle
import random
import numpy as np
import time
import matplotlib.pyplot as plt


def load_svd_model(pkl):
    """ loads a singlular value decomposition model """
    with open(pkl, 'rb') as file:
        svd = pickle.load(file)
    
    return svd


def reduce_dimensionality(model, array):
    """ transforms a data array using a pretrained model - one row at a time to reduce memory usage """
    mapped_array = []
    for row in array:
        mapped_row = model.transform([row])
        mapped_array.append(mapped_row)

    array = np.array(mapped_array)
    array = np.squeeze(array)
    print(array.shape)
    
    return array


# define neural network
input = layers.Input((384))
noise1 = layers.GaussianNoise(stddev=0.000)(input)       
dropout1 = layers.Dropout(0.00)(noise1)                    
hidden1 = layers.Dense(1024, activation='relu')(dropout1)
bnorm = layers.BatchNormalization(axis=-1)(hidden1)   
hidden2 = layers.Dense(512, activation='relu')(bnorm) 
output = layers.Dense(384, activation='softmax')(hidden2)

model = keras.models.Model(inputs=input, outputs=output)
model.summary()

# compile model
model.compile(optimizer='adam', 
              metrics='categorical_accuracy', 
              loss='categorical_crossentropy')  

# early stopping critera
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1, restore_best_weights=True)


# load data and decomposer
with open('training data/x_train.pkl', 'rb') as file:
    x_train = pickle.load(file)
with open('training data/y_train.pkl', 'rb') as file:
    y_train = pickle.load(file)
with open('training data/x_test.pkl', 'rb') as file:
    x_test = pickle.load(file)
with open('training data/y_test.pkl', 'rb') as file:
    y_test = pickle.load(file)

# decompose training data
svd = load_svd_model('models/svd.pkl')
x_train = reduce_dimensionality(svd, x_train)
y_train = reduce_dimensionality(svd, y_train)

# select random indexes from training data without replacement
VAL_SPLIT = 0.15
random_nums = random.sample(range(len(x_train)), int(len(x_train)*VAL_SPLIT))    

# move selected data to end of training datasets for automatic validation_split
random_nums.sort(reverse=True)
x_train = np.append(x_train, x_train[random_nums], axis=0)
x_train = np.delete(x_train, [random_nums], axis=0)

y_train = np.append(y_train, y_train[random_nums], axis=0)
y_train = np.delete(y_train, [random_nums], axis=0)

# train model
TITLE = ('n=1024,bn,512_bs=512_recom1')
start = time.time()
history = model.fit(x_train, y_train, 
                    batch_size=256, 
                    epochs=100, 
                    validation_split=VAL_SPLIT, 
                    callbacks=[es], 
                    verbose=2) 
end = time.time()
minutes = int((end-start)/60)
model.save('models/recom1')


# decompose testing data
x_test = reduce_dimensionality(svd, x_test)
y_test = reduce_dimensionality(svd, y_test)

# test model
test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test cat. accuracy:", 1-test_scores[1])

# plot training history
fig, (acc, los) = plt.subplots(1,2)
fig.suptitle(TITLE+'  '+str(minutes)+'_min')   #+'  '+score+'%_solved')
acc.plot(history.history['categorical_accuracy'])
acc.plot(history.history['val_categorical_accuracy'])
los.plot(history.history['loss'])
los.plot(history.history['val_loss'])
los.yaxis.tick_right()
acc.set_xlabel('epoch')
acc.set_ylabel('categorical accuracy')
los.set_xlabel('epoch')
los.set_ylabel('loss')
acc.grid()
los.grid()
acc.legend(['train', 'val'], loc='lower right')
los.legend(['train', 'val'], loc='upper right')
fig.text(0.12, 0.92, 'test cat. acc.: {}'.format(round(1-test_scores[1],4)), fontsize=9, verticalalignment='top')
fig.text(0.55, 0.92, 'test loss: {}'.format(round(test_scores[0],3)), fontsize=9, verticalalignment='top')
plt.savefig(r'training graphs/{}.png'.format(TITLE))