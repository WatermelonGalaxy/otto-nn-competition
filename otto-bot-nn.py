#import libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt


train_org = pd.read_csv('train.csv')
test_org = pd.read_csv('test.csv')
sample_org = pd.read_csv('sampleSubmission.csv')

train_org[-10:]


train_labels = train_org.target.values
train_labels[:-10]

np.unique(train_labels)

#drop ids and labels
train = train_org.drop('id', axis=1)
train = train.drop('target', axis=1)
test = test_org.drop('id', axis=1)

# convert labels to numric values
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(train_labels)
np.unique(labels)

train = train.astype("float32")
test = test.astype("float32")

print(train.shape, test.shape, train_labels.shape)

train.describe()

# pre-processing: normalize the data attributes
# scaler = preprocessing.StandardScaler()
# train = scaler.fit_transform(train)

# Create a neural network
from tensorflow import keras
from tensorflow.keras import layers

keras.backend.clear_session()

#create model
model = keras.Sequential([
    layers.Dense(512, activation='sigmoid', input_shape=train.shape[1:]), #input_shape=93
    layers.Dense(256, activation='sigmoid'),
    # layers.Dense(128, activation='sigmoid'),
    # layers.Dense(64, activation='sigmoid'),
    layers.Dropout(0.2),
    layers.Dense(9, activation='softmax')
])

#compile model
model.compile(optimizer='rmsprop',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

#train model
history = model.fit(train, labels, epochs=15, validation_split=0.2)

#plot accuracy
plt.plot(history.history['loss'], label='train')
plt.xlabel('Epoch')
plt.ylabel('train loss')
plt.ylim([0, 1])
plt.legend(loc='lower right')

# #evaluate model
# test_loss, test_acc = model.evaluate(test,  labels, verbose=2)

# print('\nTest accuracy:', test_acc)

# #predict test data
# predictions = model.predict(test)

# #convert predictions to labels
# pred_labels = np.argmax(predictions, axis=1)
# pred_labels

# #convert labels to strings
# pred_labels = lbl_enc.inverse_transform(pred_labels)
# pred_labels

#submissions: rename variables if needed
test_sub = test
model = model

#Run model on test data
predictions = model.predict(test_sub)
predictions[0]

predictions[0].argmax()

sub = pd.DataFrame(np.int32(predictions.round()))
sub

sub.columns = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
sub.insert(loc=0,column='id',value = test_org.id)
sub

sub.to_csv("kaggle_otto_submission.csv",index=False)