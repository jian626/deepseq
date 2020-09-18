
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, BatchNormalization, ReLU
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.optimizers import Adam,SGD
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import tensorflow as tf
import utili
import BioDefine
from datetime import datetime
from sklearn.metrics import classification_report


begin = datetime.now()
current_time = begin.strftime("%H:%M:%S")
print("begin time:", current_time)

fraction = 0.2
number = 40000
maxFeatures = len(BioDefine.aaList)
embeddingDims = 16 
filters = 64 
delta = 8
kernelSize = 3
hidden1Dim = 256 
hidden2Dim = 256
strides = 3  
pool_size = 3 
layerLen = 3 
convLen = 1 
poolLen = 1 
epochs = 1000
train_percent = 0.8
batch_size = 400


#myfile = 'uniprot-reviewed_yes.tab'
myfile = 'final_result.csv'
df = pd.read_csv(myfile,sep=',')
print("total set:", df.shape[0])
df = df.drop(df[df.Length>1000].index)
print("after drop:", df.shape[0])
df = df.sample(frac=fraction)
using_set_num = df.shape[0]
print("using set num:", using_set_num)
df = df.reindex(np.random.permutation(df.index))

df['Lables'] = 1 
df.loc[pd.isna(df['EC number']), 'Lables'] = 0 
print('enzyme cnt:',df[df.Lables>0].shape[0])
print('non-enzyme cnt:', df[df.Lables==0].shape[0])

maxLen = max(df['Length'])
print('maxlen:', maxLen)
df['Encode'] = df['Sequence'].apply(lambda x:utili.GetOridinalEncoding(x, BioDefine.aaList, maxLen))

training_set = df.iloc[:int(using_set_num * train_percent)]
print('training set enzyme cnt:',training_set[training_set.Lables>0].shape[0])
print('training non-enzyme cnt:', training_set[training_set.Lables==0].shape[0])
test_set = df.iloc[training_set.shape[0]:]
print("training len:", training_set.shape[0])
print("test len:", test_set.shape[0])


x_train = training_set['Encode']
x_train = sequence.pad_sequences(x_train, maxlen=maxLen)

y_train = training_set['Lables']
y_train = to_categorical(y_train)

x_test = test_set['Encode']
x_test = sequence.pad_sequences(x_test, maxlen=maxLen)

y_test = test_set['Lables']
y_test = to_categorical(y_test)


print(x_train.shape)
print(y_train.shape)
print('----------------------x_train-----------------\n')
print(x_train)
print('\n----------------------y_train-----------------\n')
print(y_train)
model = Sequential()
model.add(Embedding(maxFeatures,
                    embeddingDims,
                    input_length=maxLen))

model.add(BatchNormalization())
model.add(ReLU())
for layerNumber in range(layerLen):
	for _ in range(convLen):
		model.add(Conv1D(filters + layerNumber * delta, kernelSize, padding='same', activation='relu'))
	if layerNumber % 2 == 0:
		model.add(Dropout(0.2))
		for _ in range(poolLen):
			model.add(MaxPooling1D(pool_size=pool_size, strides=strides, padding='same'))
		model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(hidden2Dim))
model.add(Dense(hidden2Dim))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax')) 

optimizer = Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(model.summary())

callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', restore_best_weights=True, patience=40, verbose=1)

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=1/6, callbacks=[callback])
model.save_weights(save_name+ '.h5')

print('==============history=============')
loss, accuacy = model.evaluate(x_test, y_test)
print('loss:',loss)
print('accuacy:', accuacy)

y_pred = model.predict(x_test)

y_pred = (y_pred > 0.5)

report = classification_report(y_test, y_pred)
print('report:')
print(report)


end = datetime.now()
current_time = end.strftime("%H:%M:%S")
print("end time:", current_time)
print("end - begin:", end - begin)
print("========================done==========================")
