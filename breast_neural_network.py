# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:44:08 2022

@author: owner
"""

import numpy as np 
import pandas as pd
import tensorflow as tf

data = pd.read_csv(r"C:\Users\owner\Desktop\SHRDC MIDA AIML\Deep Learning\Git Repo\Neural Network Project 1\data.csv")

#%%
data = data.drop(columns=["id"])
#%%

x = data.drop(columns=["diagnosis"])
y = data["diagnosis"]
#%%

y = y.replace("M", 0)
y = y.replace("B", 1)
#%%
y = np.array(y)
x = np.array(x)
#%%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
#%%
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)
#%%
input_shape = x_train_sc.shape[1]
num_classes = len(np.unique(y_train))

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape = input_shape),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

model.summary()
#%%

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
#%%
EPOCH = 20
BATCH_SIZE = 32


history = model.fit(x_train_sc, y_train, validation_data = (x_test_sc, y_test), epochs=EPOCH, batch_size = BATCH_SIZE)

#%%