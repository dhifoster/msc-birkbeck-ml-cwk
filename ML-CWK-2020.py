"""

Birkbeck, University of London
Department of Computer & Information Systems
MSc Data Science

Student Name: Dhinta Foster
Student ID: 13156097

Date: April 2020

*** DISCLAIMER *** 

This project was part of an assessed coursework for the Machine Learning module as part of the Master of Science in Data Science degree programme at Birkbeck, University of London.

The aim was to implement a novel optimizer named "Weight–wise Adaptive learning rates with Moving average Estimator" (WAME) Optimizer and run experiments on a neural network architecture of our choice. The purpose of this upload code is to simply disserminate information and to present my implementation of this optimizer.  

More information is available on the README file. Please ensure to cite Mosca et al (2017) if any of the code following is used. Available from: https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2017-50.pdf

"""


import numpy as np

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# REPRODUCIBILITY
np.random.seed(404)

# LOAD DATA - FMNIST DATASET
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# RANDOMLY SORT X_test/y_test
indexes = np.arange(X_test.shape[0])
for _ in range(5): indexes = np.random.permutation(indexes)  # shuffle 5 times!
X_test = X_test[indexes]
y_test = y_test[indexes]

# Implement WAME optimizer using Keras backend. 
# It has the same structure as the optimizers defined in Keras documentation, 
# utilizing init, get_updates and get_config functions

from keras.optimizers import Optimizer
from keras import backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf
    
class WAME(Optimizer):
    """Weight–wise Adaptive learning rates with Moving average Estimator Optimizer. Mosca et al (2017).
        Available from:
       https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2017-50.pdf
    """
    def __init__(self, learning_rate = 0.0001, alpha=0.9, eta_plus = 1.2, eta_minus = 0.1,
                 zeta_min = 0.01, zeta_max = 100, epsilon = 1e-12, **kwargs):
        """Initialise WAME Optimizer with variable values as suggested from the Mosca et al (2017) paper"""
        super(WAME, self).__init__(**kwargs)
        self.learning_rate = K.variable(learning_rate)
        self.alpha = alpha
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max
        self.epsilon = epsilon
        self.initial_decay = kwargs.pop('decay', 0.0)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.decay = K.variable(self.initial_decay, name='decay')
    
    @K.symbolic
    def get_updates(self, params, loss, contraints=None):
        self.updates = [K.update_add(self.iterations, 1)]
        grads        = self.get_gradients(loss, params)
        shapes       = [K.int_shape(p) for p in params]
        old_grads    = [K.zeros(shape) for shape in shapes]
        weights      = [K.zeros(shape) for shape in shapes]
        
        # Learning Rate
        learning_rate = self.learning_rate
        if self.initial_decay > 0:
            learning_rate *= (1. / (1. + self.decay * self.iterations))
            
        t = self.iterations + 1
        
        # Line 2 - initialise current weights
        
        zeta      = [K.ones(shape) for shape in shapes]
        Z         = [K.zeros(shape) for shape in shapes]
        theta     = [K.zeros(shape) for shape in shapes]
        
        for p, g, w, expMA, prevZ, prevTheta, old_g in zip(params, grads, weights, zeta, Z, theta, old_grads):
            change      = g * old_g
            pos_change  = K.greater(change,0.)
            neg_change  = K.less(change,0.)
            
            # Line 3-8: For all t in [1..t] do the following
            
            zeta_t      = K.switch(pos_change,
                                   K.minimum(expMA * self.eta_plus, self.zeta_max),
                                   K.switch(neg_change, K.maximum(expmA * self.eta_minus, self.zeta_min), expMA))
            zeta_clip   = K.clip(zeta_t, self.zeta_min, self.zeta_max)
            
            # Lines 9-12: Update weights for t with amendments as proposed for line 11
            
            Z_t         = (self.alpha * prevZ) + ((1 - self.alpha) * zeta_t)
            theta_t     = (self.alpha * prevTheta) + ((1 - self.alpha) * K.square(g))
            wChange     = - (learning_rate * (zeta_clip /zeta_t) * g) / K.sqrt(theta_t + self.epsilon)
            new_weight = w + wChange 
            p_update    = p - w + new_weight
        
            self.updates.append(K.update(p,p_update))
            self.updates.append(K.update(w,new_weight))
            self.updates.append(K.update(expMA,zeta_t))
            self.updates.append(K.update(prevZ,Z_t))
            self.updates.append(K.update(prevTheta,theta_t))
        
        return self.updates
    
    def get_config(self):
        config = {'alpha': float(K.get_value(self.alpha)),
                  'learning_rate' : float(K.get_value(self.learning_rate)),
                  'eta_plus' : float(K.get_value(self.eta_plus)),
                  'eta_minus' : float(K.get_value(self.eta_minus)),
                  'zeta_min' : float(K.get_value(self.zeta_min)),
                  'zeta_max' : float(K.get_value(self.zeta_max))}
        base_config = super(WAME, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

"""Baseline Model"""

# Implement baseline model with Keras Sequential

def deep_cnn_model():
    
    # Model Layers
    model = Sequential()
    model.add(Conv2D(64, (5, 5), 
                     input_shape=(28, 28, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='softmax'))
    
    # Compile Model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer= WAME(),
                  metrics=['accuracy'])
    
    return model

# Build the model
model = deep_cnn_model()

# Fit the model - test on one epoch
model.fit(X_train,
          y_train,
          validation_data=(X_test, y_test),
          epochs=1,
          batch_size=200)

"""Experiment 1"""

# In order to do hyperparameter tuning, I will utilise more libraries

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt

# Create Keras models such that hyperparameters can be iterable

def build_model(batch_size = 128, rate = 0.2):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), 
                     input_shape=(28, 28, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate))
    model.add(Flatten())
    model.add(Dense(batch_size, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=WAME(),
                  metrics=["acc"])
    return model

# Returns summary of code

model_default = build_model()
model_default.summary()

# Hyperparameter subsets to experiment

_batch_size=[128, 256, 512, 1024]
_rate=[0.2, 0.4, 0.6, 0.8]


params=dict(batch_size=_batch_size,
            rate=_rate)

print(params)

model = KerasClassifier(build_fn=build_model,epochs=10)

np.random.seed(404) # set seed for reproducibility


# Run random search model. Iterations over 5 epochs.
rscv = RandomizedSearchCV(model, param_distributions=params, cv=2,     n_iter=5)

rscv_results = rscv.fit(X_train,y_train)

# print results

print('Best score is: {} using {}'.format(rscv_results.best_score_,
rscv_results.best_params_))

"""Experiment 2"""

# Implement WAME with different learning rate

class WAME(Optimizer):
    """Weight–wise Adaptive learning rates with Moving average Estimator Optimizer. Mosca et al (2017).
        Available from:
       https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2017-50.pdf
    """
    def __init__(self, learning_rate = 0.001, alpha=0.9, eta_plus = 1.2, eta_minus = 0.1,
                 zeta_min = 0.01, zeta_max = 100, epsilon = 1e-12, **kwargs):
        """Initialise WAME Optimizer with variable values as suggested from the Mosca et al (2017) paper"""
        super(WAME, self).__init__(**kwargs)
        self.learning_rate = K.variable(learning_rate)
        self.alpha = alpha
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max
        self.epsilon = epsilon
        self.initial_decay = kwargs.pop('decay', 0.0)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.decay = K.variable(self.initial_decay, name='decay')
    
    @K.symbolic
    def get_updates(self, params, loss, contraints=None):
        self.updates = [K.update_add(self.iterations, 1)]
        grads        = self.get_gradients(loss, params)
        shapes       = [K.int_shape(p) for p in params]
        old_grads    = [K.zeros(shape) for shape in shapes]
        weights      = [K.zeros(shape) for shape in shapes]
        
        # Learning Rate
        learning_rate = self.learning_rate
        if self.initial_decay > 0:
            learning_rate *= (1. / (1. + self.decay * self.iterations))
            
        t = self.iterations + 1
        
        # Line 2 - initialise current weights
        
        zeta      = [K.ones(shape) for shape in shapes]
        Z         = [K.zeros(shape) for shape in shapes]
        theta     = [K.zeros(shape) for shape in shapes]
        
        for p, g, w, expMA, prevZ, prevTheta, old_g in zip(params, grads, weights, zeta, Z, theta, old_grads):
            change      = g * old_g
            pos_change  = K.greater(change,0.)
            neg_change  = K.less(change,0.)
            
            # Line 3-8: For all t in [1..t] do the following
            
            zeta_t      = K.switch(pos_change,
                                   K.minimum(expMA * self.eta_plus, self.zeta_max),
                                   K.switch(neg_change, K.maximum(expmA * self.eta_minus, self.zeta_min), expMA))
            zeta_clip   = K.clip(zeta_t, self.zeta_min, self.zeta_max)
            
            # Lines 9-12: Update weights for t with amendments as proposed for line 11
            
            Z_t         = (self.alpha * prevZ) + ((1 - self.alpha) * zeta_t)
            theta_t     = (self.alpha * prevTheta) + ((1 - self.alpha) * K.square(g))
            wChange     = - (learning_rate * (zeta_clip /zeta_t) * g) / K.sqrt(theta_t + self.epsilon)
            new_weight = w + wChange 
            p_update    = p - w + new_weight
        
            self.updates.append(K.update(p,p_update))
            self.updates.append(K.update(w,new_weight))
            self.updates.append(K.update(expMA,zeta_t))
            self.updates.append(K.update(prevZ,Z_t))
            self.updates.append(K.update(prevTheta,theta_t))
        return self.updates
    
    def get_config(self):
        config = {'alpha': float(K.get_value(self.alpha)),
                  'learning_rate' : float(K.get_value(self.learning_rate)),
                  'eta_plus' : float(K.get_value(self.eta_plus)),
                  'eta_minus' : float(K.get_value(self.eta_minus)),
                  'zeta_min' : float(K.get_value(self.zeta_min)),
                  'zeta_max' : float(K.get_value(self.zeta_max))}
        base_config = super(WAME, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Build iterable model

def build_model(batch_size = 128, rate = 0.2):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), 
                     input_shape=(28, 28, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate))
    model.add(Flatten())
    model.add(Dense(batch_size, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=WAME(),
                  metrics=["acc"])
    return model

# Hyperparameters to test 
_batch_size=[128, 256, 512, 1024]
_rate=[0.2, 0.4, 0.6, 0.8]


params=dict(batch_size=_batch_size,
            rate=_rate)

print(params)

# Create model & set seed

model = KerasClassifier(build_fn=build_model,epochs=10)

np.random.seed(404)

# Run randomised search

rscv = RandomizedSearchCV(model, param_distributions=params, cv=2,     n_iter=5)

rscv_results = rscv.fit(X_train,y_train)

# Print results

print('Best score is: {} using {}'.format(rscv_results.best_score_,
rscv_results.best_params_))

"""Experiment 3"""

# Implement WAME with different learning rate

class WAME(Optimizer):
    """Weight–wise Adaptive learning rates with Moving average Estimator Optimizer. Mosca et al (2017).
        Available from:
       https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2017-50.pdf
    """
    def __init__(self, learning_rate = 0.01, alpha=0.9, eta_plus = 1.2, eta_minus = 0.1,
                 zeta_min = 0.01, zeta_max = 100, epsilon = 1e-12, **kwargs):
        """Initialise WAME Optimizer with variable values as suggested from the Mosca et al (2017) paper"""
        super(WAME, self).__init__(**kwargs)
        self.learning_rate = K.variable(learning_rate)
        self.alpha = alpha
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max
        self.epsilon = epsilon
        self.initial_decay = kwargs.pop('decay', 0.0)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.decay = K.variable(self.initial_decay, name='decay')
    
    @K.symbolic
    def get_updates(self, params, loss, contraints=None):
        self.updates = [K.update_add(self.iterations, 1)]
        grads        = self.get_gradients(loss, params)
        shapes       = [K.int_shape(p) for p in params]
        old_grads    = [K.zeros(shape) for shape in shapes]
        weights      = [K.zeros(shape) for shape in shapes]
        
        # Learning Rate
        learning_rate = self.learning_rate
        if self.initial_decay > 0:
            learning_rate *= (1. / (1. + self.decay * self.iterations))
            
        t = self.iterations + 1
        
        # Line 2 - initialise current weights
        
        zeta      = [K.ones(shape) for shape in shapes]
        Z         = [K.zeros(shape) for shape in shapes]
        theta     = [K.zeros(shape) for shape in shapes]
        
        for p, g, w, expMA, prevZ, prevTheta, old_g in zip(params, grads, weights, zeta, Z, theta, old_grads):
            change      = g * old_g
            pos_change  = K.greater(change,0.)
            neg_change  = K.less(change,0.)
            
            # Line 3-8: For all t in [1..t] do the following
            
            zeta_t      = K.switch(pos_change,
                                   K.minimum(expMA * self.eta_plus, self.zeta_max),
                                   K.switch(neg_change, K.maximum(expmA * self.eta_minus, self.zeta_min), expMA))
            zeta_clip   = K.clip(zeta_t, self.zeta_min, self.zeta_max)
            
            # Lines 9-12: Update weights for t with amendments as proposed for line 11
            
            Z_t         = (self.alpha * prevZ) + ((1 - self.alpha) * zeta_t)
            theta_t     = (self.alpha * prevTheta) + ((1 - self.alpha) * K.square(g))
            wChange     = - (learning_rate * (zeta_clip /zeta_t) * g) / K.sqrt(theta_t + self.epsilon)
            new_weight = w + wChange 
            p_update    = p - w + new_weight
        
            self.updates.append(K.update(p,p_update))
            self.updates.append(K.update(w,new_weight))
            self.updates.append(K.update(expMA,zeta_t))
            self.updates.append(K.update(prevZ,Z_t))
            self.updates.append(K.update(prevTheta,theta_t))
        return self.updates
    
    def get_config(self):
        config = {'alpha': float(K.get_value(self.alpha)),
                  'learning_rate' : float(K.get_value(self.learning_rate)),
                  'eta_plus' : float(K.get_value(self.eta_plus)),
                  'eta_minus' : float(K.get_value(self.eta_minus)),
                  'zeta_min' : float(K.get_value(self.zeta_min)),
                  'zeta_max' : float(K.get_value(self.zeta_max))}
        base_config = super(WAME, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Build iterable model

def build_model(batch_size = 128, rate = 0.2):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), 
                     input_shape=(28, 28, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate))
    model.add(Flatten())
    model.add(Dense(batch_size, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=WAME(),
                  metrics=["acc"])
    return model

# Hyperparameters to test 
_batch_size=[128, 256, 512, 1024]
_rate=[0.2, 0.4, 0.6, 0.8]


params=dict(batch_size=_batch_size,
            rate=_rate)

print(params)

# Create model & set seed

model = KerasClassifier(build_fn=build_model,epochs=10)

np.random.seed(404)

# Run randomised search

rscv = RandomizedSearchCV(model, param_distributions=params, cv=2,     n_iter=5)

rscv_results = rscv.fit(X_train,y_train)

# Print results

print('Best score is: {} using {}'.format(rscv_results.best_score_,
rscv_results.best_params_))

"""Experiment 4 - Best Model"""

# Define model layers with the best hyperparameters from experiments 1-3 and split into validation sets
# Add additional metrics to calcuate to be able to create graphs later
# Use WAME implementation with lr at 1e-3

np.random.seed(404)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

def best_model():
    model_best = Sequential()
    model_best.add(Conv2D(64, (5, 5), 
                     input_shape=(28, 28, 1),
                     activation='relu'))
    model_best.add(MaxPooling2D(pool_size=(2, 2)))
    model_best.add(Conv2D(64, (5, 5), activation='relu'))
    model_best.add(MaxPooling2D(pool_size=(2, 2)))
    model_best.add(Dropout(0.4))
    model_best.add(Flatten())
    model_best.add(Dense(512, activation='softmax'))
    model_best.compile(loss="sparse_categorical_crossentropy",
                  optimizer=WAME(),
                  metrics=['mse', 'mae', 'mape', 'cosine', 'acc'])
    return model_best

# Confirm summary of created model

model_2 = best_model()
model_2.summary()

# Re-run model over 30 epochs

history = model_2.fit(X_train, y_train, epochs=30, batch_size=1, validation_data=(X_val, y_val))

from numpy import array
from matplotlib import pyplot
from sklearn.metrics import classification_report

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Print classification report

y_pred = model_2.predict(X_test, batch_size=10, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_bool))