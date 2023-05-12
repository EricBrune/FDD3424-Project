import tensorflow as tf
from tensorflow import keras
from keras.optimizers import optimizer
from keras import layers
from keras import callbacks
import keras_tuner
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
from tensorflow.keras.applications import MobileNetV2, ConvNeXtTiny, ResNet50

tf.random.set_seed(247)

class GradientPID(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=0.01,
        rho=0.95,
        KI = 1.0,
        KD = 0.5,
        use_sign = False,
        name="GradientPID",
        **kwargs
    ):
        super().__init__(
            name=name,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.rho = rho
        self.KI = KI
        self.KD = KD
        self.use_sign = use_sign
        if isinstance(rho, (int, float)) and (
            rho < 0 or rho > 1
        ):
            raise ValueError("`rho` must be between [0, 1].")

    def build(self, var_list):
        """Initialize optimizer variables.
          var_list: list of model variables to build SGD variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.integrals = []
        for var in var_list:
            self.integrals.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="integral"
                )
            )
        self.prev_grads = []
        for var in var_list:
            self.prev_grads.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="prev_g"
                )
            )
        self._built = True

    def update_step(self, gradient, variable):
        # parameters
        eta = tf.cast(self.learning_rate, variable.dtype)
        rho = tf.cast(self.rho, variable.dtype)
        KI = tf.cast(self.KI, variable.dtype)
        KD = tf.cast(self.KD, variable.dtype)

        # memory
        var_key = self._var_key(variable)
        integral = self.integrals[self._index_dict[var_key]]
        prev_g = self.prev_grads[self._index_dict[var_key]]

        # update integral
        integral.assign(integral*rho + gradient*(1-rho))
        
        # calculate step
        if self.use_sign:
          variable.assign_add(- tf.math.sign(integral * KI + gradient + (gradient-prev_g)*KD) * eta)
        else:
          variable.assign_add(- (integral * KI + gradient + (gradient-prev_g)*KD) * eta)

        # update previous gradient
        prev_g.assign(gradient)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "rho": self.rho,
                "KI": self.KI,
                "KD": self.KD,
                "use_sign": self.use_sign
            }
        )
        return config

class MomentumPID(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=0.0001,
        rho=0.95,
        KI = 1.0,
        KD = 1.0,
        name="GradientPID",
        **kwargs
    ):
        super().__init__(
            name=name,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.rho = rho
        self.KI = KI
        self.KD = KD
        if isinstance(rho, (int, float)) and (
            rho < 0 or rho > 1
        ):
            raise ValueError("`rho` must be between [0, 1].")

    def build(self, var_list):
        """Initialize optimizer variables.
          var_list: list of model variables to build SGD variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.momentums = []
        for var in var_list:
            self.momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="momentum"
                )
            )
        self.integrals = []
        for var in var_list:
            self.integrals.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="integral"
                )
            )
        self._built = True

    def update_step(self, gradient, variable):
        # parameters
        eta = tf.cast(self.learning_rate, variable.dtype)
        rho = tf.cast(self.rho, variable.dtype)
        KI = tf.cast(self.KI, variable.dtype)
        KD = tf.cast(self.KD, variable.dtype)

        # memory
        var_idx = self._index_dict[self._var_key(variable)]
        momentum = self.momentums[var_idx]
        integral = self.integrals[var_idx]

        # update momentum and integral
        momentum.assign(momentum*rho + gradient*(1-rho))
        integral.assign(integral*rho + momentum*(1-rho))
        
        # calculate step
        variable.assign_add(- (integral*KI + momentum + gradient*KD) * eta)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "rho": self.rho,
                "KI": self.KI,
                "KD": self.KD
            }
        )
        return config
    


"""Get data"""

num_classes = 100
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

bs = 128
n_epochs = 15

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

train_gen = DataGenerator(x_train, y_train, bs)
test_gen = DataGenerator(x_test, y_test, bs)

"""Model definition"""

def makeModel():
    model = keras.Sequential()
    # 128 and not only 32 filters because there are 100 classes. 32 filters gave bad results.
    model.add(Conv2D(128, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('elu'))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('elu'))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('elu'))

    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('elu'))
    model.add(Dropout(0.4))
    model.add(Dense(100))
    model.add(Activation('softmax'))

    return model    

def scheduler(epoch, lr):
     if epoch % 5 == 0 and epoch > 0:
        return lr * 0.1
     else:
         return lr

"""Train single model"""

def trainModel(optimizer, use_sign = None):
    keras.backend.clear_session()
    model = makeModel()
    if use_sign is not None:
        model.compile(optimizer=optimizer(use_sign = use_sign), loss="categorical_crossentropy", metrics=["accuracy"])
    else:
        model.compile(optimizer=optimizer(), loss="categorical_crossentropy", metrics=["accuracy"])
    history = callbacks.History()
    schedule_callback = keras.callbacks.LearningRateScheduler(scheduler)
    model.fit(train_gen, batch_size=bs, epochs=n_epochs, validation_data=test_gen, callbacks = [history, keras.callbacks.TensorBoard("tb_logs"), schedule_callback], shuffle = True)
    return history.history

"""Gradient PID"""

def GradPID(use_sign):
    if use_sign:
        return GradientPID(learning_rate = 2e-4, rho = 0.96, KI = 2, KD = -0.3, use_sign = use_sign)
    else:
        return GradientPID(learning_rate = 5e-2, rho = 0.95, KI = 2.23, KD = 0.26, use_sign = use_sign)

"""Momentum PID"""

def MomPID():
  return MomentumPID(learning_rate = 0.08, rho=0.95, KI=1, KD=1)

"""SGD"""

def SGD(): 
  return keras.optimizers.SGD(learning_rate = 5e-2, momentum=0)
  #return keras.optimizers.SGD(learning_rate = 0.007, momentum=0.9)

def Adam():
   return keras.optimizers.Adam(learning_rate = 5e-4)

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


fig, ax = plt.subplots(2, 1, figsize = (12, 14))

histSGD = trainModel(SGD)
hist_Adam = trainModel(Adam)
histPID_no_sign = trainModel(GradPID, use_sign = False)
histPID_sign = trainModel(GradPID, use_sign = True)

"""History for graphing"""


SGD_acc = histSGD["accuracy"]
SGD_val_acc = histSGD["val_accuracy"]

PID_ns_acc = histPID_no_sign["accuracy"]
PID_ns_val_acc = histPID_no_sign["val_accuracy"]

PID_acc = histPID_sign["accuracy"]
PID_val_acc = histPID_sign["val_accuracy"]

Adam_acc = hist_Adam["accuracy"]
Adam_val_acc = hist_Adam["val_accuracy"]

SGD_loss = histSGD["loss"]
SGD_val_loss = histSGD["val_loss"]

PID_ns_loss = histPID_no_sign["loss"]
PID_ns_val_loss = histPID_no_sign["val_loss"]

PID_loss = histPID_sign["loss"]
PID_val_loss = histPID_sign["val_loss"]

Adam_loss = hist_Adam["loss"]
Adam_val_loss = hist_Adam["val_loss"]

epochs = np.arange(1, n_epochs + 1)

ax[0].plot(epochs, Adam_acc, label = "Adam, training accuracy", color = "#E69F00")
ax[0].plot(epochs, Adam_val_acc, linestyle='dashed', label = "Adam, validation accuracy", color = "#E69F00")
ax[0].plot(epochs, PID_ns_acc, linestyle='solid', label = "Gradient PID, training accuracy", color = "#56B4E9")
ax[0].plot(epochs, PID_ns_val_acc, linestyle='dashed', label = "Gradient PID validation accuracy", color = "#56B4E9")
ax[0].plot(epochs, PID_acc, linestyle='solid', label = "Spider, training accuracy", color = "#009E73")
ax[0].plot(epochs, PID_val_acc, linestyle='dashed', label = "Spider, validation accuracy", color = "#009E73")
ax[0].plot(epochs, SGD_acc, linestyle='solid', label = "SGD, training accuracy", color = "#CC79A7")
ax[0].plot(epochs, SGD_val_acc, linestyle='dashed', label = "SGD, training loss", color = "#CC79A7")
ax[1].plot(epochs, Adam_loss, linestyle='solid', label = "Adam, training loss", color = "#E69F00")
ax[1].plot(epochs, Adam_val_loss, linestyle='dashed', label = "Adam, validation loss", color = "#E69F00")
ax[1].plot(epochs, PID_ns_loss, linestyle='solid', label = "Gradient PID, training loss", color = "#56B4E9")
ax[1].plot(epochs, PID_ns_val_loss, linestyle='dashed', label = "Gradient PID validation loss", color = "#56B4E9")
ax[1].plot(epochs, PID_loss, linestyle='solid', label = "Spider, training loss", color = "#009E73")
ax[1].plot(epochs, PID_val_loss, linestyle='dashed', label = "Spider, validation loss", color = "#009E73")
ax[1].plot(epochs, SGD_loss, linestyle='solid', label = "SGD, validation loss", color = "#CC79A7")
ax[1].plot(epochs, SGD_val_loss, linestyle='dashed', label = "SGD, training loss", color = "#CC79A7")
ax[0].set_xlabel("Epochs")
ax[1].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")
ax[1].set_ylabel("Loss")
ax[0].legend(frameon = False)
ax[1].legend(frameon = False, loc = 'upper center')

fig.tight_layout()
fig.savefig("comparison_sgd.png", dpi = 300)
fig.savefig("comparison_sgd.svg")

