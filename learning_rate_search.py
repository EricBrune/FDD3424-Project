import tensorflow as tf
from tensorflow import keras
from keras.optimizers import optimizer
from keras import layers
from keras import callbacks
import keras_tuner
import numpy as np

num_classes = 10


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

"""Model definition"""

def makeModel(model_name, data):
  
    if model_name == "CNN":
        model = keras.Sequential()

        if data == "CIFAR-10":
            model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
            model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2,2)))
        elif data == "MNIST":
            model.add(layers.Conv2D(28, (3,3), padding='same', activation='relu', input_shape=(28,28,1)))
            model.add(layers.Conv2D(28, (3,3), padding='same', activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2,2)))

        model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))

        model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
        model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))

        return model
    
    elif model_name == "ANN":
        if data == "CIFAR-10":
            model = keras.Sequential(
            [
                keras.Input(shape=3072),
                layers.Dense(256, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
            )
            return model
        elif data == "MNIST":
            model = keras.Sequential(
            [
                keras.Input(shape=784),
                layers.Dense(256, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
            )
            return model
  

lrs = [i*10**j for j in range(-6, 0) for i in range(1, 10)]

def test_lr(optimizer, model_name, data, use_sign = None):
    h = {"accuracy": [], "val_accuracy": []}
    for lr in lrs:
        keras.backend.clear_session()
        model = makeModel(model_name, data)
        if use_sign is not None:
            model.compile(optimizer=optimizer(learning_rate = lr, use_sign = use_sign), loss="categorical_crossentropy", metrics=["accuracy"])
        else:
            model.compile(optimizer=optimizer(learning_rate = lr), loss="categorical_crossentropy", metrics=["accuracy"])
        history = callbacks.History()
        model.fit(train_gen, batch_size=128, epochs=3, validation_data = test_gen, callbacks = [history, keras.callbacks.TensorBoard("tb_logs")], shuffle = True)
        h["accuracy"].append(history.history["accuracy"][-1])
        h["val_accuracy"].append(history.history["val_accuracy"][-1])
        
    return h

"""Gradient PID"""

def GradPID(learning_rate, use_sign):
  if use_sign:
    return GradientPID(learning_rate = learning_rate, rho = 0.96, KI = 2, KD = -0.3, use_sign = use_sign)
  else:
    return GradientPID(learning_rate = learning_rate, rho = 0.95, KI = 2.23, KD = 0.26, use_sign = use_sign)
  #return GradientPID(learning_rate = 0.001, rho = 0.98, KI = 1.7, KD = 0.9, use_sign = True)

"""Momentum PID"""

def MomPID():
  return MomentumPID(learning_rate = 0.08, rho=0.95, KI=1, KD=1)

"""SGD"""

def SGD(learning_rate):
  return keras.optimizers.SGD(learning_rate = learning_rate, momentum=0)
  #return keras.optimizers.SGD(learning_rate = 0.007, momentum=0.9)

def Adam(learning_rate):
  return keras.optimizers.Adam(learning_rate = learning_rate)

"""Get data"""

def get_data(data, model):

    if data == "CIFAR-10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
    
    
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        
        if model == "ANN":
            x_train = x_train.reshape(len(x_train), 3072)
            x_test = x_test.reshape(len(x_test), 3072)

        return x_train, y_train, x_test, y_test
        
    elif data == "MNIST":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
    
    
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        if model == "ANN":
            x_train = x_train.reshape(len(x_train), 784)
            x_test = x_test.reshape(len(x_test), 784)
        
        return x_train, y_train, x_test, y_test
    
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

fig, axs = plt.subplots(2, 2, figsize = (20, 11))

datas = ["MNIST", "CIFAR-10"]
models = ["ANN", "CNN"]

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



for i, ax in enumerate(axs.reshape(-1)):
    data = datas[int(i/2)]
    model = models[i % 2]
    

    x_train, y_train, x_test, y_test = get_data(data, model)

    train_gen = DataGenerator(x_train, y_train, bs)
    test_gen = DataGenerator(x_test, y_test, bs)

    print(data, model)
    print("Adam")
    histAdam = test_lr(Adam, model, data)
    print("Gradient PID")
    histPID_no_sign = test_lr(GradPID, model, data, use_sign = False)
    print("Spider")
    histPID_sign = test_lr(GradPID, model, data, use_sign = True)
    print("SGD")
    hist_SGD = test_lr(SGD, model, data)

    """History for graphing"""


    Adam_acc = histAdam["accuracy"]
    Adam_val_acc = histAdam["val_accuracy"]

    PID_ns_acc = histPID_no_sign["accuracy"]
    PID_ns_val_acc = histPID_no_sign["val_accuracy"]

    PID_acc = histPID_sign["accuracy"]
    PID_val_acc = histPID_sign["val_accuracy"]

    SGD_acc = hist_SGD["accuracy"]
    SGD_val_acc = hist_SGD["val_accuracy"]

    np.save(f"Adam_{data}_{model}.npy", Adam_val_acc)
    np.save(f"GradPID_{data}_{model}.npy", PID_ns_val_acc)
    np.save(f"Spider_{data}_{model}.npy", PID_val_acc)
    np.save(f"SGD_{data}_{model}.npy", SGD_val_acc)

    p_Adam = np.poly1d(np.polyfit(lrs, Adam_val_acc, 10))
    p_SGD = np.poly1d(np.polyfit(lrs, SGD_val_acc, 10))
    p_Spider = np.poly1d(np.polyfit(lrs, PID_ns_val_acc, 10))
    p_GradPID = np.poly1d(np.polyfit(lrs, PID_val_acc, 10))

    ax.scatter(lrs, Adam_val_acc, label = "Adam val. acc.", color = "#E69F00")
    ax.plot(lrs, p_Adam(lrs), color = "#E69F00")
    ax.scatter(lrs, SGD_val_acc, label = "SGD val. acc.", color = "#CC79A7")
    ax.plot(lrs, p_SGD(lrs), color = "#CC79A7")
    ax.scatter(lrs, PID_ns_val_acc, label = "Gradient PID val. acc.", color = "#56B4E9")
    ax.plot(lrs, p_Spider(lrs), color = "#56B4E9")
    ax.scatter(lrs, PID_val_acc, label = "Spider val. acc.", color = "#009E73")
    ax.plot(lrs, p_GradPID(lrs), color = "#009E73")

    
    ax.set_title(f"Dataset: {data}, Model: {model}")
    ax.set_xscale('log')
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False)

fig.tight_layout()
fig.savefig("lr2_3.png", dpi = 300)
fig.savefig("lr2_3.svg")

