import numpy as np
import os

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


def moving_average(x, w=5):
    return np.convolve(x, np.ones(w), 'valid') / w


lrs = [i*10**j for j in range(-6, 0) for i in range(1, 10)]

fig, axs = plt.subplots(2, 2, figsize = (20, 11))

datas = ["MNIST", "CIFAR-10"]
models = ["ANN", "CNN"]
    
for i, ax in enumerate(axs.reshape(-1)):
    model = models[int(i/2)]
    data = datas[i % 2]

    for file in os.listdir("Vals"):
        if data in file and model in file:
            if "Adam" in file:
                print(data, model)
                print("Adam")
                Adam_val_acc = np.load(f"Vals/{file}")
                ax.scatter(lrs, Adam_val_acc, label = "Adam val. acc.", color = "#E69F00", s = 7)
                ax.plot(lrs[2:-2], moving_average(Adam_val_acc), color = "#E69F00")

            if "GradPID" in file:
                print("Gradient PID")
                PID_ns_val_acc = np.load(f"Vals/{file}")
                ax.scatter(lrs, PID_ns_val_acc, label = "Gradient PID val. acc.", color = "#56B4E9", s = 7)
                ax.plot(lrs[2:-2], moving_average(PID_ns_val_acc), color = "#56B4E9")

            
            if "Spider" in file:
                print("Spider")
                PID_val_acc = np.load(f"Vals/{file}")
                ax.scatter(lrs, PID_val_acc, label = "Spider val. acc.", color = "#009E73", s = 7)
                ax.plot(lrs[2:-2], moving_average(PID_val_acc), color = "#009E73")

            
            if "SGD" in file:
                print("SGD")
                SGD_val_acc = np.load(f"Vals/{file}")
                ax.scatter(lrs, SGD_val_acc, label = "SGD val. acc.", color = "#CC79A7", s = 7)
                ax.plot(lrs[2:-2], moving_average(SGD_val_acc), color = "#CC79A7")



            
        ax.set_title(f"Dataset: {data}, Model: {model}")
        ax.set_xscale('log')
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Accuracy")
        ax.legend(frameon=False)

fig.tight_layout()
fig.savefig("lr2_4_smaller.png", dpi = 300)
fig.savefig("lr2_4_smaller.svg")
