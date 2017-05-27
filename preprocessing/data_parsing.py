import os
import numpy as np

def get_data(train=True):
    datafile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dsjtzs_txfz_training.txt")

    tracex, tracey, tracet, target = [], [], [], []
    if train is True:
        label = []
    with open(datafile, "r") as datafile:
        for line in datafile:
            if train is True:
                datarow, targetline, labelline = line.rstrip().split(" ")[1:]
            else:
                datarow, targetline = line.rstrip().split(" ")[1:]
            datarow = datarow.split(";")[:-1]
            datax, datay, datat = [], [], []
            for item in datarow:
                x, y, t = item.split(",")
                datax.append(float(x))
                datay.append(float(y))
                datat.append(float(t))
            tracex.append(np.array(datax))
            tracey.append(np.array(datay))
            tracet.append(datat)
            targetx, targety = targetline.split(",")
            target.append(np.array((float(targetx), float(targety))))
            if train is True:
                label.append(int(labelline))

    return tracex, tracey, tracet, target, label

if __name__ == '__main__':
    get_data()