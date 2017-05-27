from preprocessing.data_parsing import get_data
import numpy as np

def raw_data():
    """
    This function will output the raw data parsed from training file.

    :return:
    tracexynp: numpy array of training x, y (dim is (3000, 2, length))
    tracexyt: numpy array of corresponding training t
    targetnp: numpy array of target coordinate
    labelnp: numpy array of label

    """
    tracex, tracey, tracet, target, label = get_data()
    return np.array(tracex), np.array(tracey), np.array(tracet), np.array(target), np.array(label)

def minus_data():
    tracex, tracey, tracet, target, label = raw_data()
    new_x = []
    new_y = []
    for eachx, eachy, eachtarget in zip(tracex, tracey, target):
        traceminusx = np.array(eachx).transpose() - eachtarget[0]
        traceminusy = np.array(eachy).transpose() - eachtarget[1]
        # x, y = item
            # x = x - eachtarget[0]
        new_x.append(traceminusx)
        new_y.append(traceminusy)
    return np.array(new_x), np.array(new_y), tracet, target, label

if __name__ == '__main__':
    minus_data()