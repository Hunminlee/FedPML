import numpy as np
import pandas as pd
import random



def non_iid_case1_label(data, label):
    int_label = one_hot_to_label(label)
    num_classes = 11
    P = [random.random() for _ in range(num_classes)]

    x, y = [], []

    for i in range(len(data)):
        for cls in range(num_classes):
            if int_label[i] == cls:
                if random.random() < P[cls]:
                    x.append(data[i])
                    y.append(label[i])

    return x, y


def non_iid_case2_data(data, label):
    probabilities = [random.random() for _ in range(len(data))]

    new_data = [(element, lab) for element, lab, prob in zip(data, label, probabilities) if random.random() < prob]
    x, y = zip(*new_data)

    return list(x), list(y)


def non_iid_case3(x_data, y_data, z_data):

    random_indices = np.random.choice(len(x_data), size=10000, replace=False)  #여기서 많이 뽑고 그 다음에 1000 개 거름

    x_tr, y_tr, z_tr = x_data[random_indices], y_data[random_indices], z_data[random_indices]

    X, Y = [], []

    my_list = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]
    selected_elements = random.sample(my_list, 2)
    if selected_elements[0] > selected_elements[1]:
        big, small = selected_elements[0], selected_elements[1]
    else:
        small, big = selected_elements[0], selected_elements[1]

    #print(len(z_tr))

    for i in range(len(z_tr)):
        if z_tr[i] >= small and z_tr[i] <= big:
            if len(X) > 1000:
                break
            X.append(x_tr[i])
            Y.append(y_tr[i])

    return X, Y