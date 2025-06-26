import numpy as np
import matplotlib.pyplot as plt


def plot(result):
    loss = result.history["loss"]
    acc = result.history["acc"]
    val_loss = result.history["val_loss"]
    val_acc = result.history["val_acc"]

    plt.figure(figsize=(13,4))

    plt.subplot(1,2,1)
    plt.plot(range(len(loss)),loss,label = "Train Loss")
    plt.plot(range(len(val_loss)),val_loss,label = "Validation Loss")
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(loc='upper right')

    plt.subplot(1,2,2)
    plt.plot(range(len(acc)),acc,label = "Train Accuracy")
    plt.plot(range(len(val_acc)),val_acc,label = "Validation Accuracy")
    plt.title('Model Acc')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()


from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
        else:
            print("Queue is empty")

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)



'''def check_iid(label):
    label_counts = np.sum(label, axis=0)

    plt.bar(range(len(label_counts)), label_counts)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Label Distribution in y_train (One-Hot Encoded)')
    plt.show()
    '''