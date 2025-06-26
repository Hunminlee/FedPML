import os
import tensorflow as tf

#repetitions = 5
sub_lst = [10, 3, 8, 1, 9]

meta_step_size = 0.5
meta_iters = 100
shots = 5
num_classes = 5
train_init_rep, test_rep = [1,3,4,6,8,9,10], [2,5,7]
Feature_idx = 0

K_shot = 5
sub_acc = []

alpha = 0.25

optimizer = tf.keras.optimizers.Adam()