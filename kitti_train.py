'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import numpy as np
np.random.seed(123)
#from six.moves import cPickle

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *


save_model = True                           # if weights will be saved
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')

# Data files
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

# Training parameters
nb_epoch = 150                              # number of epoch
batch_size = 4
samples_per_epoch = 500
N_seq_val = 100                             # number of sequences to use for validation

# Model parameters
nt = 10                                     # 训练序列长度
n_channels, im_height, im_width = (3, 128, 160)
input_shape = (n_channels, im_height, im_width) if K.image_dim_ordering() == 'th' else (im_height, im_width, n_channels)# theano or tensorflow,这里应该改一下!实际用的tf,th:channel first
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0.])             # 各层预测误差权重,目标是最底层重建误差最小，其它层不关心 shape:(4,0) 行向量
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)  # shape:(4,1)，在第二个方向插入轴，即变列向量
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))          # 各时刻的预测误差的权重
time_loss_weights[0] = 0


prednet = PredNet(stack_sizes, R_stack_sizes,                   # 初始化一个Prednet网络
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape)       # 定义输入张量形状（batch_size,序列长，img_row,img_col,img_channels）
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers) 计算各层A与Ahat误差？运行topology中Layer.__call__
# TimeDistributed包装器可以把一个层应用到输入的每一个时间步上，输入参数是一个层，至少3D，第一维是时间，(nb_samples, input_dim)
errors_by_time = TimeDistributed(Dense(1, weights=[layer_loss_weights, np.zeros(1)], trainable=False), trainable=False)(errors)  # calculate weighted error by layer 一个不训练有权重的dense层，实际就是给各Layer的loss加权
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt) 对batch中每个样本，展平成一维向量
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time，一个全连接层，为各时刻error加权重
model = Model(input=inputs, output=final_errors)    # 不同于model = Sequential(),直接用的泛型模型Model
model.compile(loss='mean_absolute_error', optimizer='adam')

#train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)    # 太大了跑不动
train_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, shuffle=True)         # 产生训练数据
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)

lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]    # 回调函数：学习率调度器，以epoch号为参数（从0算起的整数），返回一个新学习率（浮点数）
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))      # 使用回调函数来观察训练过程中网络内部的状态和统计信息

history = model.fit_generator(train_generator, samples_per_epoch, nb_epoch, callbacks=callbacks,            # 与fit功能类似，利用Python的生成器，逐个生成数据的batch并进行训练，速度快
                    validation_data=val_generator, nb_val_samples=N_seq_val)

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
