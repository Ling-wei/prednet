'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *

def make_gif(images1, images2,fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    assert len(images1)==len(images2 ),'len(images1) must equal len(images2) '
    try:
      x1 = images1[int(len(images1)/duration*t)]
      x2 = images2[int(len(images2) / duration * t)]
      x = np.concatenate((x1,x2))
    except:
      x = images1[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images1) / duration)

n_plot = 20
batch_size = 10
nt = 10

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
dim_ordering = layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(input=inputs, output=predictions)

test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', dim_ordering=dim_ordering)
X_test = test_generator.create_all()
X_hat = test_model.predict(X_test, batch_size)
if dim_ordering == 'th':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
f = open(RESULTS_SAVE_DIR + 'prediction_scores.txt', 'w')
f.write("Model MSE: %f\n" % mse_model)
f.write("Previous Frame MSE: %f" % mse_prev)
f.close()

# Plot some predictions
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (nt, 2*aspect_ratio))
gs = gridspec.GridSpec(2, nt)
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
# for i in plot_idx:
#     for t in range(nt):
#         plt.subplot(gs[t])
#         plt.imshow(X_test[i,t], interpolation='none')
#         plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
#         if t==0: plt.ylabel('Actual', fontsize=10)
#
#         plt.subplot(gs[t + nt])
#         plt.imshow(X_hat[i,t], interpolation='none')
#         plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
#         if t==0: plt.ylabel('Predicted', fontsize=10)
#
#     plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
#     plt.clf()
gif_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_gifs/')
if not os.path.exists(gif_save_dir): os.mkdir(gif_save_dir)
for i in plot_idx:
  make_gif(X_test[i],X_hat[i],gif_save_dir+str(i)+'gt.gif',nt)


