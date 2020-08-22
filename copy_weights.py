#   ~~~ aydao ~~~~ 2020 ~~~
#
#   Copy over the weights (trainables in G, D, Gs) to another network
#
#   This file is functionally identical to `copy_weights.py` but removes the dependency
#   on custom code embedded in the dnnlib implementation.  
#   So, this file is portable, meaning it should "just work" with any StyleGAN2 repo
#   including the official StyleGAN2 repo and subsequent forks, etc., using dnnlib
#
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import sys, getopt, os

import numpy as np
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib import tfutil
from dnnlib.tflib.autosummary import autosummary

from training import dataset
from training import misc
import pickle
import argparse


# Code for this function is modified from a function embedded in the dnnlib network.py of pbaylies' StyleGAN2 repo
# Note well that the argument order is target then source  
def copy_compatible_trainables_from(dst_net, src_net, filters) -> None:
    """Copy the compatible values of all trainable variables from the given network, including sub-networks"""
    names = []
    for name in dst_net.trainables.keys():
        name_match = [x in name for x in filters]
        if any(name_match):
            if name not in src_net.trainables:
                print("Not restoring (not present):     {}".format(name))
            elif dst_net.trainables[name].shape != src_net.trainables[name].shape:
                print("Not restoring (different shape): {}".format(name))
            elif name in src_net.trainables and dst_net.trainables[name].shape == src_net.trainables[name].shape:
                print("Restoring: {}".format(name))
                names.append(name)
            
    tfutil.set_vars(tfutil.run({dst_net.vars[name]: src_net.vars[name] for name in names}))

import re

def extract_conv_names(model):
    # layers are G_synthesis/{res}x{res}...
    # level 1 is either: Conv0_up, Const
    # Level 2 is: Conv1, ToRGB

    model_names = list(model.trainables.keys())
    conv_names = []
    #layer 4x4
    regex = r"G_synthesis\/(4x4)\/Co"
    for name in model_names:
        match = re.search(regex, name)
        if match:
            resolution = match.group(1)
            conv_names.append((name, resolution, None))

    # layers 8x8 and above
    regex = r"G_synthesis\/(\d+x\d+)\/Conv(\d)"
    for name in model_names:
        match = re.search(regex, name)
        if match:
            resolution = match.group(1)
            level = match.group(2)
            conv_names.append((name, resolution, level))

    return conv_names

import math

def blend_models(model_1, model_2, mid_point, blend_width=None):

    # y is the blending amount which y = 0 means all model 1, y = 1 means all model_2

    # TODO add small x offset for smoother blend animations
    # TODO blend weights and biases the same amount
    resolution, level = mid_point.split("-")
    
    model_1_names = extract_conv_names(model_1)
    model_2_names = extract_conv_names(model_2)

    assert all((x == y for x, y in zip(model_1_names, model_2_names)))

    model_out = model_1.clone()

    for name in model_1_names:
        print(name)

    short_names = [(x[1:]) for x in model_1_names]
    full_names = [(x[0]) for x in model_1_names]
    mid_point_index = short_names.index((resolution, level))
    
    ys = []
    for idx, name in enumerate(model_1_names):
        # low to high (res)
        x = idx - mid_point_index
        if blend_width:
            exponent = -blend_width*x
            y = 1 / (1 + math.exp(exponent))
        else:
            y = 1 if x > 1 else 0

        ys.append(y)
        print(f"Blending {name} by {y}")

    # model_out.reset_trainables()
    tfutil.set_vars(
        tfutil.run(
            {model_out.vars[name]: (model_2.vars[name] * y + model_1.vars[name] * (1-y))
             for name, y 
             in zip(full_names, ys)}
        )
    )

    return model_out

import numpy as np

def test_blend():

    high_res_pkl = "nets/ukiyoe-1024-e-000000.pkl"
    low_res_pkl = "nets/ukiyoe-1024-e-000024.pkl"
    output_pkl = "test.pkl"

    tflib.init_tf()

    with tf.Session() as sess:
        with tf.device('/gpu:0'):
            low_res_G, low_res_D, low_res_Gs = pickle.load(open(low_res_pkl, 'rb'))
            high_res_G, high_res_D, high_res_Gs = pickle.load(open(high_res_pkl, 'rb'))

            # low_res_G.reset_trainables()

            out = blend_models(low_res_Gs, high_res_Gs, "16x16-1", blend_width=0.5)

            grid_size = (3, 3)
            grid_latents = np.random.randn(np.prod(grid_size), *out.input_shape[1:])
            grid_fakes = out.run(grid_latents, None, is_validation=True, minibatch_size=2)
            misc.save_image_grid(grid_fakes, "test.jpg", drange= [-1,1], grid_size=grid_size)

            # misc.save_pkl((out, targetD, targetGs), os.path.join('./', output_pkl))
            

def main(args, filters):

    source_pkl = args.source_pkl
    target_pkl = args.target_pkl
    output_pkl = args.output_pkl

    tflib.init_tf()

    with tf.Session() as sess:
        with tf.device('/gpu:0'):

            sourceG, sourceD, sourceGs = pickle.load(open(source_pkl, 'rb'))
            targetG, targetD, targetGs = pickle.load(open(target_pkl, 'rb'))
            
            print('Source:')
            sourceG.print_layers()
            sourceD.print_layers() 
            sourceGs.print_layers()
            
            print('Target:')
            targetG.print_layers()
            targetD.print_layers() 
            targetGs.print_layers()
            
            copy_compatible_trainables_from(targetG, sourceG, filters=filters)
            copy_compatible_trainables_from(targetD, sourceD, filters=filters)
            copy_compatible_trainables_from(targetGs, sourceGs, filters=filters)
            
            misc.save_pkl((targetG, targetD, targetGs), os.path.join('./', output_pkl))

        
if __name__ == '__main__':
    test_blend()
    # parser = argparse.ArgumentParser(description='Copy weights from one StyleGAN pkl to another', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('source_pkl', help='Path to the source pkl (weights copied from this one). This will *not* be overwritten or modified.')
    # parser.add_argument('target_pkl', help='Path to the target pkl (weights copied onto this one). This will *not* be overwritten or modified.')
    # parser.add_argument('--output_pkl', default='network-copyover.pkl', help='Path to the output pkl (source_pkl weights copied into target_pkl architecture)')
    # args = parser.parse_args()

    # filters = ['ToRGB', '1024', '512', '256', '128', '64', '32', '16', '8', '4']

    # # for n in range(11):
    # n = 8
    # print(filters[:n])
    # args.output_pkl = "afhq-mixed/" +  str(n) + "-rev.pkl"
    # main(args, filters[:n])

    # # examples
    # blend_models(model_1, model_2, "256-0", 1)