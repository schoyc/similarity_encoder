#!/bin/bash

# These first
python train_encoder.py -1 # Use all transforms
python train_encoder.py 5 # All but brightness
python train_encoder.py 6 # All but contrast

# These if we have time
python train_encoder.py 0 # All but uniform
python train_encoder.py 1 # All but translate
python train_encoder.py 2 # All but rotate
python train_encoder.py 3 # All but pixel scale
python train_encoder.py 4 # All but crop resize
python train_encoder.py 7 # All but Gaussian
python train_encoder.py 8 # All but Poisson
