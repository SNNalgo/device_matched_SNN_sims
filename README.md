# device_matched_SNN_sims
This is an **snntorch** based implementation of SNN training with real spiking neuron device characteristics approximated by suitable parameters of an LIF model. A 10-layer spiking CNN is trained on CIFAR-10 data with 8-bit quantized weights

## Requirements
Pytorch, numpy, snntorch, matlab

## Description
1. **Spiking_CNN_quant_training_with_theta_beta_variations_8b_2.py** - trains a 10-layer Spiking convolutional neural network training with 8-bit weights for CIFAR-10
2. **curr_driven_SNNmodeling.py** - models spiking neuron device characteristics as LIF neurons, Capacitance determined by inspection as of now
3. **iris_classification_matlab** - old matlab code for training 1 layer SNN for Fisher-iris classification with STDP, main script - testScript3.m
