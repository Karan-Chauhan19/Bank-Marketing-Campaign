'''
Author : Karan Chauhan
github : @Karan-Chauhan19
Organization : L.J University
'''

# config.py
from gpu_config.check import *
# Dataset configurations
DATASET = {
    "path": "/home/karan-chauhan/WorkStation/Project/Bank-Marketing-Campaign/Data/bank.csv",
    "train_split": 0.8,
    "validation_split": 0.1,
    "test_split": 0.1,
    "batch_size": 32,
    "shuffle": True,
}

# Model configurations
MODEL = {
    "type": "ANN",
    "layers": 5,
    "activation": "ReLU",
    "dropout_rate": 0.5,
    "input_dim": 38,
    "output_classes": 2,
}

# Training configurations
TRAINING = {
    "epochs": 100,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_function": "binary_crossentropy",
    "metrics": ["accuracy"],
}


# Hardware configurations
HARDWARE = {
    "use_gpu": GPU_Config().check_gpu_configration(),
    "device": "cuda:0",
}

# Experiment details
EXPERIMENT = {
    "name": "Bank-Marketing-Campaign",
    "description": "Understand customer behavior and preferences",
    "seed": 42,
}



