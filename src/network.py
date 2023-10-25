"""Neural Network class"""
import numpy as np
import torch


class NeuralNetowrk:
    def __init__(self, input_shape: tuple, n_outputs: int, lr: float = 0.001,):
        self._input_shape = input_shape
        self._n_outputs = n_outputs
        self._lr = lr
        self.loss = torch.nn.L1Loss()

    def train(self, train_data, n_episodes: int):
        NotImplemented

    def eval(self, test_data):
        NotImplemented

    # convert sample to pytorch format
    def process_samples(self):
        NotImplemented

    def reset(self):
        NotImplemented
