"""
This module handles the training loop of a model.

Vladimir Mikulik 2019
"""

import time
import string

import numpy as np
import torch
from torch.optim import Adam

from data_handling import DataLoader
from model import LSTM


def train(model, data, training_params, output_params):

    print_every = output_params.get('print every', 200)
    plot_every = output_params.get('plot every', 50)
    num_iterations = training_params.get('number of iterations', 2000)
    batch_size = training_params.get('batch size', 128)
    subseq_length = training_params.get('subsequence length', 100)
    optimiser = Adam(model.parameters(),
                     lr=training_params.get('learning rate', 0.0005))
    criterion = torch.nn.NLLLoss(ignore_index=-1)

    # helper functions
    def time_since(since):
        """
        returns a nicely-formatted string showing time elapsed.
        """
        now = time.time()
        s = now - since
        m = int(s // 60)
        s -= m * 60
        return "{:>3d}m {:>2.0f}s".format(m, s)

    def detach(state):
        """
        truncates backpropagation to the start of this subsequence
        """
        return [[s.detach() for s in layer] for layer in state]

    # training step
    def train_on_batch(input_tensor, target_tensor):
        """
        trains on the input batch. Note that this actually takes several
        optimisation steps if subseq_length < seq_length.

        The reason for this is to train not only on all-zero initialisations
        """

        total_loss = 0
        state = model.init_state(input_tensor)

        # slice input into shorter subsequences
        # that will have manageable gradients
        for i in range(0,
                       max(input_tensor.size(0) - subseq_length, 1),
                       subseq_length):

            subsequence_input = input_tensor[i:i+subseq_length]
            subsequence_target = target_tensor[i:i+subseq_length]

            # forward pass
            state = detach(state)
            output, state = model(subsequence_input, state)
            output = output.transpose(1, 2)
            loss = criterion(output, subsequence_target)

            # backward pass
            model.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()

        # make sure reported loss is invariant to seq_length and batch_size
        normalizing_const = input_tensor.size(0) * input_tensor.size(1)
        return total_loss / normalizing_const

    # main loop
    plotted_losses = []
    plotted_iterations = []
    iter_loss = 0
    model.train()
    start = time.time()
    for i in range(1, num_iterations+1):

        # take training step
        input_tensor, target_tensor = data.fast_minibatch(batch_size)
        loss = train_on_batch(input_tensor, target_tensor)

        # log metrics
        iter_loss += loss
        if i % print_every == 0:
            print("{} ({} {:.2f}%) loss: {:.7f}".format(
                time_since(start),
                i,
                i/num_iterations*100,
                iter_loss / plot_every))
        if i % plot_every == 0:
            plotted_losses.append(iter_loss / plot_every)
            plotted_iterations.append(i)
            iter_loss = 0

    return plotted_iterations, plotted_losses
