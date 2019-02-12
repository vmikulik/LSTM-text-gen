"""
This module implements an LSTM with gradient clipping and dropout.

Vladimir Mikulik, 2019
"""

import torch
import torch.nn as nn


def clip_grad_on_backprop(x, min, max):
    # instructs autograd to clamp the value of the gradient
    # to the specified range in the midst of backprop.
    # this is needed to stabilise backpropagation-through-time.
    if x.requires_grad:
        x.register_hook(lambda grad: grad.clamp(min, max))
    return x


class LSTMCell(nn.Module):
    """
    A wrapper for torch.nn.LSTMCell with dropout and gradient clipping.
    """
    def __init__(self, n_input, n_hidden, dropout, grad_clipping=None):
        """
        Args:
            n_input (int): input width.
            n_hidden (int): hidden width.
            dropout (float in [0,1]): dropout probability after cell.
            grad_clipping (int or None): if not None, the gradient
                through this module during backprop will be clipped
                to values in [-grad_clipping, grad_clipping].
                This aids convergence in backpropagation through time.
        """
        super(LSTMCell, self).__init__()
        self.grad_clipping = grad_clipping
        self.cell = nn.LSTMCell(n_input, n_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch, state):
        hidden, cell = self.cell(batch, state)
        if self.grad_clipping is not None:
            hidden = clip_grad_on_backprop(hidden,
                                           -self.grad_clipping,
                                           self.grad_clipping)
        hidden = self.dropout(hidden)
        state = (hidden, cell)
        return hidden, state


class LSTM(nn.Module):
    """
    An LSTM net with a logsoftmax head.
    Supports dropout and gradient clipping during backprop.

    Args: (when applied)
        batch (tensor): the input minibatch
        state ((tensor, tensor)): the state of the module.

    input shapes:
        batch:  (seq_len, batch_size, n_input)
        state:  list of (hidden, cell) where
            hidden:   (batch_size, hidden_size)
            cell:     (batch_size, hidden_size)
            len(state) == n_layers

    output shapes:
         batch: (seq_len, batch_size, n_output),
         state: as above

    """
    def __init__(self,
                 n_input,
                 n_hidden,
                 n_output,
                 n_layers,
                 dropout=0.5,
                 grad_clipping=10):
        """
        Args:
            n_input (int): input width.
            n_hidden (int): hidden width.
            n_output (int): output width.
            n_layers (int): number of LSTM layers. Note that there
                is an extra torch.nn.Linear layer at the end to convert
                from the hidden width to the output width.
            dropout (float in [0,1]): dropout probability after cell.
            grad_clipping (int or None): if not None, the gradient
                through this module during backprop will be clipped
                to values in [-grad_clipping, grad_clipping].
                This aids convergence in backpropagation through time.
        """

        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers

        self.cells = []
        for i in range(n_layers):
            cell = LSTMCell(n_hidden if i > 0 else n_input,
                            n_hidden,
                            dropout,
                            grad_clipping)
            self.cells.append(cell)
            setattr(self, 'cell_{}'.format(i), cell)

        self.linear = nn.Linear(n_hidden, n_output)
        self.softmax = nn.LogSoftmax(2)

    def forward_pre_softmax(self, batch, state):
        """
        computes a forward pass through the RNN, but does not
        apply softmax at the end. Useful for adjusting sampling
        temperature.

        input shapes:
            batch:  (seq_len, batch_size, n_input)
            state:  list of (hidden, cell) where
                hidden:   (n_layers, batch_size, hidden_size)
                cell:     (n_layers, batch_size, hidden_size)
                len(state) == n_layers

        output shapes:
             batch: (seq_len, batch_size, n_output),
             state: list of (hidden, cell) where
                hidden:   (batch_size, hidden_size)
                cell:     (batch_size, hidden_size)
                len(state) == n_layers
        """
        seq_len = batch.size(0)
        batch_size = batch.size(1)

        output = batch.new_zeros(seq_len, batch_size, self.n_hidden)

        # run input through LSTM layers
        for t in range(seq_len):
            x = batch[t]
            for l in range(self.n_layers):
                x, state[l] = self.cells[l](x, state[l])
            output[t] = x

        # resize from LSTM layer width to output width
        output = self.linear(output)
        return output, state

    def forward(self, batch, state):
        """
        computes a forward pass through the RNN.

        input shapes:
            batch:  (seq_len, batch_size, n_input)
            state:  list of (hidden, cell) where
                hidden:   (batch_size, hidden_size)
                cell:     (batch_size, hidden_size)
                len(state) == n_layers

        output shapes:
             batch: (seq_len, batch_size, n_output),
             state: list of (hidden, cell) where
                hidden:   (batch_size, hidden_size)
                cell:     (batch_size, hidden_size)
                len(state) == n_layers
        """

        output, state = self.forward_pre_softmax(batch, state)
        output = self.softmax(output)
        return output, state

    def sample(self, batch, state, temperature=1):
        """
        computes a the RNN's predicitons on the batch.

        input shapes:
            batch:  (seq_len, batch_size, n_input)
            state:  list of (hidden, cell) where
                hidden:   (batch_size, hidden_size)
                cell:     (batch_size, hidden_size)
                len(state) == n_layers

        output shapes:
             batch: (seq_len, batch_size, n_output),
             state: list of (hidden, cell) where
                hidden:   (batch_size, hidden_size)
                cell:     (batch_size, hidden_size)
                len(state) == n_layers
        """
        output, state = self.forward_pre_softmax(batch, state)
        output = self.softmax(output / temperature)
        return output, state

    def init_state(self, batch):
        """
        creates an initial state object, with shape matching batch
        and all values set to zeros, ready to be fed into LSTM.forward.

        Args:
            batch (torch.Tensor): the batch that we want to match state to.
        """

        batch_size = batch.size(1)

        state = []
        for l in range(self.n_layers):
            hidden = batch.new_zeros(batch_size, self.n_hidden)
            cells = batch.new_zeros(batch_size, self.n_hidden)
            state.append((hidden, cells))
        return state
