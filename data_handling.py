"""
This module loads and preprocesses the data for training.

Vladimir Mikulik, 2019
"""

import numpy as np
import random
import string
import torch


class DataLoader():
    """
    A loader for .txt input files.
    """

    def __init__(self, filename, all_letters, seq_length,
                 device, verbose=False):
        """
        Imports the dataset.

        Assumes that paragraphs are divided
        using "\\n\\n" and reserves "$" to be the end-of-paragraph
        symbol.

        Args:
            filename (str): the path to the input .txt file.
            all_letters(str): only these letters are retained in the text
                after preprocessing.
            seq_length (int): the desired length of the minibatches.
            device (torch.device): the device to store the dataset in.
            verbose (bool): whether to print warnings and the progress.
        """

        full_text = open(filename).read()
        full_text = full_text.replace("\n\n", "$")  # end-of-paragraph symbol
        # filter out illegal letters from the text
        full_text = ''.join([c for c in full_text if c in all_letters])
        # chop up the text into smaller sequences, truncating at the end
        # to ensure equal-sized batch lengths
        self.seqs = [full_text[i-seq_length:i]
                     for i in range(seq_length, len(full_text), seq_length)]
        self.n_seqs = len(self.seqs)
        if verbose and self.n_seqs < 50:
            print("Warning: small dataset (less than 50 sequences).")
        self.all_letters = all_letters + "$"
        self.n_letters = len(self.all_letters)

        # precompute a list of training pairs for efficient retrieval
        # during training
        if verbose:
            print("precomputing data tensors...")
        self.data = []
        for i, seq in enumerate(self.seqs, 1):
            tensor = self.seq2tensor(seq, truncate=True).to(device)
            target = self.seq2target(seq).to(device)
            self.data.append((tensor, target))
            if verbose and i % (self.n_seqs // 5) == 0:
                print("{:.2f}%".format(i/len(self.seqs)*100))

        if verbose:
            print("100%: done")

    def seq2tensor(self, seq, truncate=False):
        """
        converts a string into an input feature tensor of shape

        seq_length, num_features

        Args:
            seq (str)
            truncate (bool): when True, removes the last letter
                before transforming into a tensor.
        """
        if truncate:
            # in training, we don't have the pair
            # to the last letter, so we discard it
            seq = seq[:-1]
        tensor = torch.zeros(len(seq), self.n_letters)
        for i, letter in enumerate(seq):
            tensor[i][self.all_letters.index(letter)] = 1
        return tensor

    def seq2target(self, seq):
        """
        Converts a string into a target index tensor of length self.seq_length
        """
        # we don't have the pair to the first letter, so we discard it
        target = seq[1:]
        indices = [self.all_letters.index(letter) for letter in target]
        return torch.LongTensor(indices)

    def fast_minibatch(self, size):
        """
        samples a minibatch from the precomputed data tensors.

        Returns:
            batch (torch.Tensor) one-hot encoded input batch
            target (torch.Tensor) index-encoded target batch

        Shapes:
            batch: (seq_length, batch_size, n_letters)
            target: (seq_length, batch_size)
        """
        idxs = np.random.randint(0, self.n_seqs, size)
        pairs = [self.data[i] for i in idxs]

        # this acts as 'unzip': https://stackoverflow.com/questions/12974474
        seqs, targets = zip(*pairs)

        return torch.stack(seqs, 1), torch.stack(targets, 1)
