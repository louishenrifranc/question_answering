import numpy as np
import os
import sys; sys.path.append("..")
from termcolor import cprint
import argparse

from utils.data_utils import get_babi_data

from memn2n import MemN2N
from nlp_utils import create_vocabulary


def memn2n_test(task, max_epoch, n_hop, train):
    """
    Test the memn2n model using the wanted task in the babi dataset

    Parameters
    ----------
    task : int
        Task to test on (between 1 and 20)
    max_epoch : int
        Maximum number of epochs
    n_hop : int
        Maximum number of memory hops
    train : bool
        If the model should be trained
    """

    context_train, question_train, answer_train = get_babi_data(task, "1k", "train")
    context_test, question_test, answer_test = get_babi_data(task, "1k", "test")

    vocabulary = create_vocabulary(context_train)
    n_emb = 100
    n_mem = 50
    positional_encoding = True
    temporal_encoding = True
    weight_tying = "adjacent"
    random_context = 0.1

    batch_size = 32
    learning_rate = 0.005
    save_name = "memn2n_task_{}".format(task)
    early_stopping = False
    
    model = MemN2N(vocabulary, n_emb, n_hop, n_mem, positional_encoding, temporal_encoding, weight_tying, random_context)

    if train:
        model.train(context_train, question_train, answer_train, batch_size, max_epoch, learning_rate, save_name, early_stopping)
    else:
        model = model.load(save_name)

    test_accuracy = model.accuracy(context_test, question_test, answer_test)
    if test_accuracy > 0.5:
        model.save(save_name)
        
    cprint("Test accuracy : {0:<5.2f} %".format(
        test_accuracy * 100),
           "magenta", attrs=["bold"])

    i = 23
    name = "memn2n_attention_task_{}".format(task)

    model.visualize_attention(context_test[i], question_test[i], answer_test[i], name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=int, default=1, help="Task number")
    parser.add_argument("-e", type=int, default=100, help="Max number of epochs")
    parser.add_argument("-m", type=int, default=3, help="Number of memory hops")
    parser.add_argument("--train", action="store_true", default=False, help="If the model should be trained")
    args = parser.parse_args()
    memn2n_test(args.t, args.e, args.m, args.train)
    
