import numpy as np
import sys; sys.path.append("..")
from termcolor import cprint
import argparse

from data_utils import read_q1, read_q2, read_q3
from print_utils import dprint

from dmn import DMN
from nlp_utils import create_vocabulary


def dmn_test(task, max_epoch, n_hop):
    """
    Test the dmn model using the wanted task in the babi dataset

    Parameters
    ----------
    task : int
        Task to test on (between 1 and 20)
    max_epoch : int
        Maximum number of epochs
    n_hop : int
        Number of hops
    """

    if task == 1:
        context_train, question_train, answer_train = read_q1("train", "1k")
        context_test, question_test, answer_test = read_q1("test", "1k")
    elif task == 2:
        context_train, question_train, answer_train = read_q2("train", "1k")
        context_test, question_test, answer_test = read_q2("test", "1k")
    elif task == 3:
        context_train, question_train, answer_train = read_q3("train", "1k")
        context_test, question_test, answer_test = read_q3("test", "1k")

    vocabulary = create_vocabulary(context_train)
    n_hid = 100
    name = "test_task_{}".format(task)
    batch_size = 32
    dropout = 1.0
    
    model = DMN(n_hid, n_hop, vocabulary=vocabulary, name=name)
    model.train(context_train, question_train, answer_train, max_epoch=max_epoch,
                batch_size=batch_size, dropout=dropout)
    
    # accuracy = model.compute_accuracy(context_test, question_test, answer_test)

    # cprint("Testing accuracy : {0: .2f} %".format(accuracy * 100),
    #        "cyan", attrs=["bold"])

    i = 23

    # model.visualize_attention(context_test[i], question_test[i], answer_test[i],
    #                           max_answer_length=1)
    model.visualize_attention(context_train[i], question_train[i], answer_train[i],
                              max_answer_length=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=int, default=1, help="Task number")
    parser.add_argument("-e", type=int, default=25, help="Max number of epochs")
    parser.add_argument("-m", type=int, default=3, help="Number of memory hops")
    args = parser.parse_args()
    dmn_test(args.t, args.e, args.m)
