import numpy as np
import sys; sys.path.append("..")
from termcolor import cprint
import argparse

from utils.data_utils import get_babi_data

from lstm import LSTM
from nlp_utils import create_vocabulary


def lstm_test(task, max_epoch):

    context_train, question_train, answer_train = get_babi_data(task, "1k", "train")
    context_test, question_test, answer_test = get_babi_data(task, "1k", "test")

    vocabulary = create_vocabulary(context_train)

    batch_size = 32
    learning_rate = 0.01
    save_name = "lstm_task_{}".format(task)

    model = LSTM(vocabulary, n_hid)

    model.train(context_train, question_train, answer_train, batch_size, max_epoch, learning_rate, save_name)
