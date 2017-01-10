import numpy as np
import os
import sys; sys.path.append(".")
import tensorflow as tf
from termcolor import cprint
import pickle
import math
import matplotlib.pyplot as plt

from nlp_utils import load_embeddings, sentence2index, index2sentence, zero_pad, zero_pad_2d
from functions_utils import read_input_char
from print_utils import dprint
from tf_functions_utils import index_tensor, tensor_matrix_mul


class DMN(object):
    """
    Dynamic Memory Network (Kumar 2016)
    """

    def __init__(self, n_hid=100, n_hop=3, vocabulary=None, name="default"):
        """
        Parameters
        ----------
        n_hid : int (default : 100)
            Number of hidden dimensions
        n_hop : int (default : 3)
            Number of memory hops
        vocabulary : set of str or None (default : None)
            Reduced vocabulary to use
        name : str (default : "default")
            Name under which to save the files
        """

        self._n_hid = n_hid
        self._n_hop = n_hop
        self._name = name

        # load glove embeddings
        embeddings, self._word2index, self._index2word = load_embeddings(vocabulary=vocabulary)
        self._n_vocab = embeddings.shape[0]
        self._n_emb = embeddings.shape[1]

        cprint("Building Dynamic Memory Network with\n" +
               "n_hid = {}, n_hop = {}, n_emb = {}, n_vocab = {}".format(
                   self._n_hid, self._n_hop, self._n_emb, self._n_vocab),
               "magenta", attrs=["bold"])

        ## General values ##

        self._embeddings = tf.Variable(embeddings, name="embeddings")

        self._initialiser = tf.random_normal_initializer(stddev=0.1)

        self._dropout = tf.placeholder(tf.float32, name="dropout")

        ## Placeholders ##

        # context for the questions
        # size : batch_size x context_size x context_length
        self._context = tf.placeholder(tf.int32, [None, None, None], name="context")
        # length of each context sentence
        # size : batch_size x context_size
        self._context_length = tf.placeholder(tf.int32, [None, None], name="context_length")

        # question
        # size : batch_size x question_length
        self._question = tf.placeholder(tf.int32, [None, None], name="question")
        # question length
        # size : batch_size
        self._question_length = tf.placeholder(tf.int32, [None], name="question_length")

        # maximum length of the given answer
        self._max_answer_length = tf.placeholder(tf.int32, name="max_answer_length")

        # correct answer
        # size : batch_size x answer_length
        self._correct_answer = tf.placeholder(tf.int32, [None, None], name="correct_answer")

        ## Calculations ##

        # size : batch_size x context_size x n_hid
        encoded_context = self._encode_context(self._context, self._context_length)

        # size : batch_size x n_hid
        encoded_question = self._encode_question(self._question, self._question_length)

        # size : batch_size x n_hid
        memory, self._attention = self._compute_memory(encoded_context, encoded_question)

        # size : batch_size x max_answer_length x n_vocab, batch_size x max_answer_length
        logits, self._answer = self._generate_answer(memory, encoded_question)

        self._loss = self._compute_loss(logits, self._correct_answer, self._attention)

        self._accuracy = self._compute_accuracy(self._answer, self._correct_answer)


    def _encode_context(self, context, context_length):
        """
        Encodes the context using a GRU over each sentence and then combining
        them with a second GRU.

        Parameters
        ----------
        context : Tensor-3 of size batch_size x context_size x context_length
            Context to encode
        context_length : Tensor-2 of size batch_size x context_size
            True length of each sentence

        Returns
        -------
        Tensor-3 of size batch_size x context_size x n_hid
            Encoded context
        """

        with tf.variable_scope("context_sentence"):

            # reshape it so that the first dimension combines batch_size and context_size
            encoded_context = tf.reshape(context, [-1, tf.shape(context)[2]])

            # transform the sentence to embeddings
            encoded_context = tf.nn.embedding_lookup(self._embeddings, encoded_context)

            cell = tf.nn.rnn_cell.GRUCell(self._n_hid)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, self._dropout)

            _, encoded_context = tf.nn.dynamic_rnn(cell, encoded_context, dtype=tf.float32)

        with tf.variable_scope("context"):

            encoded_context = tf.reshape(encoded_context,
                                         [tf.shape(context)[0], tf.shape(context)[1], self._n_hid])

            cell = tf.nn.rnn_cell.GRUCell(self._n_hid)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, self._dropout)

            encoded_context, _ = tf.nn.dynamic_rnn(cell, encoded_context, dtype=tf.float32)

        return encoded_context

            
    def _encode_question(self, question, question_length):
        """
        Encodes the question using a GRU.

        Parameters
        ----------
        question : Tensor-2 of size batch_size x question_length
            Question to encode
        question_length : Tensor-1 of size batch_size
            True length of the question

        Returns
        -------
        Tensor-2 of size batch_size x n_hid
            Encoded question
        """

        with tf.variable_scope("question"):

            encoded_question = tf.nn.embedding_lookup(self._embeddings, question)

            cell = tf.nn.rnn_cell.GRUCell(self._n_hid)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, self._dropout)

            _, encoded_question = tf.nn.dynamic_rnn(cell, encoded_question, dtype=tf.float32)

        return encoded_question


    def _compute_gate(self, context, question, memory):
        """
        Computes the attention gates for all sentences of the context.

        Parameters
        ----------
        context : Tensor-3 of size batch_size x context_size x n_hid
            Encoded context
        question : Tensor-2 of size batch_size x n_hid
            Encoded question
        memory : Tensor-2 of size batch_size x n_hid
            Current state of the memory

        Returns
        -------
        Tensor-2 of size batch_size x context_size
            Attention gates for all context sentences
        """

        with tf.variable_scope("attention"):

            # expand the dimensions so that broadcasting is correctly done
            question = tf.expand_dims(question, 1)
            memory = tf.expand_dims(memory, 1)

            # size : batch_size x context_size x 4*n_hid
            Z = tf.concat(2, [context * question,
                              context * memory,
                              tf.abs(context - question),
                              tf.abs(context - memory)])

            W_1 = tf.get_variable("W_1", [4 * self._n_hid, self._n_hid],
                                  initializer=self._initialiser)
            b_1 = tf.get_variable("b_1", [self._n_hid],
                                  initializer=tf.constant_initializer(0.0))
            W_2 = tf.get_variable("W_2", [self._n_hid, 1],
                                  initializer=self._initialiser)
            b_2 = tf.get_variable("b_2", [1],
                                  initializer=tf.constant_initializer(0.0))

            gate = tf.tanh(tensor_matrix_mul(Z, W_1) + b_1)
            gate = tensor_matrix_mul(gate, W_2) + b_2
            gate = tf.reshape(gate, [tf.shape(gate)[0], tf.shape(gate)[1]])
            gate = tf.sigmoid(gate)

        return gate


    def _episode_step(self, episode_prev, context, gate):
        """
        Computes a single episode step.

        Parameters
        ----------
        episode_prev : Tensor-2 of size batch_size x n_hid
            Previous episode
        context : Tensor-2 of size batch_size x n_hid
            Current context
        gate : Tensor-1 of size batch_size
            Gate for the context

        Returns
        -------
        Tensor-2 of size batch_size x n_hid
            Current episode
        """

        gate = tf.expand_dims(gate, 1)

        with tf.variable_scope("episode_step"):
            W_r = tf.get_variable("W_r", [self._n_hid, self._n_hid],
                                  initializer=self._initialiser)
            U_r = tf.get_variable("U_r", [self._n_hid, self._n_hid],
                                  initializer=self._initialiser)
            b_r = tf.get_variable("b_r", [self._n_hid],
                                  initializer=tf.constant_initializer(0.0))
            W = tf.get_variable("W", [self._n_hid, self._n_hid],
                                initializer=self._initialiser)
            U = tf.get_variable("U", [self._n_hid, self._n_hid],
                                initializer=self._initialiser)
            b = tf.get_variable("b", [self._n_hid],
                                initializer=tf.constant_initializer(0.0))
            R = tf.sigmoid(tf.matmul(context, W_r) + tf.matmul(episode_prev, U_r) + b_r)
            E_tilde = tf.tanh(tf.matmul(context, W) + R * tf.matmul(episode_prev, U) + b)
            episode = gate * E_tilde + (1 - gate) * episode_prev

        return episode


    def _memory_hop(self, memory_prev, attention_prev, context, question, t):
        """
        Single hop of the memory.

        Parameters
        ----------
        memory_prev : Tensor-2 of size batch_size x n_hid
            Previous state of the memory
        attention_prev : Tensor-2 of size batch_size x context_size
            Previous attention
        context : Tensor-3 of size batch_size x context_size x n_hid
            Encoded context
        question : Tensor-2 of size batch_size x n_hid
            Encoded question
        t : int
            Index of the current hop

        Returns
        -------
        Tensor-2 of size batch_size x n_hid
            Current state of the memory
        Tensor-2 of size batch_size x context_size
            Attention gate for the hop
        """

        with tf.variable_scope("memory_hop", reuse=(t!=0)):

            gate = self._compute_gate(context, question, memory_prev)

            elems = (tf.reverse(tf.transpose(context, [1,0,2]), [True,False,False]),
                     tf.reverse(tf.transpose(gate), [True,False]))
            # elems = (tf.transpose(context, [1,0,2]),
            #          tf.transpose(gate), [True,False])

            initialiser = tf.zeros([tf.shape(context)[0], self._n_hid])

            episodes = tf.scan(lambda E, C_G: self._episode_step(E, C_G[0], C_G[1]),
                               elems, initializer=initialiser)

            X = tf.concat(1, [memory_prev, episodes[-1], question])

            b = tf.get_variable("b", [self._n_hid],
                                initializer=tf.constant_initializer(0.0))

        with tf.variable_scope("memory_hop_{}".format(t + 1)):

            W = tf.get_variable("W", [3 * self._n_hid, self._n_hid],
                                initializer=self._initialiser)
        
        memory = tf.nn.relu(tf.matmul(X, W) + b)

        gate = tf.reverse(gate, [True,False])

        return memory, gate


    def _compute_memory(self, context, question):
        """
        Computes the final state of the memory.

        Parameters
        ----------
        context : Tensor-3 of size batch_size x context_size x n_hid
            Encoded context
        question : Tensor-2 of size batch_size x n_hid
            Encoded question

        Returns
        -------
        Tensor-2 of size batch_size x n_hid
            Final state of the memory
        """

        with tf.variable_scope("memory"):

            initializer = (question, tf.zeros([tf.shape(context)[0], tf.shape(context)[1]]))

            elems = (tf.tile(tf.expand_dims(context, 0), [self._n_hop,1,1,1]),
                     tf.tile(tf.expand_dims(question, 0), [self._n_hop,1,1]),
                     tf.range(1, self._n_hop + 1))

            memory = question
            att = tf.zeros([tf.shape(context)[0], tf.shape(context)[1]])

            for t in range(self._n_hop):
                memory, att = self._memory_hop(memory, att, context, question, t)
                if t == 0:
                    attention = tf.expand_dims(att, 0)
                else:
                    attention = tf.concat(0, [attention, tf.expand_dims(att, 0)])

            # memories, attention = tf.scan(lambda M_A, C_Q_t: self._memory_hop(M_A[0], M_A[1], C_Q_t[0], C_Q_t[1], C_Q_t[2]),
            #                               elems, initializer)

            # memory = memories[-1,:,:]
            attention = tf.transpose(attention, [1,0,2])

        return memory, attention


    def _answer_step(self, H_prev, Y_prev, y_prev, question):
        """
        Computes a single step for the answer.

        Parameters
        ----------
        H_prev : Tensor-2 of size batch_size x n_hid
            Previous hidden outputs
        Y_prev : Tensor-2 of size batch_size x n_vocab
            Previous logits
        y_prev : Tensor-1 of size batch_size
            Previous answer
        question : Tensor-2 of size batch_size x n_hid
            Encoded question

        Returns
        -------
        Tensor-2 of size batch_size x n_hid
            Current hidden outputs
        Tensor-2 of size batch_size x n_vocab
            Current logits
        Tensor-1 of size batch_size
            Current answer
        """

        X = tf.concat(1,[tf.nn.embedding_lookup(self._embeddings, y_prev),
                         question])
        X = tf.stop_gradient(X)

        with tf.variable_scope("answer_step"):

            W_z = tf.get_variable("W_z", [self._n_emb + self._n_hid, self._n_hid],
                                  initializer=self._initialiser)
            U_z = tf.get_variable("U_z", [self._n_hid, self._n_hid],
                                  initializer=self._initialiser)
            b_z = tf.get_variable("b_z", [self._n_hid],
                                  initializer=tf.constant_initializer(0.0))
            W_r = tf.get_variable("W_r", [self._n_emb + self._n_hid, self._n_hid],
                                  initializer=self._initialiser)
            U_r = tf.get_variable("U_r", [self._n_hid, self._n_hid],
                                  initializer=self._initialiser)
            b_r = tf.get_variable("b_r", [self._n_hid],
                                  initializer=tf.constant_initializer(0.0))
            W = tf.get_variable("W", [self._n_emb + self._n_hid, self._n_hid],
                                initializer=self._initialiser)
            U = tf.get_variable("U", [self._n_hid, self._n_hid],
                                initializer=self._initialiser)
            b = tf.get_variable("b", [self._n_hid],
                                initializer=tf.constant_initializer(0.0))
            Z = tf.sigmoid(tf.matmul(X, W_z) + tf.matmul(H_prev, U_z) + b_z)
            R = tf.sigmoid(tf.matmul(X, W_r) + tf.matmul(H_prev, U_r) + b_r)
            H_tilde = tf.sigmoid(tf.matmul(X, W) + R * tf.matmul(H_prev, U) + b)
            H = Z * H_tilde + (1 - Z) * H_prev

            W_p = tf.get_variable("W_p", [self._n_hid, self._n_vocab],
                                  initializer=self._initialiser)
            b_p = tf.get_variable("b_p", [self._n_vocab],
                                  initializer=tf.constant_initializer(0.0))

            Y = tf.matmul(H, W_p) + b_p
            y = tf.to_int32(tf.argmax(Y, 1))

        return H, Y, y


    def _generate_answer(self, memory, question):
        """
        Generates the answer using a GRU.

        Parameters
        ----------
        memory : Tensor-2 of size batch_size x n_hid
            Final state of the memory
        question : Tensor-2 of size batch_size x n_hid
            Encoded_question

        Returns
        -------
        Tensor-3 of size batch_size x max_answer_length x n_vocab
            Logits
        Tensor-2 of size batch_size x max_answer_length
            Answer
        """

        with tf.variable_scope("answer"):

            question = tf.tile(tf.expand_dims(question, 0), [self._max_answer_length, 1, 1])

            initialiser = (memory,
                           tf.zeros([tf.shape(memory)[0], self._n_vocab]),
                           tf.zeros([tf.shape(memory)[0]], dtype=tf.int32))

            _, logits, answer = tf.scan(lambda H_Y_y, Q: self._answer_step(H_Y_y[0], H_Y_y[1], H_Y_y[2], Q),
                                        question, initializer=initialiser)

            logits = tf.transpose(logits, [1,0,2])
            answer = tf.transpose(answer)

        return logits, answer


    def _compute_loss(self, logits, correct_answer, attention):
        """
        Computes the cross entropy loss.

        Parameters
        ----------
        logits : Tensor-3 of shape batch_size x max_answer_length x n_vocab
            Logits
        correct_answer : Tensor-2 of size batch_size x max_answer_length
            Correct answer
        attention : Tensor-2 of size batch_size x n_hop
            Attention gates

        Returns
        -------
        Tensor-0
            Loss
        """

        logits = tf.reshape(logits, [-1, self._n_vocab])
        correct_answer = tf.reshape(correct_answer, [-1])

        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, correct_answer))

        loss += 1e-5 * tf.reduce_sum(tf.abs(attention))

        return loss


    def _compute_accuracy(self, answer, correct_answer):
        """
        Computes the exact word accuracy.

        Parameters
        ----------
        answer : Tensor-2 of size batch_size x max_answer_length
            Generated answer
        correct_answer : Tensor-2 of size batch_size x max_answer_length
            Correct answer

        Returns
        -------
        Tensor-0
            Accuracy rate
        """

        answer = tf.reshape(answer, [-1])
        correct_answer = tf.reshape(correct_answer, [-1])

        accuracy = tf.reduce_mean(tf.to_float(tf.equal(answer, correct_answer)))

        return accuracy

    
    def _prepare_data(self, context, question, answer=None, batch_size=None):
        """
        Prepares the data for the model to use.

        Parameters
        ----------
        context : list-1 of str or list-2 of str
            List of context sentences
        question : str or list-1 of str
            Question string
        answer : str or list-1 of str or None (default : None)
            Answer string if wanted
        batch_size : int or None (default : None)
            Size of each minibatch if used
        
        Returns
        -------
        list-2 of int or list-3 of int
            Transformed and 0-padded context
        list-1 of int or list-2 of int
            True length of each context
        list-1 of int or list-2 of int
            Transformed and 0-padded question
        int or list-1 of int
            True length of each question
        list-1 of int or list-2 of int (optional)
            Transformed and 0-padded answer if asked for
        """

        cprint("[*] Preparing data...", "yellow", end="\r")

        # if this for only one example
        if isinstance(question, str):

            context = sentence2index(context, self._word2index, end_token=True)
            context_length = [len(c) for c in context]
            
            question = sentence2index([question], self._word2index, end_token=True)[0]
            question_length = len(question)

            if answer is not None:
                answer = sentence2index([answer], self._word2index, end_token=False)[0]

        else:

            context = sentence2index(context, self._word2index, end_token=True)
            context, context_length = zero_pad_2d(context, pad="left",
                                                  batch_size=batch_size, length=True)

            question = sentence2index(question, self._word2index, end_token=True)
            question, question_length = zero_pad(question, pad="left",
                                                 batch_size=batch_size, length=True)

            if answer is not None:
                answer = sentence2index(answer, self._word2index, end_token=False)
                answer = zero_pad(answer, pad="right", batch_size=batch_size)

        cprint("[+] Data prepared.   ", "cyan")

        if answer is None:
            return context, context_length, question, question_length
        else:
            return context, context_length, question, question_length, answer
            

    def train(self, context, question, answer, batch_size=128, max_epoch=25,
              dropout=1.0):
        """
        Trains the model.

        Parameters
        ----------
        context : list of list of str
            Context sentences
        question : list of str
            Question sentences
        answer : list of str
            Answer sentences
        batch_size : int (default : 128)
            Size of each minibatch
        max_epoch : int (default : 25)
            Maximum number of epochs
        dropout : float (default : 1.0)
            Dropout keep rate
        """

        optimizer = tf.train.AdamOptimizer()
        grads = optimizer.compute_gradients(self._loss)
        capped_grads = [(tf.clip_by_norm(grad, 50), var) for grad, var in grads]
        training_step = optimizer.apply_gradients(capped_grads)

        saver = tf.train.Saver(max_to_keep=1)
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "models")

        if self._name is None:
            save_path = os.path.join(save_path, "test.dmn_model")
        else:
            save_path = os.path.join(save_path, "{}.dmn_model".format(self._name))

        if os.path.exists(save_path):
            cprint("There is a saved model under the given name.\n" +
                   "Would you like to load it? (y/n)",
                   "cyan", attrs=["bold"])
            user_input = read_input_char()
            while user_input != "y" and user_input != "n":
                cprint("I'm sorry, but I didn't get a valid answer.\n" +
                       "Would you like to load the model? (y/n)",
                       "cyan", attrs=["bold"])
                user_input = read_input_char()
            if user_input == "y":
                success = self.load()
                if success:
                    cprint("Do you want to keep training the model? (y/n)",
                           "cyan", attrs=["bold"])
                    user_input = read_input_char()
                    while user_input != "y" and user_input != "n":
                        cprint("I'm sorry, but I didn't get a valid answer.\n" +
                               "Do you want to keep training the model? (y/n)",
                               "cyan", attrs=["bold"])
                        user_input = read_input_char()
                    if user_input == "n":
                        return
                else:
                    self._sess = tf.Session()
                    self._sess.run(tf.global_variables_initializer())
            else:
                self._sess = tf.Session()
                self._sess.run(tf.global_variables_initializer())
        else:
            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())

        cprint("Training model...", "magenta", attrs=["bold"])

        # randomize the examples
        random_indices = np.random.permutation(len(context))
        context = [context[i] for i in random_indices]
        question = [question[i] for i in random_indices]
        answer = [answer[i] for i in random_indices]

        context_train = context[:int(0.9 * len(context))]
        question_train = question[:int(0.9 * len(question))]
        answer_train = answer[:int(0.9 * len(answer))]

        context_valid = context[int(0.9 * len(context)):]
        question_valid = question[int(0.9 * len(question)):]
        answer_valid = answer[int(0.9 * len(answer)):]

        (context_train, context_length_train,
         question_train, question_length_train,
         answer_train) = self._prepare_data(context_train,
                                            question_train,
                                            answer_train,
                                            batch_size=batch_size)

        (context_valid, context_length_valid,
         question_valid, question_length_valid,
         answer_valid) = self._prepare_data(context_valid,
                                            question_valid,
                                            answer_valid,
                                            batch_size=batch_size)

        n_train_batches = math.ceil(len(context_train) / batch_size)
        n_valid_batches = math.ceil(len(context_valid) / batch_size)

        epoch = 0
        train_losses = []
        train_accuracies = []
        valid_accuracies = [0]

        while epoch < max_epoch:
            epoch += 1

            for i in range(n_train_batches):

                feed_dict = {self._context: context_train[i * batch_size: (i+1) * batch_size],
                             self._context_length: context_length_train[i * batch_size: (i+1) * batch_size],
                             self._question: question_train[i * batch_size: (i+1) * batch_size],
                             self._question_length: question_length_train[i * batch_size: (i+1) * batch_size],
                             self._correct_answer: answer_train[i * batch_size: (i+1) * batch_size],
                             self._max_answer_length: len(answer_train[i * batch_size]),
                             self._dropout: dropout}
                
                loss, accuracy, _ = self._sess.run([self._loss,
                                                    self._accuracy,
                                                    training_step],
                                                   feed_dict)
                train_losses.append(loss)
                train_accuracies.append(accuracy)

                cprint("Epoch {0:>3}, Batch {1:>2} / {2:}, Loss {3:>10.6f}, Accuracy {4:>6.2f} %, Valid Accuracy {5:>6.2f} %".format(
                    epoch, i + 1, n_train_batches,
                    np.mean(train_losses[-batch_size:]),
                    np.mean(train_accuracies[-batch_size:]) * 100,
                    valid_accuracies[-1] * 100),
                       "yellow", end="\r")

            accuracy = 0
            for i in range(n_valid_batches):
                
                feed_dict = {self._context: context_valid[i * batch_size: (i+1) * batch_size],
                             self._context_length: context_length_valid[i * batch_size: (i+1) * batch_size],
                             self._question: question_valid[i * batch_size: (i+1) * batch_size],
                             self._question_length: question_length_valid[i * batch_size: (i+1) * batch_size],
                             self._correct_answer: answer_valid[i * batch_size: (i+1) * batch_size],
                             self._max_answer_length: len(answer_valid[i * batch_size]),
                             self._dropout: 1.0}

                accuracy += self._sess.run(self._accuracy,
                                           feed_dict)

            accuracy /= n_valid_batches

            if epoch == 1:
                valid_accuracies = [accuracy]
            else:
                valid_accuracies.append(accuracy)

            if epoch > 50 and np.mean(valid_accuracies[-10:]) <= np.mean(valid_accuracies[-30:-10]):
                break

        saver.save(self._sess, save_path)

        cprint("\nTraining done.", "green", attrs=["bold"])


    def load(self):
        """
        Loads the saved model.

        Returns
        -------
        bool
             If the encoder was loaded or not
        """

        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "models")

        if self._name is None:
            save_path = os.path.join(save_path, "test.dmn_model")
        else:
            save_path = os.path.join(save_path, "{}.dmn_model".format(self._name))

        if os.path.exists(save_path + ".index"):

            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            saver.restore(self._sess, save_path)

            cprint("The encoder was loaded.", "green", attrs=["bold"])

            return True

        else:

            cprint("There were no encoder under the given name.", "magenta")

            return False


    def visualize_attention(self, context, question, answer, max_answer_length=3):
        """
        Creates an image with the attention matrix

        Parameters
        ----------
        context : list of str
             List of context sentences
        question : str
             Question
        answer : str
             Answer
        
        max_answer_length : int (default : 3)
            Maximum length of the answer

        Returns
        -------
        Nothing
        """

        if not hasattr(self, "_sess"):
            eprint("The model has to be trained first.")
            sys.exit(1)

        graph_labels = context
        question_label = question
        answer_label = answer

        (context, context_length,
         question, question_length) = self._prepare_data([context], [question], batch_size=None)

        feed_dict = {self._context: context,
                     self._context_length: context_length,
                     self._question: question,
                     self._question_length: question_length,
                     self._max_answer_length: max_answer_length,
                     self._dropout: 1}

        answer, attention = self._sess.run([self._answer, self._attention], feed_dict)

        answer = index2sentence(answer, self._index2word)[0]

        attention = attention[0, :, :].T

        save_location = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "results", "dmn_attention_{}.svg".format(self._name))

        fig = plt.figure()
        ax = plt.subplot(211)
        ax.pcolor(attention, cmap=plt.cm.Blues, alpha=0.8)

        fig = plt.gcf()

        ax.set_frame_on(False)
        
        ax.set_xticks(np.arange(attention.shape[1]) + 0.5, minor=False)
        ax.set_yticks(np.arange(attention.shape[0]) + 0.5, minor=False)

        ax.xaxis.tick_top()

        ax.set_xticklabels(np.arange(self._n_hop) + 1, minor=False)
        ax.set_yticklabels(graph_labels, minor=False, fontsize=10)

        ax.grid(False)

        ax.tick_params(axis="both", which="both", length=0)

        ax = plt.subplot(212)

        ax.text(0.5, 0.67, question_label, horizontalalignment="center")
        ax.text(0.25, 0.33, answer_label, horizontalalignment="center")
        ax.text(0.75, 0.33, answer, horizontalalignment="center")

        ax.set_frame_on(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.tight_layout()

        plt.savefig(save_location)
