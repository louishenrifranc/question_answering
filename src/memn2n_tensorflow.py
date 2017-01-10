import tensorflow as tf
import numpy as np
import os
import sys
from termcolor import cprint
import math

import matplotlib.pyplot as plt

from utils.tf_functions_utils import tensor_matrix_mul
from utils.nlp_utils import load_embeddings, sentence2index, zero_pad, zero_pad_2d


class MemN2N(object):
    """
    End-to-end memory network.
    """

    def __init__(self, vocabulary, n_hid=100, n_hop=3, n_mem=50,
                 positional_encoding=True, temporal_encoding=True,
                 weight_tying="layer", random_context=0):
        """
        Parameters
        ----------
        vocabulary : set of str
            Vocabulary to use
        n_hid : int (default : 100)
            Dimension of the intermediary states
        n_hop : int (default : 3)
            Number of memory hops
        n_mem : int (default : 50)
            Maximum number of memories
        positional_encoding : bool (default : False)
            If positional encoding should be used
        temporal_encoding : bool (default : False)
            If temporal encoding should be used
        weight_tying : "none", "layer", "adjacent" (default : "layer")
            Which type of weight tying should be used
        random_context : float (default : 0)
            How much random empty memories to add to the context
        """

        self._n_vocab = len(vocabulary) + 2

        self._word2index, self._index2word = self._get_word_dictionaries(vocabulary)

        self._n_hid = n_hid
        self._n_hop = n_hop
        self._n_mem = n_mem
        self._random_context = random_context

        # initializer for the embeddings
        self._initializer = tf.truncated_normal_initializer(stddev=0.1)

        # context sentences placeholder
        # size : batch_size x n_context x n_context_length
        self._context = tf.placeholder(tf.int32, [None, None, None], name="context")

        # question placeholder
        # size : batch_size x n_question_length
        self._question = tf.placeholder(tf.int32, [None, None], name="question")

        # correct answer placeholder
        # size : batch_size x max_answer_length
        self._correct_answer = tf.placeholder(tf.int32, [None, None], name="correct_answer")

        # maximum length of the answer
        self._max_answer_length = tf.placeholder(tf.int32, name="max_answer_length")

        # final output of the memory hops and attention
        # size : batch_size x n_hid, batch_size x n_context x n_hop
        output, self._attention = self._compute_output(self._context, self._question, positional_encoding,
                                                       temporal_encoding, weight_tying)

        # logits and answer
        # size : batch_size x max_answer_length x n_vocab, batch_size x max_answer_length
        logits, self._answer = self._generate_answer(output)

        self._loss = self._compute_loss(logits, self._correct_answer)
        self._accuracy = self._compute_accuracy(self._answer, self._correct_answer)

        
    def _encode_context(self, context, embeddings, positional_encoding=False,
                        temporal_embeddings=None):
        """
        Encodes the `context` using the given `embeddings` and using positional
        encoding and temporal encoding if wanted.

        Parameters
        ----------
        context : Tensor-3 of size batch_size x n_context x n_context_length
            Context to encode
        embeddings : Tensor-2 of size n_vocab x n_hid
            Embeddings with which to encode the sentence
        positional_encoding : bool (default : False)
            If positional encoding should be used
        temporal_embeddings : Tensor-2 of size n_context x n_hid or None (default : None)
            Temporal embeddings to use if wanted

        Returns
        -------
        Tensor-3 of size batch_size x n_context x n_hid
            Encoded context
        """

        encoded_context = tf.nn.embedding_lookup(embeddings, context)

        if positional_encoding:
            
            L = tf.to_float(tf.ones_like(context))
            d = tf.to_float(self._n_hid)
            J = tf.to_float(tf.shape(context)[2])
            j = tf.to_float(tf.reshape(tf.range(1, tf.shape(context)[2] + 1), [1,1,tf.shape(context)[2]]))
            k = tf.to_float(tf.reshape(tf.range(1, tf.shape(context)[1] + 1), [1,tf.shape(context)[1],1]))
            L = (L - j / J) - k / d * (L - 2 * j / J)
            print(L)
            encoded_context = encoded_context * tf.expand_dims(L, -1)

        encoded_context = tf.reduce_sum(encoded_context, 2)

        if temporal_embeddings is not None:
            
            encoded_context = encoded_context + temporal_embeddings[-tf.shape(encoded_context)[1]:,:]

        return encoded_context


    def _encode_question(self, question, embeddings, positional_encoding=False):
        """
        Encodes the `question` using the given `embeddings` and positional encoding
        if wanted

        Parameters
        ----------
        question : Tensor-2 of size batch_size x n_question_length
        embeddings : Tensor-2 of size n_vocab x n_hid
            Embeddings with which to encode the sentence
        positional_encoding : bool (default : False)
            If positional encoding should be used

        Returns
        -------
        Tensor-2 of size batch_size x n_hid
            Encoded question
        """
        
        encoded_question = tf.nn.embedding_lookup(embeddings, question)

        if positional_encoding:
            
            L = tf.to_float(tf.ones_like(question))
            d = tf.to_float(self._n_hid)
            J = tf.to_float(tf.shape(question)[1])
            j = tf.to_float(tf.reshape(tf.range(1, tf.shape(question)[1] + 1), [1,tf.shape(question)[1]]))
            L = (L - j / J) - 1.0 / d * (L - 2.0 * j / J)
            encoded_question = encoded_question * tf.expand_dims(L, -1)

        encoded_question = tf.reduce_sum(encoded_question, 1)

        return encoded_question


    def _hop(self, context, current_state, context_embeddings, attention_embeddings,
             context_temporal_embeddings=None, attention_temporal_embeddings=None,
             positional_encoding=False):
        """
        Performs a single memory hop using the current state and the given embeddings.

        Parameters
        ----------
        context : Tensor-3 of size batch_size x n_context x n_context_length
            Context
        current_state : Tensor-2 of size batch_size x n_hid
            Current state of the controller
        context_embeddings : Tensor-2 of size n_vocab x n_hid
            Embeddings to encode the context
        attention_embeddings : Tensor-2 of size n_vocab x n_hid
            Embeddings to determine attention over context
        context_temporal_embeddings : Tensor-2 of size n_mem x n_hid or None (default : None)
            Embeddings to encode temporal information for the context
        attention_temporal_embeddings : Tensor-2 of size n_mem x n_hid or None (default : None)
            Embeddings to encode temporal information for the attention
        positional_encoding : bool (default : False)
            If positional encoding should be used

        Returns
        -------
        Tensor-2 of size batch_size x n_hid
            New state of the controller
        Tensor-2 of size batch_size x n_context
            Attention over the context
        """

        encoded_context = self._encode_context(context, context_embeddings,
                                               positional_encoding=positional_encoding,
                                               temporal_embeddings=context_temporal_embeddings)

        attention = self._encode_context(context, attention_embeddings,
                                         positional_encoding=positional_encoding,
                                         temporal_embeddings=attention_temporal_embeddings)

        current_state = tf.expand_dims(current_state, 1)

        attention = tf.nn.softmax(tf.reduce_sum(current_state * attention, 2))

        new_state = tf.reduce_sum(encoded_context * tf.expand_dims(attention, 2), 1)

        return new_state, attention


    def _compute_output(self, context, question, positional_encoding=False,
                        temporal_encoding=False, weight_tying="layer"):
        """
        Computes the final output after the wanted number of hops.

        Parameters
        ----------
        context : Tensor-3 of size batch_size x n_context x n_context_length
            Context
        question : Tensor-2 of size batch_size x n_question_length
            Question
        positional_encoding : bool (default : False)
            If positional encoding should be used
        temporal_encoding : bool (default : False)
            If temporal encoding should be used
        weight_tying : "none", "layer", "adjacent" (default : "layer")
            Which type of weight tying should be used
        
        Returns
        -------
        Tensor-2 of size batch_size x n_hid
            Final output
        Tensor-3 of size batch_size x n_context x n_hop
            Attention over all hops and context sentences
        """

        B = tf.concat(0, [tf.zeros([1, self._n_hid]),
                          tf.get_variable("B", [self._n_vocab-1, self._n_hid],
                                          initializer=self._initializer)])

        current_state = self._encode_question(question, B, positional_encoding)

        if weight_tying == "none":

            for h in range(self._n_hop):

                with tf.variable_scope("hop_{}".format(h+1)):

                    if temporal_encoding:
                        context_temporal_embeddings = tf.get_variable("T_C", [self._n_mem, self._n_hid],
                                                                      initializer=self._initializer)
                        attention_temporal_embeddings = tf.get_variable("T_A", [self._n_mem, self._n_hid],
                                                                        initializer=self._initializer)
                    else:
                        context_temporal_embeddings = None
                        attention_temporal_embeddings = None

                    context_embeddings = tf.concat(0, [tf.zeros([1, self._n_hid]),
                                                       tf.get_variable("C", [self._n_vocab - 1, self._n_hid],
                                                                       initializer=self._initializer)])
                    attention_embeddings = tf.concat(0, [tf.zeros([1, self._n_hid]),
                                                         tf.get_variable("A", [self._n_vocab - 1, self._n_hid],
                                                                         initializer=self._initializer)])

                    new_state, att = self._hop(context, current_state,
                                               context_embeddings, attention_embeddings,
                                               context_temporal_embeddings,
                                               attention_temporal_embeddings,
                                               positional_encoding)

                    current_state = current_state + new_state

                if h == 0:
                    attention = tf.expand_dims(att, 2)
                else:
                    attention = tf.concat(2, [attention, tf.expand_dims(att, 2)])

        elif weight_tying == "layer":

            with tf.variable_scope("hops"):

                if temporal_encoding:
                    context_temporal_embeddings = tf.get_variable("T_C", [self._n_mem, self._n_hid],
                                                                  initializer=self._initializer)
                    attention_temporal_embeddings = tf.get_variable("T_A", [self._n_mem, self._n_hid],
                                                                    initializer=self._initializer)
                else:
                    context_temporal_embeddings = None
                    attention_temporal_embeddings = None

                context_embeddings = tf.concat(0, [tf.zeros([1, self._n_hid]),
                                                       tf.get_variable("C", [self._n_vocab - 1, self._n_hid],
                                                                       initializer=self._initializer)])
                attention_embeddings = tf.concat(0, [tf.zeros([1, self._n_hid]),
                                                     tf.get_variable("A", [self._n_vocab - 1, self._n_hid],
                                                                     initializer=self._initializer)])
                
                H = tf.get_variable("H", [self._n_hid, self._n_hid],
                                    initializer=self._initializer)

                for h in range(self._n_hop):

                    new_state, att = self._hop(context, current_state,
                                               context_embeddings, attention_embeddings,
                                               context_temporal_embeddings,
                                               attention_temporal_embeddings,
                                               positional_encoding)

                    current_state = tf.matmul(current_state, H) + new_state

                    if h == 0:
                        attention = tf.expand_dims(att, 2)
                    else:
                        attention = tf.concat(2, [attention, tf.expand_dims(att, 2)])

        elif weight_tying == "adjacent":

            for h in range(self._n_hop):

                with tf.variable_scope("hop_{}".format(h+1)):

                    if temporal_encoding:
                        
                        if h == 0:
                            attention_temporal_embeddings = tf.get_variable("T_A", [self._n_mem, self._n_hid],
                                                                            initializer=self._initializer)
                        else:
                            attention_temporal_embeddings = context_temporal_embeddings
                            
                        context_temporal_embeddings = tf.get_variable("T_C", [self._n_mem, self._n_hid],
                                                                      initializer=self._initializer)

                    else:
                        context_temporal_embeddings = None
                        attention_temporal_embeddings = None

                    if h == 0:
                        context_embeddings = B
                    else:
                        context_embeddings = attention_embeddings

                    attention_embeddings = tf.concat(0, [tf.zeros([1, self._n_hid]),
                                                         tf.get_variable("A", [self._n_vocab - 1, self._n_hid],
                                                                         initializer=self._initializer)])
                        
                    new_state, att = self._hop(context, current_state,
                                               context_embeddings, attention_embeddings,
                                               context_temporal_embeddings,
                                               attention_temporal_embeddings,
                                               positional_encoding)

                    current_state = current_state + new_state

                    if h == 0:
                        attention = tf.expand_dims(att, 2)
                    else:
                        attention = tf.concat(2, [attention, tf.expand_dims(att, 2)])

        return current_state, attention


    def _answer_step(self, H_prev, Y_prev, y_prev):
        """
        Single step of te answer generation.

        Parameters
        ----------
        H_prev : Tensor-2 of size batch_size x n_hid
            Previous hidden output
        Y_prev : Tensor-2 of size batch_size x n_vocab
            Previous logits
        y_prev : Tensor-1 of size batch_size
            Previous answer

        Returns
        -------
        Tensor-2 of size batch_size x n_hid
            Current hidden output
        Tensor-2 of size batch_size x n_vocab
            Current logits
        Tensor-1 of size batch_size
            Current answer
        """

        with tf.variable_scope("answer_step"):

            U = tf.get_variable("U", [self._n_hid, self._n_hid],
                                initializer=self._initializer)
            b = tf.get_variable("b", [self._n_hid],
                                initializer=tf.constant_initializer(0.1))

            H = tf.nn.relu(tf.matmul(H_prev, U) + b)

            W_p = tf.get_variable("W_p", [self._n_hid, self._n_vocab],
                                  initializer=self._initializer)
            b_p = tf.get_variable("b_p", [self._n_vocab],
                                  initializer=tf.constant_initializer(0.0))

            Y = tf.matmul(H, W_p) + b_p
            y = tf.to_int32(tf.argmax(Y, 1))

        return H, Y, y

    
    def _generate_answer(self, output):
        """
        Generates the answer given the final output.

        Parameters
        ----------
        output : Tensor-2 of size batch_size x n_hid
            Final output of the memory hops

        Returns
        -------
        Tensor-2 of size batch_size x n_vocab
            Logits
        Tensor-1 of size batch_size
            Answer
        """

        with tf.variable_scope("answer_generation"):

            # initializer = (output,
            #                tf.zeros([tf.shape(output)[0], self._n_vocab]),
            #                tf.zeros([tf.shape(output)[0]], dtype=tf.int32))

            # elems = tf.zeros([self._max_answer_length])

            # _, logits, answer = tf.scan(lambda H_Y_y, X: self._answer_step(*H_Y_y),
            #                             elems, initializer=initializer)

            # logits = tf.transpose(logits, [1,0,2])
            # answer = tf.transpose(answer)

            W = tf.get_variable("W", [self._n_hid, self._n_vocab],
                                initializer=self._initializer)
            logits = tf.matmul(output, W)
            answer = tf.to_int32(tf.argmax(logits, 1))

        return logits, answer


    def _compute_loss(self, logits, correct_answer):
        """
        Computes the cross entropy loss.

        Parameters
        ----------
        logits : Tensor-2 of size batch_size x n_vocab
            Logits
        correct_answer : Tensor-1 of size batch_size
            Correct answer

        Returns
        -------
        Tensor-0
            Cross entropy loss
        """

        # logits = tf.reshape(logits, [-1, self._n_vocab])
        correct_answer = tf.reshape(correct_answer, [-1])

        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                                                            correct_answer))

        return loss


    def _compute_accuracy(self, answer, correct_answer):
        """
        Computes the accuracy.

        Parameters
        ----------
        answer : Tensor-1 of size batch_size
            Generated answer
        correct_answer : Tensor-1 of size batch_size
            Correct answer
    
        Returns
        -------
        Tensor-0
            Accuracy
        """

        # answer = tf.reshape(answer, [-1])
        correct_answer = tf.reshape(correct_answer, [-1])

        accuracy = tf.reduce_mean(tf.to_float(tf.equal(answer, correct_answer)))

        return accuracy


    def _get_word_dictionaries(self, vocabulary):
        """
        Creates word dictionaries from the vocabulary

        Parameters
        ----------
        vocabulary : set of str
            Vocabulary to use

        Returns
        -------
        dict of str to int
            Dictionary of word to index
        dict of int to str
            Dictionary of index to word
        """

        word2index = {w: i + 2 for i,w in enumerate(vocabulary)}
        word2index["__empty__"] = 0
        word2index["__unk__"] = 1

        index2word = {i: w for w,i in word2index.items()}

        return word2index, index2word
        

    def _prepare_data(self, context, question, answer, batch_size=None):
        """
        Prepares the data to be usable by the model.

        Parameters
        ----------
        context : list-2 of str
            Context sentences
        question : list of str
            Questions
        answer : list of str
            Answers
        batch_size : int or None (default : None)
            Size of each minibatch

        Returns
        -------
        list-3 of int
            Converted context
        list-2 of int
            Converted question
        list-2 of int
            Converted answer
        """
        if self._random_context > 0:
            random_indices = [np.random.choice(np.arange(len(c)),
                                               size=int(len(c) * self._random_context),
                                               replace=False)
                              for c in context]
            for c in range(len(context)):
                for i in random_indices[c]:
                    context[c].insert(i, "")

        context = sentence2index(context, self._word2index)
        context = zero_pad_2d(context, pad="left", batch_size=batch_size)
        context = [c[-self._n_mem:] for c in context]

        question = sentence2index(question, self._word2index)
        question = zero_pad(question, pad="left", batch_size=batch_size)

        answer = sentence2index(answer, self._word2index)
        answer = zero_pad(answer, pad="right", batch_size=batch_size)

        return context, question, answer

    
    def train(self, context, question, answer, batch_size=32, max_epoch=25,
              learning_rate=0.01, save_name="memn2n"):
        """
        Trains the model.

        Parameters
        ----------
        context : list-2 of str
            Context sentences
        question : list of str
            Questions
        answer : list of str
            Answers
        batch_size : int (default : 32)
            Size of each minibatch
        max_epoch : int (default : 25)
            Maximum number of epochs
        learning_rate : float (default : 0.01)
            Starting learning rate
        save_name : str (default : "memn2n")
            Name under which to save the model
        """

        train_indices = np.random.choice(np.arange(len(context)), int(0.9 * len(context)))

        context_train = [context[i] for i in range(len(context)) if i in train_indices]
        question_train = [question[i] for i in range(len(question)) if i in train_indices]
        answer_train = [answer[i] for i in range(len(answer)) if i in train_indices]

        context_valid = [context[i] for i in range(len(context)) if i not in train_indices]
        question_valid = [question[i] for i in range(len(question)) if i not in train_indices]
        answer_valid = [answer[i] for i in range(len(answer)) if i not in train_indices]

        (context_train,
         question_train,
         answer_train) = self._prepare_data(context_train, question_train,
                                            answer_train, batch_size)

        (context_valid,
         question_valid,
         answer_valid) = self._prepare_data(context_valid, question_valid,
                                            answer_valid, batch_size)

        n_train_batches = math.ceil(len(context_train) / batch_size)
        n_valid_batches = math.ceil(len(context_valid) / batch_size)

        alpha = tf.placeholder(tf.float32, name="learning_rate")
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdamOptimizer()
        grads = optimizer.compute_gradients(self._loss)
        grads = [(tf.clip_by_norm(grad, 40), var) for grad, var in grads]
        training_step = optimizer.apply_gradients(grads)

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        train_losses = []
        valid_losses = []

        train_accuracies = []
        valid_accuracies = []

        epoch = 0
        converged = False
        
        while epoch < max_epoch and not converged:
            epoch += 1
            if epoch % 25 == 0:
                learning_rate = learning_rate / 2

            for i in range(n_train_batches):

                feed_dict = {self._context: context_train[i * batch_size: (i+1) * batch_size],
                             self._question: question_train[i * batch_size: (i+1) * batch_size],
                             self._correct_answer: answer_train[i * batch_size: (i+1) * batch_size],
                             self._max_answer_length: len(answer_train[i * batch_size]),
                             alpha: learning_rate}

                loss, accuracy, _ = self._sess.run([self._loss, self._accuracy,
                                                    training_step],
                                                   feed_dict)

                train_losses.append(loss)
                train_accuracies.append(accuracy)

                cprint("Epoch {0:<3}, Batch {1:<3}/{2:}, Loss {3:<9.6f}, Accuracy {4:<5.2f} %".format(
                    epoch, i + 1, n_train_batches,
                    np.mean(train_losses[-batch_size:]),
                    np.mean(train_accuracies[-batch_size:]) * 100),
                       "yellow", end="\r")

            if epoch % 10 == 0:
                print()
                for i in range(n_valid_batches):

                    feed_dict = {self._context: context_valid[i * batch_size: (i+1) * batch_size],
                                 self._question: question_valid[i * batch_size: (i+1) * batch_size],
                                 self._correct_answer: answer_valid[i * batch_size: (i+1) * batch_size],
                                 self._max_answer_length: len(answer_valid[i * batch_size])}

                    loss, accuracy = self._sess.run([self._loss, self._accuracy],
                                                    feed_dict)

                    valid_losses.append(loss)
                    valid_accuracies.append(accuracy)

                    cprint("Epoch {0:<3}, Batch {1:<3}/{2:}".format(
                        epoch, i + 1, n_valid_batches),
                           "cyan", end="\r")

                cprint("Epoch {0:<3}, Loss {1:<9.6f}, Accuracy {2:<5.2f} %".format(
                    epoch, np.mean(valid_losses[-batch_size:]),
                    np.mean(valid_accuracies[-batch_size:]) * 100),
                       "cyan")

                # if epoch > 5 and np.mean(valid_losses[-batch_size:]) > np.mean(valid_losses[-6*batch_size:-batch_size]):
                    # converged = True

        cprint("Training done.", "green", attrs=["bold"])

        saver = tf.train.Saver(max_to_keep=1)
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "models", "{}.tf_model".format(save_name))
        saver.save(self._sess, save_path)


    def load(self, save_name="memn2n"):
        """
        Loads the model under the given name.

        Parameters
        ----------
        save_name : str (default : "memn2n")
            Name under which the model is saved
        """

        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "models", "{}.tf_model".format(save_name))

        if not os.path.exists(save_path + ".index"):
            cprint("[-] There is no saved model under the name \"{}\"".format(save_name),
                   "red", attrs=["bold"])
            sys.exit()

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self._sess, save_path)

        cprint("[+] The model was loaded.", "green", attrs=["bold"])

        
    def visualize_attention(self, context, question, answer, name):
        """
        Creates a heatmap of attention.
        
        Parameters
        ----------
        context : list of str
            Context sentences
        question : str
            Questions
        answer : str
            Answers
        name : str
            Name under which to save the graph
        """

        correct_answer = answer

        prepared_data = self._prepare_data([context], [question], [correct_answer])

        feed_dict = {self._context: prepared_data[0],
                     self._question: prepared_data[1],
                     self._correct_answer: prepared_data[2]}
        
        answer, attention = self._sess.run([self._answer, self._attention],
                                           feed_dict)

        answer, attention = answer[0], attention[0,:,:]

        fig, ax = plt.subplots()

        heatmap = ax.pcolor(attention, cmap=plt.cm.Blues, alpha=0.8)

        fig = plt.gcf()
        fig.set_size_inches(8,11)

        ax.set_title("{}\n{}\n{}".format(question, self._index2word[answer], correct_answer), fontsize=16)

        ax.set_frame_on(False)

        ax.set_yticks(np.arange(attention.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(attention.shape[1]) + 0.5, minor=False)

        # ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels([] * attention.shape[1], minor=False)
        ax.set_yticklabels(context, minor=False)

        ax.grid(False)

        ax = plt.gca()

        for t in ax.xaxis.get_major_ticks():
            t.tick10n = False
            t.tick20n = False
        for t in ax.yaxis.get_major_ticks():
            t.tick10n = False
            t.tick20n = False

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "{}.svg".format(name))

        plt.tight_layout()
        plt.tick_params(axis="both", which="both", length=0)
        plt.savefig(path)

        
