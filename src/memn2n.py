import numpy as np
import theano.tensor as T
from theano import scan, function, shared, In
from theano.ifelse import ifelse
import os
import sys; sys.setrecursionlimit(50000)
from termcolor import cprint
import math
from six.moves import cPickle
import matplotlib.pyplot as plt

from utils.nlp_utils import load_embeddings, sentence2index, zero_pad, zero_pad_2d
from utils.theano_utils import get_embeddings, get_weigths_and_bias
from utils.theano_utils import embedding_lookup, softmax


class MemN2N(object):
    """
    End-to_end memory network.
    """

    def __init__(self, vocabulary, n_emb=100, n_hop=3, n_mem=50,
                 positional_encoding=True, temporal_encoding=True,
                 weight_tying="layer", random_context=0):
        """
        Parameters
        ----------
        vocabulary : set of str
            Vocabulary to use
        n_emb : int (default : 100)
            Dimension of the embeddings
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

        cprint("Building model...", "magenta", attrs=["bold"])

        self._n_vocab = len(vocabulary) + 2

        self._word2index, self._index2word = self._get_word_dictionaries(vocabulary)

        self._n_emb = n_emb
        self._n_hop = n_hop
        self._n_mem = n_mem
        self._weight_tying = weight_tying
        self._positional_encoding = positional_encoding
        self._temporal_encoding = temporal_encoding
        self._random_context = random_context

        self._params = []

        self._context = T.itensor3("context")
        self._question = T.imatrix("question")
        self._correct_answer = T.ivector("correct_answer")

        self._ls = T.bscalar("linear_start")

        output, self._attention = self._compute_output(self._context, self._question)

        logits, self._answer = self._generate_answer(output)

        self._loss = self._compute_loss(logits, self._correct_answer)

        self._accuracy = self._compute_accuracy(self._answer, self._correct_answer)

        self._get_accuracy = function(inputs=[self._context, self._question, self._correct_answer, In(self._ls, value=False)],
                                      outputs=self._accuracy)
        self._get_answer = function(inputs=[self._context, self._question, In(self._ls, value=False)],
                                    outputs=self._answer)
        self._get_answer_and_attention = function(inputs=[self._context, self._question, In(self._ls, value=False)],
                                                  outputs=[self._answer, self._attention])


    def _encode(self, inputs, embeddings, temporal_embeddings=None):
        """
        Encodes the `inputs` by summing the embeddings over the last dimension.

        Parameters
        ----------
        inputs : Tensor-2 of size batch_size x seq_length or tensor-3 of size batch_size x n_context x seq_length
            Question or context
        embeddings : Tensor-2 of size n_vocab x n_emb
            Embeddings
        temporal_embeddings : Tensor-2 of size n_mem x n_emb (default : None)
            Temporal embeddings

        Returns
        -------
        Tensor-2 of size batch_size x n_emb or tensor-3 of size batch_size x n_context x n_emb
            Encoded inputs
        """

        encoded = embedding_lookup(embeddings, inputs)

        if self._positional_encoding:

            L = T.cast(T.ones_like(inputs), "float32")
            d = self._n_emb
            J = inputs.shape[-1]
            j = T.arange(1, inputs.shape[-1] + 1).dimshuffle("x",0)
            if inputs.ndim == 3:
                k = T.arange(1, inputs.shape[1] + 1).dimshuffle("x",0,"x")
                j = j.dimshuffle("x",0,1)
                L = (L - j / J) - k / d * (L - 2 * j / J)
                encoded = encoded * L.dimshuffle(0,1,2,"x")
            else:
                k = T.ones([])
                L = (L - j / J) - k / d * (L - 2 * j / J)
                encoded = encoded * L.dimshuffle(0,1,"x")

        encoded = encoded.sum(axis=-2)

        if temporal_embeddings is not None:

            encoded = encoded + temporal_embeddings

        return encoded


    def _hop(self, output_prev, attention_prev,
             context, context_embeddings, attention_embeddings,
             context_temporal_embeddings, attention_temporal_embeddings):
        """
        Computes a memory hop.

        Parameters
        ----------
        output_prev : Tensor-2 of size batch_size x n_emb
            Previous output
        attention_prev : Tensor-2 of size batch_size x n_context
            Previous attention
        context : Tensor-3 of size batch_size x n_context x context_length
            Context indices
        context_embeddings : Tensor-2 of size n_vocab x n_emb
            Embeddings for the context memories
        attention_embeddings : Tensor-2 of size n_vocab x n_emb
            Embeddings for the attention over memories
        context_temporal_embeddings : Tensor-2 of size n_mem x n_emb
            Embeddings for the temporal component of the the context
        attention_temporal_embeddings : Tensor-2 of size n_mem x n_emb
            Embeddings for the temporal component of the the attention
        
        Returns
        -------
        Tensor-2 of size batch_size x n_emb
            Output
        Tensor-2 of size batch_size x n_context
            Attention
        """

        memories = self._encode(context, context_embeddings, context_temporal_embeddings)
        attention = self._encode(context, attention_embeddings, attention_temporal_embeddings)

        attention = T.batched_dot(attention, output_prev.dimshuffle(0,1,"x")).squeeze()
        attention = ifelse(self._ls, attention, softmax(attention))

        output = (memories * attention.dimshuffle(0,1,"x")).sum(axis=1)

        output = output + output_prev

        return output, attention


    def _compute_output(self, context, question):
        """
        Computes the final ouput.

        Parameters
        ----------
        context : Tensor-3 of size batch_size x n_context x context_length
            Context indices
        question : Tensor-2 of size batch_size x question_length
            Question

        Returns
        -------
        Tensor-2 of size batch_size x n_emb
            Final output of the memory hops
        Tensor-3 of size batch_size x n_hop x n_context
            Attention of all hops
        """

        question_embeddings = get_embeddings(self._n_vocab, self._n_emb,
                                                 name="question")
        self._params += [question_embeddings]
        question_embeddings = T.concatenate([T.zeros([1, self._n_emb]), question_embeddings],
                                            axis=0)

        output = self._encode(question, question_embeddings)

        if self._weight_tying == "layer":
            context_embeddings = get_embeddings(self._n_vocab-1, self._n_emb,
                                                name="context")
            attention_embeddings = get_embeddings(self._n_vocab-1, self._n_emb,
                                                  name="attention")
            self._params += [context_embeddings, attention_embeddings]
            context_embeddings = T.concatenate([T.zeros([1, self._n_emb]), context_embeddings],
                                            axis=0)
            attention_embeddings = T.concatenate([T.zeros([1, self._n_emb]), attention_embeddings],
                                            axis=0)

            if self._temporal_encoding:
                context_temporal_embeddings = get_embeddings(self._n_mem, self._n_emb,
                                                             name="context_temporal")
                attention_temporal_embeddings = get_embeddings(self._n_mem, self._n_emb,
                                                               name="attention_temporal")
                self._params += [context_temporal_embeddings, attention_temporal_embeddings]
            else:
                context_temporal_embeddings, attention_temporal_embeddings = None, None

            [output, attention] ,_ = scan(self._hop,
                                          outputs_info=[output, T.zeros(context.shape[:2])],
                                          non_sequences=[context,
                                                         context_embeddings,
                                                         attention_embeddings,
                                                         context_temporal_embeddings,
                                                         attention_temporal_embeddings],
                                          n_steps=self._n_hop)

            output = output[-1,:,:]
            attention = attention.dimshuffle(1,0,2)

            W, _ = get_weigths_and_bias(self._n_emb, self._n_emb, name="final")
            self._params += [W]

            output = T.dot(output, W)
            

        elif self._weight_tying == "adjacent":

            for h in range(self._n_hop):

                if h == 0:
                    attention_embeddings = question_embeddings
                    if self._temporal_encoding:
                        attention_temporal_embeddings = get_embeddings(self._n_mem, self._n_emb,
                                                                       name="attention_temporal")
                        self._params += [attention_temporal_embeddings]
                    else:
                        attention_temporal_embeddings = None
                else:
                    attention_embeddings = context_embeddings
                    attention_temporal_embeddings = context_temporal_embeddings

                context_embeddings = get_embeddings(self._n_vocab-1, self._n_emb,
                                                    name="context_{}".format(h+1))
                self._params += [context_embeddings]
                context_embeddings = T.concatenate([T.zeros([1, self._n_emb]), context_embeddings],
                                            axis=0)
                if self._temporal_encoding:
                    context_temporal_embeddings = get_embeddings(self._n_mem, self._n_emb,
                                                                 name="context_temporal_{}".format(h+1))
                    self._params += [context_temporal_embeddings]
                else:
                    context_temporal_embeddings = None

                if h == 0:
                    att = T.zeros(context.shape[:2])
                    output, att = self._hop(output, att, context,
                                            context_embeddings,
                                            attention_embeddings,
                                            context_temporal_embeddings,
                                            attention_temporal_embeddings)
                    attention = att.dimshuffle(0,"x",1)
                else:
                    output, att = self._hop(output, att, context,
                                            context_embeddings,
                                            attention_embeddings,
                                            context_temporal_embeddings,
                                            attention_temporal_embeddings)
                    attention = T.concatenate([attention, att.dimshuffle(0,"x",1)],
                                              axis=1)

        elif self._weight_tying == "none":

            for h in range(self._n_hop):

                context_embeddings = get_embeddings(self._n_vocab-1, self._n_emb,
                                                    name="context_{}".format(h+1))
                attention_embeddings = get_embeddings(self._n_vocab-1, self._n_emb,
                                                      name="attention_{}".format(h+1))
                self._params += [context_embeddings, attention_embeddings]
                context_embeddings = T.concatenate([T.zeros([1, self._n_emb]), context_embeddings],
                                                   axis=0)
                attention_embeddings = T.concatenate([T.zeros([1, self._n_emb]), attention_embeddings],
                                                     axis=0)

                if self._temporal_encoding:
                    context_temporal_embeddings = get_embeddings(self._n_mem, self._n_emb,
                                                                 name="context_temporal_{}".format(h+1))
                    attention_temporal_embeddings = get_embeddings(self._n_mem, self._n_emb,
                                                                   name="attention_temporal_{}".format(h+1))
                    self._params += [context_temporal_embeddings, attention_temporal_embeddings]
                else:
                    context_temporal_embeddings, attention_temporal_embeddings = None, None

                if h == 0:
                    att = T.zeros(context.shape[:2])
                    output, att = self._hop(output, att, context,
                                            context_embeddings,
                                            attention_embeddings,
                                            context_temporal_embeddings,
                                            attention_temporal_embeddings)
                    attention = att.dimshuffle(0,"x",1)
                else:
                    output, att = self._hop(output, att, context,
                                            context_embeddings,
                                            attention_embeddings,
                                            context_temporal_embeddings,
                                            attention_temporal_embeddings)
                    attention = T.concatenate([attention, att.dimshuffle(0,"x",1)],
                                              axis=1)

        else:
            cprint("[-] \"{}\" is not a valid weight tying method.".format(self._weight_tying),
                   "red", attrs=["bold"])
            sys.exit()

        return output, attention


    def _generate_answer(self, output):
        """
        Generates the answer given the final `output`.

        Parameters
        ----------
        output : Tensor-2 of size batch_size x n_emb
            Final output of the memory hops

        Returns
        -------
        Tensor-2 of size batch_size x n_vocab
            Logits
        Tensor-1 of size batch_size
            Answer
        """

        W, b = get_weigths_and_bias(self._n_emb, self._n_vocab, name="answer_generation")
        self._params += [W,b]

        logits = softmax(T.dot(output, W) + b)
        answer = T.argmax(logits, axis=1)

        return logits, answer


    def _compute_loss(self, logits, correct_answer):
        """
        Computes the mean cross-entropy loss.

        Parameters
        ----------
        logits : Tensor-2 of size batch_size x n_out
            Logits
        correct_answer : Tensor-1 of size batch_size
            Correct answer
        
        Returns
        -------
        Tensor-0
            Loss
        """

        loss = -(T.log(logits[T.arange(correct_answer.shape[0]), correct_answer] + 1e-32)).sum()

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

        accuracy = T.eq(answer, correct_answer).mean()

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
        

    def _prepare_data(self, context, question, answer=None, batch_size=None):
        """
        Prepares the data to be usable by the model.

        Parameters
        ----------
        context : list-2 of str
            Context sentences
        question : list of str
            Questions
        answer : list of str (default : None)
            Answers
        batch_size : int or None (default : None)
            Size of each minibatch

        Returns
        -------
        list-3 of int
            Converted context
        list-2 of int
            Converted question
        list-2 of int (optional)
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
        context = [[[0] * len(c[0])] * (self._n_mem - len(c)) + c
                   if len(c) < self._n_mem
                   else c[-self._n_mem:]
                   for c in context]

        question = sentence2index(question, self._word2index)
        question = zero_pad(question, pad="left", batch_size=batch_size)

        if answer is None:
            
            return context, question
        
        else:
            
            answer = sentence2index(answer, self._word2index)
            answer = [a[0] for a in answer]

            return context, question, answer

        
    def train(self, context, question, answer, batch_size=32, max_epoch=25,
              learning_rate=0.01, save_name="memn2n", early_stopping=True):
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
        early_stopping : bool (default : True)
            If early stopping should be used
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

        n_train_batch = math.ceil(len(context_train) / batch_size)
        n_valid_batch = math.ceil(len(context_valid) / batch_size)

        grads = T.grad(self._loss, self._params)
        norm = 40
        grads = [ifelse(T.lt(grad.norm(2), norm), grad, grad * 40 / grad.norm(2))
                 for grad in grads]

        alpha = T.scalar("learning_rate")

        updates = tuple([(param, param - alpha * grad)
                         for param, grad in zip(self._params, grads)])

        train_batch = function(inputs=[self._context, self._question, self._correct_answer, self._ls, alpha],
                               outputs=[self._loss, self._accuracy],
                               updates=updates)

        valid_batch = function(inputs=[self._context, self._question, self._correct_answer, self._ls],
                               outputs=[self._loss, self._accuracy])

        cprint("Training model...", "magenta", attrs=["bold"])
                               
        epoch = 0
        linear_start = True
        converged = False

        train_losses = []
        valid_losses = []

        train_accuracies = []
        valid_accuracies = []

        while epoch < max_epoch and not converged:
            epoch += 1
            if epoch % 25 == 0:
                learning_rate /= 2

            for i in range(n_train_batch):
                c = context_train[i * batch_size: (i+1) * batch_size]
                q = question_train[i * batch_size: (i+1) * batch_size]
                a = answer_train[i * batch_size: (i+1) * batch_size]
                loss, accuracy = train_batch(c, q, a, linear_start, learning_rate)
                train_losses.append(float(loss))
                train_accuracies.append(float(accuracy))
                cprint("Epoch {0:>4}, Batch {1:>5}/{2:}, Loss {3:>9.6f}, Accuracy {4:>5.2f} %".format(
                    epoch, i+1, n_train_batch,
                    np.mean(train_losses[-batch_size:]),
                    np.mean(train_accuracies[-batch_size:]) * 100),
                       "yellow", end="\r")
            print()
            valid_losses.append(0)
            valid_accuracies.append(0)
            for i in range(n_valid_batch):
                c = context_valid[i * batch_size: (i+1) * batch_size]
                q = question_valid[i * batch_size: (i+1) * batch_size]
                a = answer_valid[i * batch_size: (i+1) * batch_size]
                loss, accuracy = valid_batch(c, q, a, linear_start)
                valid_losses[-1] += loss
                valid_accuracies[-1] += accuracy
                cprint("Epoch {0:>4}, Batch {1:>5}/{2:}".format(
                    epoch, i+1, n_valid_batch),
                       "cyan", end="\r")
            valid_losses[-1] /= n_valid_batch
            valid_accuracies[-1] /= n_valid_batch
            cprint("Epoch {0:>4}, Loss {1:>9.6f}, Accuracy {2:>5.2f} %".format(
                    epoch, 
                    valid_losses[-1],
                    valid_accuracies[-1] * 100),
                   "cyan")
            if epoch > 15 and np.mean(valid_losses[-5:]) > np.mean(valid_losses[-15:-5]):
                if not linear_start and early_stopping:
                    converged = True
                else:
                    linear_start = False

        cprint("Training done.", "green", attrs=["bold"])


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

        answer, attention = self.answer(context, question, return_attention=True)

        attention = attention.T

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
        ax.set_yticklabels([str(i+1) + " " + c for i,c in enumerate(context)], minor=False)

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


    def accuracy(self, context, question, answer, batch_size=None):
        """
        Computes the accuracy.

        Parameters
        ----------
        context : list-2 of str
            Context sentences
        question : list of str
            Questions
        answer : list of str
            Answers
        batch_size : int (default : None)
            Size of each minibatch

        Returns
        -------
        float
            Accuracy rate
        """

        prepared_data = self._prepare_data(context, question, answer, batch_size)

        return self._get_accuracy(*prepared_data)


    def answer(self, context, question, batch_size=None, return_attention=False):
        """
        Computes the accuracy.

        Parameters
        ----------
        context : list-2 of str
            Context sentences
        question : list of str
            Questions
        batch_size : int (default : None)
            Size of each minibatch
        return_attention : bool (default : False)
            If the attention should be returned

        Returns
        -------
        int or list of int
            Answer
        Matrix or tensor-3
            Attention matrix
        """

        single = isinstance(question, str)

        if single:
            prepared_data = self._prepare_data([context], [question], batch_size=batch_size)
        else:
            prepared_data = self._prepare_data(context, question, batch_size=batch_size)

        if return_attention:
            answer, attention = self._get_answer_and_attention(*prepared_data)
            if single:
                answer, attention = answer[0], attention[0]
            return answer, attention
        else:
            answer = self._get_answer(*prepared_data)
            if single:
                answer = answer[0]
            return answer


    def save(self, name):
        """
        Saves the model under the given `name`.
        
        Parameters
        ----------
        name : str
            Name under which to save
        """

        cprint("[*] Saving model...", "yellow", end="\r")

        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "models", "{}.theano_model".format(name))

        with open(save_path, "wb") as f:
            cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)

        cprint("[+] Model saved.   ", "cyan")


    def load(self, name):
        """
        Loads the model under the given `name`.
        
        Parameters
        ----------
        name : str
            Name under which the file is saved

        Returns
        -------
        MemN2n
            Saved instance of the model
        """

        cprint("[*] Loading model...", "yellow", end="\r")

        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "models", "{}.theano_model".format(name))

        if not os.path.exists(save_path):
            cprint("[-] There is no saved model under the name \"{}\"".format(name),
                   "red", attrs=["bold"])
            sys.exit()

        with open(save_path, "rb") as f:
            self = cPickle.load(f)

        cprint("[+] Model loaded.   ", "cyan")

        return self
