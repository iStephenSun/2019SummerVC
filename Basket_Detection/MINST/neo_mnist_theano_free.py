"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.
Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.
Mathematically, this can be written as:
.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}
The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).
.. math::
  y_{pred} = argmax_i P(Y=i|x,W,b)
This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.
References:
    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy
import numpy as np

import theano
import theano.tensor as T





class LogisticRegression(object):
    """Multi-class Logistic Regression Class
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie
        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = np.zeros((n_in, n_out), dtype=float)

        # initialize the biases b as a vector of n_out 0s
        self.b = np.zeros((n_out,), dtype=float)  # 1*10 row 行向量


        # hyperplane-k
        # self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.p_y_given_x = self.softmax(np.dot(input, self.W) + self.b)




        # symbolic description of how to compute prediction as class whose
        # probability is maximal

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        # theta?
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input



    def softmax(self,x):
        """
        对输入x的每一行计算softmax。

        该函数对于输入是向量（将向量视为单独的行）或者矩阵（M x N）均适用。

        代码利用softmax函数的性质: softmax(x) = softmax(x + c)

        参数:
        x -- 一个N维向量，或者M x N维numpy矩阵.

        返回值:
        x -- 在函数内部处理后的x
        """


        # 根据输入类型是矩阵还是向量分别计算softmax
        if len(x.shape) > 1:
            # 矩阵
            tmp = np.max(x, axis=1)  # 得到每行的最大值，用于缩放每行的元素，避免溢出
            x -= tmp.reshape((x.shape[0], 1))  # 利用性质缩放元素
            x = np.exp(x)  # 计算所有值的指数
            tmp = np.sum(x, axis=1)  # 每行求和
            x /= tmp.reshape((x.shape[0], 1))  # 求softmax
        else:
            # 向量
            tmp = np.max(x)  # 得到最大值
            x -= tmp  # 利用最大值缩放数据
            x = np.exp(x)  # 对所有元素求指数
            tmp = np.sum(x)  # 求元素和
            x /= tmp  # 求somftmax
        return x

    # Defining a Loss Function
    # y.shape[0]是y的行数，即样本个数n
    #
    # T.arange(y.shape[0])为[0,1,2,... n-1]
    #
    # T.log(self.p_y_given_x)为矩阵，n行每行为一个样本，10列每列为一类
    #
    # 最后return 一个10维行向量
    # def negative_log_likelihood(self, y):
    def cost_funtion(self, y):

        # return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        return -np.mean(np.log(self.p_y_given_x)[np.arange(y.shape[0]), y])

    def gradient_W_b(self, x , y):
        cost_grad = -np.mean((x.T)[[np.arange(y.shape[0]), y] - self.p_y_given_x])
        return cost_grad

    def update_theta(self, theta, alpha, cost_grad):
        new_theta = theta - alpha * cost_grad
        return new_theta
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):
    floatX = 'float32'
    intY = 'int32'
    data_path = dataset
    # Load the dataset
    with gzip.open(data_path, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    def feature_label(data_xy):
        data_x, data_y = data_xy
        data_x = np.asarray(data_x, dtype=floatX)
        data_y = np.asarray(data_y, dtype=intY)
        return data_x, data_y

    test_set_x, test_set_y = feature_label(test_set)
    valid_set_x, valid_set_y = feature_label(valid_set)
    train_set_x, train_set_y = feature_label(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval
#
# def load_data(dataset):
#     ''' Loads the dataset
#     :type dataset: string
#     :param dataset: the path to the dataset (here MNIST)
#     '''
#
#     #############
#     # LOAD DATA #
#     #############
#
#     # Download the MNIST dataset if it is not present
#     data_dir, data_file = os.path.split(dataset)
#     if data_dir == "" and not os.path.isfile(dataset):
#         # Check if dataset is in the data directory.
#         new_path = os.path.join(
#             os.path.split(__file__)[0],
#             "..",
#             "data",
#             dataset
#         )
#         if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
#             dataset = new_path
#
#     if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
#         from six.moves import urllib
#         origin = (
#             'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
#         )
#         print('Downloading data from %s' % origin)
#         urllib.request.urlretrieve(origin, dataset)
#
#     print('... loading data')
#
#     # Load the dataset
#     with gzip.open(dataset, 'rb') as f:
#         try:
#             train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
#         except:
#             train_set, valid_set, test_set = pickle.load(f)
#
#     # train_set, valid_set, test_set format: tuple(input, target)
#     # input is a numpy.ndarray of 2 dimensions (a matrix)
#     # where each row corresponds to an example. target is a
#     # numpy.ndarray of 1 dimension (vector) that has the same length as
#     # the number of rows in the input. It should give the target
#     # to the example with the same index in the input.
#
#     def shared_dataset(data_xy, borrow=True):
#         """ Function that loads the dataset into shared variables
#         The reason we store our dataset in shared variables is to allow
#         Theano to copy it into the GPU memory (when code is run on GPU).
#         Since copying data into the GPU is slow, copying a minibatch everytime
#         is needed (the default behaviour if the data is not in a shared
#         variable) would lead to a large decrease in performance.
#         """
#         data_x, data_y = data_xy
#         shared_x = theano.shared(numpy.asarray(data_x,
#                                                dtype=theano.config.floatX),
#                                  borrow=borrow)
#         shared_y = theano.shared(numpy.asarray(data_y,
#                                                dtype=theano.config.floatX),
#                                  borrow=borrow)
#         # When storing data on the GPU it has to be stored as floats
#         # therefore we will store the labels as ``floatX`` as well
#         # (``shared_y`` does exactly that). But during our computations
#         # we need them as ints (we use labels as index, and if they are
#         # floats it doesn't make sense) therefore instead of returning
#         # ``shared_y`` we will have to cast it to int. This little hack
#         # lets ous get around this issue
#         return shared_x, T.cast(shared_y, 'int32')
#
#     test_set_x, test_set_y = shared_dataset(test_set)
#     valid_set_x, valid_set_y = shared_dataset(valid_set)
#     train_set_x, train_set_y = shared_dataset(train_set)
#
#     rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
#             (test_set_x, test_set_y)]
#     return rval






def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model
    This is demonstrated on MNIST.
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    ##################
    ##初始化逻辑回归类##
    #################
    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = np.array(T.matrix('x')) # data, presented as rasterized images
    y = np.array(T.ivector('y'))  # labels, presented as 1D vector of [int] labels


    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    ###################
    ##需要优化的损失变量##
    ###################
    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.cost_funtion(y)
    print(cost)



    # # compiling a Theano function that computes the mistakes that are made by
    # # the model on a minibatch,面向测试集
    # # 输入参数是数据批的索引号，函数计算该索引号所对应的数据批中错误分类的个数
    # test_model = theano.function(
    #     inputs=[index],
    #     outputs=classifier.errors(y),
    #     givens={
    #         x: test_set_x[index * batch_size: (index + 1) * batch_size],
    #         y: test_set_y[index * batch_size: (index + 1) * batch_size]
    #     }
    # )
    # #面向验证集
    # validate_model = theano.function(
    #     inputs=[index],
    #     outputs=classifier.errors(y),
    #     givens={
    #         x: valid_set_x[index * batch_size: (index + 1) * batch_size],
    #         y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    #     }
    # )
    #
    ##提供变量导数##
    # compute the gradient of cost with respect to theta = (W,b)
    # g_W = T.grad(cost=cost, wrt=classifier.W)
    # g_b = T.grad(cost=cost, wrt=classifier.b)

    ## cost_grad
    cost_grad = classifier.gradient_W_b(x, y)
    #
    ## 梯度下降 Gradient Decent
    theta = [classifier.W, classifier.b]
    theta = classifier.update_theta(theta, learning_rate, cost_grad)


    # updates = [(classifier.W, classifier.W - learning_rate * g_W),
    #            (classifier.b, classifier.b - learning_rate * g_b)]
    #
    # # compiling a Theano function `train_model` that returns the cost, but in
    # # the same time updates the parameter of the model based on the rules
    # # defined in `updates`
    # train_model = theano.function(
    #     inputs=[index],
    #     outputs=cost,
    #     updates=updates,
    #     givens={
    #         x: train_set_x[index * batch_size: (index + 1) * batch_size],
    #         y: train_set_y[index * batch_size: (index + 1) * batch_size]
    #     }
    # )
    # # end-snippet-3
    # ##每一次train_model函数被调用，其都会计算当前batch的损失函数，
    # ##并且进行一次梯度下降，通过输入index来调整不同batch数据。
    #
    # ###############
    # # TRAIN MODEL #
    # ###############
    # print('... training the model')
    # # early-stopping parameters
    # patience = 5000  # look as this many examples regardless
    # patience_increase = 2  # wait this much longer when a new best is
    # # found
    # improvement_threshold = 0.995  # a relative improvement of this much is
    # # considered significant
    # validation_frequency = min(n_train_batches, patience // 2)
    # # go through this many
    # # minibatche before checking the network
    # # on the validation set; in this case we
    # # check every epoch
    #
    # best_validation_loss = numpy.inf
    # test_score = 0.
    # start_time = timeit.default_timer()
    #
    # done_looping = False
    # epoch = 0
    # while (epoch < n_epochs) and (not done_looping):
    #     epoch = epoch + 1
    #     for minibatch_index in range(n_train_batches):
    #
    #         minibatch_avg_cost = train_model(minibatch_index)
    #         # iteration number
    #         iter = (epoch - 1) * n_train_batches + minibatch_index
    #
    #         if (iter + 1) % validation_frequency == 0:
    #             # compute zero-one loss on validation set
    #             validation_losses = [validate_model(i)
    #                                  for i in range(n_valid_batches)]
    #             this_validation_loss = numpy.mean(validation_losses)
    #
    #             print(
    #                 'epoch %i, minibatch %i/%i, validation error %f %%' %
    #                 (
    #                     epoch,
    #                     minibatch_index + 1,
    #                     n_train_batches,
    #                     this_validation_loss * 100.
    #                 )
    #             )
    #
    #             # if we got the best validation score until now
    #             if this_validation_loss < best_validation_loss:
    #                 # improve patience if loss improvement is good enough
    #                 if this_validation_loss < best_validation_loss * \
    #                         improvement_threshold:
    #                     patience = max(patience, iter * patience_increase)
    #
    #                 best_validation_loss = this_validation_loss
    #                 # test it on the test set
    #
    #                 test_losses = [test_model(i)
    #                                for i in range(n_test_batches)]
    #                 test_score = numpy.mean(test_losses)
    #
    #                 print(
    #                     (
    #                         '     epoch %i, minibatch %i/%i, test error of'
    #                         ' best model %f %%'
    #                     ) %
    #                     (
    #                         epoch,
    #                         minibatch_index + 1,
    #                         n_train_batches,
    #                         test_score * 100.
    #                     )
    #                 )
    #
    #                 # save the best model
    #                 with open('best_model.pkl', 'wb') as f:
    #                     pickle.dump(classifier, f)
    #
    #         if patience <= iter:
    #             done_looping = True
    #             break
    #
    # end_time = timeit.default_timer()
    # print(
    #     (
    #         'Optimization complete with best validation score of %f %%,'
    #         'with test performance %f %%'
    #     )
    #     % (best_validation_loss * 100., test_score * 100.)
    # )
    # print('The code run for %d epochs, with %f epochs/sec' % (
    #     epoch, 1. * epoch / (end_time - start_time)))
    # print(('The code for file ' +
    #        os.path.split(__file__)[1] +
    #        ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)


##当训练达到最低误差的时候，我们可以重新载入模型对新数据的标签进行预测
def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = pickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)


if __name__ == '__main__':

    sgd_optimization_mnist()

