import numpy as np
from numpy import tanh
from time import time

sigmoid= lambda x: 1/(1+np.exp(-x))
sigmoid_derivative = lambda x: sigmoid(x) * (1-sigmoid(x))
relU = lambda x: np.max(0, x)
relU_derivative = lambda x: 1 if x > 0 else 0
tanh_derivative = lambda x: 1.0 - np.square(tanh(x))


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))



def load_data(input,output,test_size=0.2,normalize=1.0):
    x = np.loadtxt(input, np.float)
    y = np.array([np.loadtxt(output, np.float)])
    train_with_labels = np.concatenate((x, y.T), axis=1)
    np.random.shuffle(train_with_labels)
    num_examples =(int)(test_size*train_with_labels.shape[0])
    #print "num examples {0}".format(num_examples)
    # return train_x train_y test_x test_y
    return train_with_labels[0:20, 0:-1]/normalize, train_with_labels[0:20, -1], train_with_labels[20:,0:-1]/normalize,train_with_labels[20:, -1]


class FCN:
    def __init__(self, input_size, hidden_layer_size, activation_func, derivative_activation_func, output_size = 10):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.activation_func = activation_func
        self.derivative_activation_func = derivative_activation_func
        self.output_size = output_size
        self.w1, self.b1, self.w2, self.b2 = self.initialize_weights_and_bias(hidden_layer_size, input_size, output_size)

    def initialize_weights_and_bias(self, hidden_layer_size, input_size, output_size):
        b1 = np.array([np.random.uniform(-0.2, 0.2) for _ in range(hidden_layer_size)])
        b2 = np.array([np.random.uniform(-0.2, 0.5) for _ in range(output_size)])
        w1 = np.array([np.random.uniform(-0.3, 0.3) for _ in range(hidden_layer_size * input_size)]).reshape(hidden_layer_size, input_size)
        w2 = np.array([np.random.uniform(-0.2, 0.2) for _ in range(hidden_layer_size * output_size)]).reshape(output_size, hidden_layer_size)
        return w1, b1, w2, b2

    def predict(self, x):
        h1 = self.activation_func(self.w1.dot(x)+self.b1)
        prob_vector = softmax(self.w2.dot(h1)+self.b2)
        return h1, prob_vector

    def back_propogation(self, x, y, lr=0.0, param=None):
        y = (int)(y)
        if param is None:
            h1, prob_vector = self.predict(x)
        else:
            h1, prob_vector = param

        w2_grad = np.outer(prob_vector, h1)
        w2_grad[y] -= h1

        b2_grad = np.copy(prob_vector)
        b2_grad[y] -= 1

        temp = np.copy(prob_vector).dot(self.w2) - self.w2[y, :]
        derivative = self.derivative_activation_func(self.w1.dot(x)+self.b1)

        b1_grad = temp*derivative
        w1_grad = np.outer(b1_grad, x)

        if lr != 0.0:
            self.update_weights(w1_grad,  b1_grad, w2_grad, b2_grad, lr)
        return -np.log(prob_vector[y])

    def update_weights(self, w1_grad, b1_grad, w2_grad, b2_grad, lr):
        self.w1 -= w1_grad * lr
        self.b1 -= b1_grad * lr
        self.w2 -= w2_grad * lr
        self.b2 -= b2_grad * lr






def main(epocs= 30,lr=0.1, layer_size=200, noramlized=255.0, activation_func= (sigmoid, sigmoid_derivative)):
    train_x,train_y,test_x,test_y = load_data("train_x", "train_y", normalize=noramlized)
    train_y = train_y.astype(int)
    nn= FCN(train_x.shape[1], layer_size, activation_func[0], activation_func[1], 10)
    num_examples = train_x.shape[0]

    indexes = range(train_x.shape[0])
    for epoc in range(epocs):
        correct_pred = avg_loss = 0.0
        np.random.shuffle(indexes)
        for i in indexes:
            h1, prob_vector = nn.predict(train_x[i])
            avg_loss += nn.back_propogation(train_x[i], train_y[i], lr, (h1, prob_vector))
            correct_pred += (prob_vector.argmax() == train_y[i])
        #print "----------------------------epoc: {0}-----------------------------------".format(epoc)
        #print "avg loss: {0}\naccuracy: {1}".format(avg_loss/num_examples, correct_pred/num_examples*100)
    correct_pred = 0.0
    #print "++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    #print "++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    #print "++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    for x,y in zip(test_x,test_y):
        correct_pred += (nn.predict(x)[1].argmax() == (int)(y))
     #   print "{0}, {1}".format(nn.predict(x)[1].argmax(), (int)(y))
    #print "real accuracy is: {0}".format(correct_pred/test_y.shape[0]*100)
    return correct_pred/test_y.shape[0]*100


if __name__=='__main__':
    accuracy = 0
    params = {}
    epocs = range(30, 200, 25)
    learning_rates = [0.01, 0.03, 0.05]
    layer_size = [200, 220, 250, 300, 340]
    activation_functions = [(sigmoid, sigmoid_derivative)]
    t = time()

    for e in epocs:
        for lr in learning_rates:
            for ls in layer_size:
                for af in activation_functions:
                    start = time()
                    temp_params = {"epocs": e, "lr": lr, "ls": ls, "af": af}
                    try:
                        temp_accuracy = main(e, lr, ls, activation_func=af)
                    except Exception:
                        print "+---------------------EXCEPTION!!!----------------+"
                        print "params: {0}".format(temp_params)
                        print "+---------------------EXCEPTION!!!----------------+"
                        continue
                    if temp_accuracy > accuracy:
                        accuracy = temp_accuracy
                        params = temp_params
                    print "+-------------------------------------------------+"
                    print "accuracy: {0}\nparams: {1}\ntime: {2}".format(temp_accuracy, temp_params, time()-start)
                    print "+-------------------------------------------------+"
    print "*********************************************"
    print "**************and the winner is:*************"
    print "accuracy: {0}\nparams: {1}\ntotal time: {2}".format(accuracy, params, float((time()-t)/60))
    print "*********************************************"
