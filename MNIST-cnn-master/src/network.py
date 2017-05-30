import numpy as np

import layers as l
import utils as u
from random import  randint


class NeuralNetwork():
    def __init__(self, layers, loss_func):
        assert len(layers) > 0

        assert isinstance(layers[0], l.InputLayer)
        self.input_layer = layers[0]

        assert isinstance(layers[-1], l.FullyConnectedLayer)
        self.output_layer = layers[-1]

        self.layers = [(prev_layer, layer) for prev_layer, layer in zip(layers[:-1], layers[1:])]

        self.loss_func = loss_func

        for prev_layer, layer in self.layers:
            layer.connect_to(prev_layer)



    def set_extra_data(self,train_data,batch_size,batch_amount):
        self.train_data = train_data
        self.batch_size = batch_size
        self.batch_amount = int(batch_amount)

    def feedforward(self, x):
        self.input_layer.z = x
        self.input_layer.a = x

        for prev_layer, layer in self.layers:
            layer.feedforward(prev_layer)



    def backpropagate_svrg(self, batch, lr):

        history_w = {layer: np.zeros_like(layer.w) for _, layer in self.layers}
        history_b = {layer: np.zeros_like(layer.b) for _, layer in self.layers}

        _tilde_w_w,_tilde_w_b = history_w,history_b
        _mu = 0

        cur_data_list = [self.train_data[i:i + self.batch_size] for i in range(0, len(self.train_data), self.batch_size)]
        #set history w,b
        for i in range(self.batch_amount):
            cur_data = cur_data_list[i]

            for x, y in cur_data:
                self.feedforward(x)

                # propagate the error backward
                loss = self.loss_func(self.output_layer.a, y)
                delta = loss * self.output_layer.der_act_func(self.output_layer.z, y)
                for prev_layer, layer in reversed(self.layers):
                    der_w, der_b, prev_delta = layer.backpropagate(prev_layer, delta)
                    history_w[layer] += der_w
                    history_b[layer] += der_b
                    delta = prev_delta

        _tilde_mu_w,_tilde_mu_b = {},{}
        for prev_layer, layer in reversed(self.layers):
            _tilde_mu_w[layer],_tilde_mu_b[layer]  = history_w[layer]/self.batch_amount,history_b[layer]/self.batch_amount

        old_w,old_b = _tilde_w_w,_tilde_w_b
        new_w, new_b = {},{}


        for j in range(self.batch_amount*2):
            i_k = randint(0,self.batch_amount-1)

            another_w = {layer: np.zeros_like(layer.w) for _, layer in self.layers}
            another_b = {layer: np.zeros_like(layer.b) for _, layer in self.layers}
            sub_w = {layer: np.zeros_like(layer.w) for _, layer in self.layers}
            sub_b = {layer: np.zeros_like(layer.b) for _, layer in self.layers}

            # sub
            cur_data = cur_data_list[i_k]

            for x, y in cur_data:
                self.feedforward(x)

                # propagate the error backward
                loss = self.loss_func(self.output_layer.a, y)
                delta = loss * self.output_layer.der_act_func(self.output_layer.z, y)
                for prev_layer, layer in reversed(self.layers):
                    der_w, der_b, prev_delta = layer.backpropagate(prev_layer, delta)
                    sub_w[layer] += der_w
                    sub_b[layer] += der_b
                    delta = prev_delta


            backup_layers_w,backup_layers_b = {},{}
            for prev_layer, layer in reversed(self.layers):
                backup_layers_w[layer] = layer.w
                backup_layers_b[layer] = layer.b

                layer.w = old_w[layer]
                layer.b = old_b[layer]

            for x, y in cur_data:
                self.feedforward(x)

                # propagate the error backward
                loss = self.loss_func(self.output_layer.a, y)
                delta = loss * self.output_layer.der_act_func(self.output_layer.z, y)
                for prev_layer, layer in reversed(self.layers):
                    der_w, der_b, prev_delta = layer.backpropagate(prev_layer, delta)
                    another_w[layer] += der_w
                    another_b[layer] += der_b
                    delta = prev_delta

            #recover w,b
            for prev_layer, layer in reversed(self.layers):
                layer.w = backup_layers_w[layer]
                layer.b = backup_layers_b[layer]

            for prev_layer, layer in reversed(self.layers):
                new_w[layer] = old_w[layer] - lr * (another_w[layer] -sub_w[layer]+_tilde_mu_w[layer])
                new_b[layer] = old_b[layer] - lr * (another_b[layer] -sub_b[layer] + _tilde_mu_b[layer])

    def backpropagate(self, batch, optimizer):
        sum_der_w = {layer: np.zeros_like(layer.w) for _, layer in self.layers}
        sum_der_b = {layer: np.zeros_like(layer.b) for _, layer in self.layers}

        for x, y in batch:
            self.feedforward(x)

            # propagate the error backward
            loss = self.loss_func(self.output_layer.a, y)
            delta = loss * self.output_layer.der_act_func(self.output_layer.z, y)
            for prev_layer, layer in reversed(self.layers):
                der_w, der_b, prev_delta = layer.backpropagate(prev_layer, delta)
                sum_der_w[layer] += der_w
                sum_der_b[layer] += der_b
                delta = prev_delta

        # update weights and biases
        optimizer.apply(self.layers, sum_der_w, sum_der_b, len(batch),{})

                        #metadata={"gradient_history":self.gradient_history,"train_data":self.train_data,"batch_size":self.batch_size,"batch_amount":self.batch_amount})


def train(net, optimizer, num_epochs, batch_size, trn_set, vld_set=None):
    assert isinstance(net, NeuralNetwork)
    assert num_epochs > 0
    assert batch_size > 0

    trn_x, trn_y = trn_set
    inputs = [(x, y) for x, y in zip(trn_x, trn_y)]

    net.set_extra_data(train_data=inputs,batch_size=batch_size,batch_amount=len(inputs)/batch_size)

    for i in range(num_epochs):
        #np.random.shuffle(inputs)

        # divide input observations into batches
        batches = [inputs[j:j+batch_size] for j in range(0, len(inputs), batch_size)]
        inputs_done = 0
        for j, batch in enumerate(batches):
            net.backpropagate_svrg(batch,lr=0.01)
            #net.backpropagate(batch,optimizer)
            inputs_done += len(batch)
            accuracy = test(net, vld_set)
            print("acc: {}".format(accuracy))
            u.print("Epoch %02d %s [%d/%d]" % (i+1, u.bar(inputs_done, len(inputs)), inputs_done, len(inputs)), override=True)

        if vld_set:
            # test the net at the end of each epoch
            u.print("Epoch %02d %s [%d/%d] > Testing..." % (i+1, u.bar(inputs_done, len(inputs)), inputs_done, len(inputs)), override=True)
            accuracy = test(net, vld_set)
            u.print("Epoch %02d %s [%d/%d] > Validation accuracy: %0.2f%%" % (i+1, u.bar(inputs_done, len(inputs)), inputs_done, len(inputs), accuracy*100), override=True)
        u.print()

def test(net, tst_set):
    assert isinstance(net, NeuralNetwork)

    tst_x, tst_y = tst_set
    tests = [(x, y) for x, y in zip(tst_x, tst_y)]

    accuracy = 0
    for x, y in tests:
        net.feedforward(x)
        if np.argmax(net.output_layer.a) == np.argmax(y):
            accuracy += 1
    accuracy /= len(tests)
    return accuracy
