from load_mnist import read
from random import  shuffle,randint
from collections import deque
import numpy as np
import math
import pickle,random


batch_size = 128
validate_interval = 4


#0.001,128 perfect!

momentum_rho = 0.9
backtracing_step = 0.94

bb_lower_bound = 1e-10
bb_upper_bound = 1e20
bb_sigma = 0.93
bb_reseted_step_size = 0.0001
bb_M = 18
bb_M_deque = deque(maxlen=bb_M)



train_imgs,train_labels,val_imgs,val_labels = read(dataset="training")
test_imgs,test_labels = read(dataset="testing")

print("training_imgs: {}".format(len(train_imgs)))
print("validation_imgs: {}".format(len(val_imgs)))
print("testing_imgs: {}".format(len(test_imgs)))
train_batch_num = int(math.floor(len(train_imgs)/batch_size))
val_batch_num = int(math.floor(len(val_imgs)/batch_size))
test_batch_num = int(math.floor(len(test_imgs)/batch_size))
print("train_batch_num: {}".format(train_batch_num))
print("val_batch_num: {}".format(val_batch_num))
print("test_batch_num: {}".format(test_batch_num))
w = np.random.normal(0,0.01,(785,))
g_history = np.zeros((785,))
w_history = np.zeros((785,))
alpha_history = 0

###SVG
svg_gradient_history = np.zeros((train_batch_num,785))

###SVRG
svrg_w_history = np.zeros((785,))
svrg_m = batch_size
import os
#1,modify name,2,modify method
output_name =  os.path.join("result","SAGA-backtracking-l2")
optimize_method ="svg"
regulation_type = "l2"
regulation = 1e-4
step_size = 0.01
record_terminal = 1
epoch_num = 30
effective_pass = 25

#sgd,sag,saga,svrg,

def optimize(epoch_index,batch_index_one_epoch, total_batch_num, xs, ys):
    #exponential_decay
    loss = calculate_loss_with_regulation(xs, ys, w)
    #sgd(xs,ys,epoch_index,step_size_strategy = "backtracking")
    sag(epoch_index,batch_index_one_epoch, total_batch_num, xs, ys, sub_mod="saga",step_size_strategy="backtracking")
    #sgd(xs, ys, epoch_index,step_size_strategy="backtracking")
    #svrg(batch_index_one_epoch, total_batch_num, xs, ys, lr=0.01)
    #momentum(xs, ys, epoch_index,nesterov=True, step_size_strategy="backtracking")
    return loss


def calculate_grad_with_regularization000(xs, ys, w):
    gradient = 0
    for i in range(batch_size):
        gradient += (-np.exp(-ys[i] * np.inner(w, xs[i])) * (ys[i] * xs[i]) / (1 + np.exp(-ys[i] * np.inner(w, xs[i]))))

    gradient /= batch_size


    if regulation_type=="l1":
        gradient += regulation * np.sign(w)
    elif regulation_type=="l2":
        gradient += regulation * w
    else:
        print("error regulation type")
        exit()

    return gradient


def calculate_grad_with_regularization(xs, ys, w):

    new_w = w.reshape(785,1)
    e = np.exp(np.multiply(-ys,np.dot(xs,new_w)))
    left =  np.divide(e,1+e)
    left_more = np.transpose(np.multiply(-ys,left))
    gradient = np.dot(left_more,xs)/batch_size


    gradient = np.transpose(gradient)
    gradient = np.squeeze(gradient)

    if regulation_type == "l1":
        gradient += regulation * np.sign(w)
    elif regulation_type == "l2":
        gradient += regulation * w
    else:
        print("error regulation type")
        exit()

    return gradient


def calculate_loss_with_regulation(xs, ys, w):
    new_w = w.reshape(785, 1)
    aa = -ys
    bb = np.dot(xs,new_w)
    #print("aver: {}".format(np.average(np.exp(np.multiply(aa,bb)))))
    loss = np.log(1+np.exp(np.multiply(aa,bb)))
    #print("loss: {}".format(loss))
    loss = np.average(loss)


    if regulation_type=="l1":
        loss += regulation * np.linalg.norm(w, 1)
    elif regulation_type=="l2":
        loss += regulation * np.linalg.norm(w, 2)
    else:
        print("error regulation type")
        exit()

    return loss


def calculate_loss_with_regulation00000(xs, ys, w):
    # calculate loss
    loss = 0
    for i in range(batch_size):
        loss += np.log(1 + np.exp(-ys[i] * np.inner(w, xs[i])))

    loss /= batch_size

    if regulation_type=="l1":
        loss += regulation * np.linalg.norm(w, 1)
    elif regulation_type=="l2":
        loss += regulation * np.linalg.norm(w, 2)
    else:
        print("error regulation type")
        exit()

    return loss


def svrg(batch_index_one_epoch,total_batch_num,xs,ys,lr=0.01):
    global w,g_history,w_history,alpha_history,svrg_w_history

    _tilde_w = svrg_w_history

    _tilde_mu = 0
    for ii in range(total_batch_num):
        current_xs = train_imgs[ii * batch_size:(ii + 1) * batch_size]
        current_ys = train_labels[ii * batch_size:(ii + 1) * batch_size]
        _tilde_mu +=calculate_grad_with_regularization(current_xs,current_ys,w)
    _tilde_mu /= total_batch_num

    _w_old = _tilde_w


    for i in range(svrg_m):
        f_s = randint(0,total_batch_num-1)
        #print np.inner(_w_old, xs[_i_t])
        current_xs = train_imgs[f_s*batch_size:(f_s+1)*batch_size]
        current_ys = train_labels[f_s * batch_size:(f_s + 1) * batch_size]
        _w = _w_old -lr*(
            calculate_grad_with_regularization(current_xs,current_ys,_w_old)-calculate_grad_with_regularization(current_xs,current_ys,_tilde_w)+_tilde_mu
        )
        _w_old = _w

    #_w += regulation * np.sign(_w)

    w = _w

    svrg_w_history = _w





def sag(epoch_index,batch_index_one_epoch,total_batch_num,xs,ys,step_size_strategy = "backtracking",sub_mod="sag"):
    global w,g_history,w_history,alpha_history

    current_g = calculate_grad_with_regularization(xs, ys, w)
    if sub_mod=="sag":
        g = (current_g - svg_gradient_history[batch_index_one_epoch,:])/total_batch_num
    elif sub_mod=="saga":
        g = (current_g - svg_gradient_history[batch_index_one_epoch, :])
    else:
        print("invalid svg method: {}".format(sub_mod))
        exit()

    for ii in range(total_batch_num):
        g += svg_gradient_history[ii,:]/total_batch_num



    #backtracking line search
    cur_sz = 1

    # <<Optimization theory and methods - nonlinear programming by Wenyu Sun, Ya-xiang Yuan>>,,Backtracking line search
    if step_size_strategy =="bb":
            cur_loss = calculate_loss_with_regulation(xs, ys, w)

            if alpha_history == 0:
                alpha = 1/bb_reseted_step_size
            else:
                alpha = alpha_history

            #if alpha < bb_lower_bound or alpha > bb_upper_bound:
            #    alpha = 1/bb_reseted_step_size
            if alpha < bb_lower_bound:
                alpha = bb_lower_bound

            if alpha > bb_upper_bound:
                alpha = bb_upper_bound

                #print("bb step size out of boundry,reset step size to {}".format(bb_reseted_step_size))


            bb_M_deque.append(cur_loss)
            tmp_list = list(bb_M_deque)
            tmp_list.sort()
            max_M = tmp_list[-1]

            while calculate_loss_with_regulation(xs, ys, w-alpha*g) > max_M - 0.5*alpha*np.inner(g, g):
                    #update alpha
                    #current_sigma = random.uniform(bb_sigma1, bb_sigma2)
                    alpha = bb_sigma* alpha

            bb_step = alpha
            #print("bb_step: {}".format(bb_step))
            w += (-bb_step*g)
            #calculate new alpha
            current_yk = calculate_grad_with_regularization(xs, ys, w) - g
            alpha_history = - (alpha*np.inner(g,g))/np.inner(g,current_yk)


    elif step_size_strategy=="backtracking":
        while calculate_loss_with_regulation(xs, ys, w-cur_sz*g) > calculate_loss_with_regulation(xs, ys, w) -cur_sz*math.pow(np.linalg.norm(g, 2), 2)/2:
                cur_sz = cur_sz * backtracing_step
                #print cur_sz
    elif step_size_strategy == "exponential_decay":
        cur_sz = 1 * math.pow(0.9, epoch_index / 3)
    else:
        cur_sz = step_size




    w_history = np.copy(w)
    w += (-cur_sz*g)
    g_history = np.copy(g)



def sgd(xs,ys,epoch_index,step_size_strategy = "backtracking",momentum=0):
    global w,g_history,w_history,alpha_history

    g = calculate_grad_with_regularization(xs, ys, w)


    #backtracking line search
    cur_sz = 1

    # <<Optimization theory and methods - nonlinear programming by Wenyu Sun, Ya-xiang Yuan>>,,Backtracking line search
    if step_size_strategy =="bb":

            cur_loss = calculate_loss_with_regulation(xs, ys, w)

            if alpha_history == 0:
                alpha = 1/bb_reseted_step_size
            else:
                alpha = alpha_history
                #print("bb step size out of boundry,reset step size to {}".format(bb_reseted_step_size))


            bb_M_deque.append(cur_loss)
            tmp_list = list(bb_M_deque)
            tmp_list.sort()
            max_M = tmp_list[-1]

            i=0
            while calculate_loss_with_regulation(xs, ys, w-alpha*g) > max_M - 0.5*alpha*np.inner(g, g):
                    #update alpha
                    #current_sigma = random.uniform(bb_sigma1, bb_sigma2)
                    i += 1
                    alpha = bb_sigma* alpha
            #print i
            bb_step = alpha
            #print("bb_step: {}".format(bb_step))
            w += (-bb_step*g)
            #calculate new alpha
            current_yk = calculate_grad_with_regularization(xs, ys, w) - g
            alpha_history = - (alpha*np.inner(g,g))/np.inner(g,current_yk)

            if alpha_history < bb_lower_bound:
                alpha_history = bb_lower_bound

            if alpha_history > bb_upper_bound:
                alpha_history = bb_upper_bound


    elif step_size_strategy=="backtracking":
        while calculate_loss_with_regulation(xs, ys, w-cur_sz*g) > calculate_loss_with_regulation(xs, ys, w) -cur_sz*math.pow(np.linalg.norm(g, 2), 2)/2:
                cur_sz = cur_sz * backtracing_step
                #print cur_sz

    elif step_size_strategy=="exponential_decay":
        cur_sz = 1*math.pow(0.9,epoch_index/3)

    else:
        cur_sz = step_size

    #print("epoch num:{} ,current lr: {}".format(epoch_index,cur_sz))


    w_history = np.copy(w)
    w += (-cur_sz*g)
    g_history = np.copy(g)



def momentum(xs, ys,epoch_index, nesterov=False,step_size_strategy = "backtracking"):
    global g_history,w
    # subgradient
    gradient = calculate_grad_with_regularization(xs, ys, w)

    # backtracking line search
    cur_sz = 1

    if step_size_strategy=="backtracking":
        while 1:
            #<<Optimization theory and methods - nonlinear programming by Wenyu Sun, Ya-xiang Yuan>>,P108,Backtracking line search
            if calculate_loss_with_regulation(xs, ys, w - cur_sz * gradient) > calculate_loss_with_regulation(xs, ys, w) - cur_sz * math.pow(np.linalg.norm(gradient, 2), 2) / 2:
                cur_sz = cur_sz * backtracing_step

            else:
                break
        # print("step size: {}".format(cur_sz))

    elif step_size_strategy=="exponential_decay":
        cur_sz = 1*math.pow(0.9,epoch_index/3)
    else:

        cur_sz = step_size

    if nesterov:
        to_be_added = (momentum_rho * g_history - cur_sz * calculate_grad_with_regularization(xs, ys, w + momentum_rho * g_history))
    else:
        to_be_added = (momentum_rho * g_history - cur_sz * gradient)

    w += to_be_added
    g_history = to_be_added



def predict(x):
    f = np.inner(w,x)

    if f > 0:
        return 1
    else:
        return -1

def do_predict():
    hit = 0
    #do testing
    for te in range(len(test_imgs)):
        cur_result = predict(test_imgs[te])
        if cur_result == test_labels[te]:
            hit += 1
    acc =hit*1.0/len(test_imgs)
    print("test accuracy: {}".format(acc))
    return acc


val_index = 0


result_np = np.zeros(shape=(6,effective_pass+1))
min_loss = 999
best_acc = 0
matplot_index = 0
for ep in range(epoch_num):
    #shuffle(train_imgs)
    batch_index = 0
    for bn in range(train_batch_num):
        batch_index += 1
        loss = optimize(ep,bn,train_batch_num,train_imgs[bn * batch_size:(bn + 1) * batch_size], train_labels[bn * batch_size:(bn + 1) * batch_size])
        if batch_index%100 == 0:
            print("epoch  {},  batch-{}-loss: {}".format(ep,bn,loss))



    if matplot_index > effective_pass:
        print("best acc: {}, min loss: {}".format(best_acc, min_loss))
        # with open(output_name, 'wb') as fp:
        #    pickle.dump(result_dict, fp)
        np.save(output_name, result_np)

        exit()

    print("cur: {}".format(matplot_index))
    x_coordinate = matplot_index
    matplot_index += 1
    #train loss
    result_np[0, x_coordinate] = x_coordinate
    result_np[1, x_coordinate] = loss


    validate_loss = optimize(ep,val_index,val_batch_num,val_imgs[val_index * batch_size:(val_index + 1) * batch_size],
                             val_labels[val_index * batch_size:(val_index + 1) * batch_size])
    val_index = (val_index + 1 + val_batch_num) % val_batch_num
    #validate loss
    result_np[2, x_coordinate] = x_coordinate
    result_np[3, x_coordinate] = validate_loss
    print("******************epoch  {},  batch-{}-validation loss: {}".format(ep, batch_index, validate_loss))

    acc = do_predict()
    #acc
    result_np[4, x_coordinate] = x_coordinate
    result_np[5, x_coordinate] = acc

    if acc > best_acc:
        best_acc = acc
        min_loss = loss









