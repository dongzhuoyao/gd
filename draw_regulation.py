import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
"""
# Fixing random state for reproducibility
np.random.seed(19680801)
# make up some data in the interval ]0, 1[
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))
"""

sgd_1 = np.load("result/regulation/SGD-backtracking-regulation1e1.npy")
sgd_0 = np.load("result/regulation/SGD-backtracking-regulation1.npy")
sgd_minus1 = np.load("result/regulation/SGD-backtracking-regulation1e-1.npy")
sgd_minus2 = np.load("result/regulation/SGD-backtracking-regulation1e-2.npy")
sgd_minus3 = np.load("result/regulation/SGD-backtracking-regulation1e-3.npy")
sgd_minus4 = np.load("result/regulation/SGD-backtracking-regulation1e-4.npy")


momentum_1 = np.load("result/regulation/SGD-momentum09-backtracking-regulation1e1.npy")
momentum_0 = np.load("result/regulation/SGD-momentum09-backtracking-regulation1.npy")
momentum_minus1 = np.load("result/regulation/SGD-momentum09-backtracking-regulation1e-1.npy")
momentum_minus2 = np.load("result/regulation/SGD-momentum09-backtracking-regulation1e-2.npy")
momentum_minus3 = np.load("result/regulation/SGD-momentum09-backtracking-regulation1e-3.npy")
momentum_minus4 = np.load("result/regulation/SGD-momentum09-backtracking-regulation1e-4.npy")


sag_1 = np.load("result/regulation/SAG-backtracking-regulation1e1.npy")
sag_0 = np.load("result/regulation/SAG-backtracking-regulation1.npy")
sag_minus1 = np.load("result/regulation/SAG-backtracking-regulation1e-1.npy")
sag_minus2 = np.load("result/regulation/SAG-backtracking-regulation1e-2.npy")
sag_minus3 = np.load("result/regulation/SAG-backtracking-regulation1e-3.npy")
sag_minus4 = np.load("result/regulation/SAG-backtracking-regulation1e-4.npy")

svrg_1 = np.load("result/regulation/SVRG-lr001-regulation1e1.npy")
svrg_0 = np.load("result/regulation/SVRG-lr001-regulation1.npy")
svrg_minus1 = np.load("result/regulation/SVRG-lr001-regulation1e-1.npy")
svrg_minus2 = np.load("result/regulation/SVRG-lr001-regulation1e-2.npy")
svrg_minus3 = np.load("result/regulation/SVRG-lr001-regulation1e-3.npy")
svrg_minus4 = np.load("result/regulation/SVRG-lr001-regulation1e-4.npy")



optimum = 0.3500
#colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

# plot with various axes scales
fig = plt.figure(1,figsize=(20,10 ))
#fig.suptitle('Regulation Paramter Experiment', fontsize=14, fontweight='bold')

ax = fig.add_subplot(221)
sgd_1_handler, = ax.plot(list(sgd_1[4]), list(sgd_1[5]), color="b", linestyle="-", marker='o', label='L2=1e1')
sgd_0_handler, = ax.plot(list(sgd_0[4]), list(sgd_0[5]), color="r", linestyle="-", marker='o', label='L2=1e0')
sgd_minus1_handler, = ax.plot(list(sgd_minus1[4]), list(sgd_minus1[5]), color="g", linestyle="-", marker='o', label='L2=1e-1')
sgd_minus2_handler, = ax.plot(list(sgd_minus2[4]), list(sgd_minus2[5]), color="m", linestyle="-", marker='o', label='L2=1e-2')
sgd_minus3_handler, = ax.plot(list(sgd_minus3[4]), list(sgd_minus3[5]), color="y", linestyle="-", marker='o', label='L2=1e-3')
sgd_minus4_handler, = ax.plot(list(sgd_minus4[4]), list(sgd_minus4[5]), color="k", linestyle="-", marker='o', label='L2=1e-4')



ax.set_title("(a). SGD")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Test Accuracy')
ax.set_ylim([0.5,0.91])
ax.legend(loc="lower right",handles=[sgd_1_handler,sgd_0_handler,sgd_minus1_handler,sgd_minus2_handler,sgd_minus3_handler,sgd_minus4_handler])



ax = fig.add_subplot(222)
momentum_1_handler, = ax.plot(list(momentum_1[4]), list(momentum_1[5]), color="b", linestyle="-", marker='o', label='L2=1e1')
momentum_0_handler, = ax.plot(list(momentum_0[4]), list(momentum_0[5]), color="r", linestyle="-", marker='o', label='L2=1e0')
momentum_minus1_handler, = ax.plot(list(momentum_minus1[4]), list(momentum_minus1[5]), color="g", linestyle="-", marker='o', label='L2=1e-1')
momentum_minus2_handler, = ax.plot(list(momentum_minus2[4]), list(momentum_minus2[5]), color="m", linestyle="-", marker='o', label='L2=1e-2')
momentum_minus3_handler, = ax.plot(list(momentum_minus3[4]), list(momentum_minus3[5]), color="y", linestyle="-", marker='o', label='L2=1e-3')
momentum_minus4_handler, = ax.plot(list(momentum_minus4[4]), list(momentum_minus4[5]), color="k", linestyle="-", marker='o', label='L2=1e-4')


ax.set_title("(b). Momemtum")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Test Accuracy')
ax.set_ylim([0.43,0.91])
ax.legend(loc="lower right",handles=[momentum_1_handler,momentum_0_handler,momentum_minus1_handler,momentum_minus2_handler,momentum_minus3_handler,momentum_minus4_handler])





ax = fig.add_subplot(223)
sag_1_handler, = ax.plot(list(sag_1[4]), list(sag_1[5]), color="b", linestyle="-", marker='o', label='L2=1e1')
sag_0_handler, = ax.plot(list(sag_0[4]), list(sag_0[5]), color="r", linestyle="-", marker='o', label='L2=1e0')
sag_minus1_handler, = ax.plot(list(sag_minus1[4]), list(sag_minus1[5]), color="g", linestyle="-", marker='o', label='L2=1e-1')
sag_minus2_handler, = ax.plot(list(sag_minus2[4]), list(sag_minus2[5]), color="m", linestyle="-", marker='o', label='L2=1e-2')
sag_minus3_handler, = ax.plot(list(sag_minus3[4]), list(sag_minus3[5]), color="y", linestyle="-", marker='o', label='L2=1e-3')
sag_minus4_handler, = ax.plot(list(sag_minus4[4]), list(sag_minus4[5]), color="k", linestyle="-", marker='o', label='L2=1e-4')


ax.set_title("(c). SAG")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Test Accuracy')
ax.set_ylim([0.78,0.9])
ax.legend(loc="lower right",handles=[sag_1_handler,sag_0_handler,sag_minus1_handler,sag_minus2_handler,sag_minus3_handler,sag_minus4_handler])



ax = fig.add_subplot(224)
svrg_1_handler, = ax.plot(list(svrg_1[4]), list(svrg_1[5]), color="b", linestyle="-", marker='o', label='L2=1e1')
svrg_0_handler, = ax.plot(list(svrg_0[4]), list(svrg_0[5]), color="r", linestyle="-", marker='o', label='L2=1e0')
svrg_minus1_handler, = ax.plot(list(svrg_minus1[4]), list(svrg_minus1[5]), color="g", linestyle="-", marker='o', label='L2=1e-1')
svrg_minus2_handler, = ax.plot(list(svrg_minus2[4]), list(svrg_minus2[5]), color="m", linestyle="-", marker='o', label='L2=1e-2')
svrg_minus3_handler, = ax.plot(list(svrg_minus3[4]), list(svrg_minus3[5]), color="y", linestyle="-", marker='o', label='L2=1e-3')
svrg_minus4_handler, = ax.plot(list(svrg_minus4[4]), list(svrg_minus4[5]), color="k", linestyle="-", marker='o', label='L2=1e-4')


ax.set_title("(d). SVRG")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Test Accuracy')
ax.set_ylim([0.78,0.9])
ax.legend(loc="lower right",handles=[svrg_1_handler,svrg_0_handler,svrg_minus1_handler,svrg_minus2_handler,svrg_minus3_handler,svrg_minus4_handler])



plt.savefig('regulation.pdf', format='pdf')
#plt.show()
