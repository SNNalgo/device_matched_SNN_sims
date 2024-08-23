import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

#-1.1V Vgs case
vth = 0.7
Ith = 8e-10
currs = np.array([5.08E-11, 9.64E-11, 1.88E-10, 3.79E-10, 7.51E-10, 1.44E-09, 2.67E-09, 5.10E-09, 1.02E-08, 2.11E-08, 4.34E-08, 8.58E-08])
ref_fs = np.array([0, 0, 0, 0, 0, 19588.61938, 97950.77949, 128093.3453, 217221.135, 232379.6232, 262963.9379, 285510.1458])
Cs = np.logspace(np.log10(20e-15), np.log10(40e-15), 10) #sweep Capacitances

#-1V Vgs case
#vth = 0.7
#Ith = 4.5e-10
#currs = np.array([2.78E-11, 5.06E-11, 9.88E-11, 2.02E-10, 4.34E-10, 9.10E-10, 1.80E-09, 3.51E-09, 7.10E-09])
#ref_fs = np.array([0, 0, 0, 0, 0, 24231.58237, 63431.24734, 119295.4364, 198931.9092])
#Cs = np.logspace(np.log10(2e-14), np.log10(2e-13), 10) #sweep Capacitances

#-0.9V Vgs case
#vth = 0.8
#Ith = 5.5e-10
#currs = np.array([1.69E-11, 2.88E-11, 5.18E-11, 1.06E-10, 2.31E-10, 5.18E-10, 1.10E-09, 2.26E-09, 4.69E-09, 9.97E-09, 2.03E-08, 4.24E-08, 7.99E-08])
#ref_fs = np.array([0, 0, 0, 0, 0, 0, 12503.2854, 51233.39658, 105168.9652, 181669.3944, 222049.3443, 232379.6232, 249812.4531])
#Cs = np.logspace(np.log10(20e-15), np.log10(35e-15), 10) #sweep Capacitances

vrst = 0
R = (vth - vrst)/Ith

spk_fs_C = []
for C in Cs:
    tau = R*C
    time_step = tau/20
    lif1 = snn.Lapicque(R=R, C=C, time_step=time_step, threshold=vth, reset_mechanism='zero')
    num_steps = 100 #Tsim = num_steps*time_step = 20*tau
    spk_fs_I = []
    for curr in currs:
        mem = torch.zeros(1)
        spk_out = torch.zeros(1)
        spk_rec = [spk_out]
        for step in range(num_steps):
            spk_out, mem = lif1(curr, mem)
            spk_rec.append(spk_out)
        spk_rec = torch.stack(spk_rec).numpy()
        spk_f = np.sum(spk_rec)/(num_steps*time_step)
        spk_fs_I.append(spk_f)
    spk_fs_C.append(spk_fs_I)
    print("C : ",  C/1e-15, "fF")
    print("tau : ",  tau/1e-6, "us")

spk_fs_C_leaky = []
for C in Cs:
    tau = R*C
    time_step = tau/20
    lif1 = snn.Leaky(beta=1-time_step/tau, threshold=vth, reset_mechanism='zero')
    num_steps = 100
    spk_fs_I = []
    for curr in currs:
        mem = torch.zeros(1)
        spk_out = torch.zeros(1)
        spk_rec = [spk_out]
        for step in range(num_steps):
            spk_out, mem = lif1(curr*time_step/C, mem)
            spk_rec.append(spk_out)
        spk_rec = torch.stack(spk_rec).numpy()
        spk_f = np.sum(spk_rec)/(num_steps*time_step)
        spk_fs_I.append(spk_f)
    spk_fs_C_leaky.append(spk_fs_I)
    print("C : ",  C/1e-15, "fF")
    print("tau : ",  tau/1e-6, "us")

plt.figure()
plt.xlabel("I(nA)")
plt.ylabel("f(kHz)")
legend = ['reference fvsI']
plt.plot(currs/1e-9, np.array(ref_fs)/1000, linestyle='--', marker='d')
for i in range(Cs.shape[0]):
    C = Cs[i]
    plt.plot(currs/1e-9, np.array(spk_fs_C[i])/1000, marker='s')
    legend.append("C = " + str(C/1e-15) + "fF")
plt.legend(legend)
plt.title("Lapicque model")
plt.grid()
plt.savefig("./lapicque_model.png", dpi=300)
plt.show()

plt.figure()
plt.xlabel("I(nA)")
plt.ylabel("f(kHz)")
legend = ['reference fvsI']
plt.plot(currs/1e-9, np.array(ref_fs)/1000, linestyle='--', marker='d')
for i in range(Cs.shape[0]):
    C = Cs[i]
    plt.plot(currs/1e-9, np.array(spk_fs_C_leaky[i])/1000, marker='s')
    legend.append("C = " + str(C/1e-15) + "fF")
plt.legend(legend)
plt.title("Equivalent Leaky model")
plt.grid()
plt.savefig("./leaky_model.png", dpi=300)
plt.show()
