import os, sys
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.axisartist.axislines import Subplot

matplotlib.rc('xtick', labelsize=17)
matplotlib.rc('ytick', labelsize=17)

def parse_log(file_name):
    rounds = []
    accu = []
    loss = []
    sim = []

    for line in open(file_name, 'r'):

        search_train_accu = re.search(r'At round (.*) training accuracy: (.*)', line, re.M | re.I)
        if search_train_accu:
            rounds.append(int(search_train_accu.group(1)))

        search_test_accu = re.search(r'At round (.*) testing accuracy: (.*)', line, re.M | re.I)
        if search_test_accu:
            accu.append(float(search_test_accu.group(2)))

        search_loss = re.search(r'At round (.*) training loss: (.*)', line, re.M | re.I)
        if search_loss:
            loss.append(float(search_loss.group(2)))

        search_loss = re.search(r'gradient difference: (.*)', line, re.M | re.I)
        if search_loss:
            sim.append(float(search_loss.group(1)))

    return rounds, sim, loss, accu

idx = 0
f = plt.figure(figsize=[5, 4])


ax = plt.subplot(1, 1, 1)

rounds1, sim1, losses1, test_accuracies1 = parse_log("log_synthetic/iid_fedavg_c10_e20")
rounds2, sim2, losses2, test_accuracies2 = parse_log("log_synthetic/iid_fedprox_c10_e20")
rounds3, sim3, losses3, test_accuracies3 = parse_log("log_synthetic/iid_feddane_c10_e20")

plt.plot(np.asarray(rounds1[:len(losses1)]), np.asarray(losses1), '--', linewidth=3.0, label='FedAvg',
             color="#17becf")
plt.plot(np.asarray(rounds2[:len(losses2)]), np.asarray(losses2), linewidth=3.0, label='FedProx',
             color="#e377c2")
plt.plot(np.asarray(rounds3[:len(losses3)]), np.asarray(losses3), ':', linewidth=3.0, label='FedDane',
             color="#ff7f0e")

plt.xlabel("# Rounds", fontsize=22)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)


plt.ylabel('Training Loss', fontsize=22)
leg = plt.legend(fontsize=20)
leg.get_frame().set_linewidth(0.0)

plt.title("Synthetic (IID)", fontsize=22)
ax.tick_params(color='#dddddd')
ax.spines['bottom'].set_color('#dddddd')
ax.spines['top'].set_color('#dddddd')
ax.spines['right'].set_color('#dddddd')
ax.spines['left'].set_color('#dddddd')

plt.tight_layout()

f.savefig("iid_low_participation.pdf")
