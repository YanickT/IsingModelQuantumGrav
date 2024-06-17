import os
import numpy as np
import matplotlib.pyplot as plt
import math

path1 = "C:/Users/yanic/Documents/GitHub/IsingModelQuantumGrav/IsingModelQuantumGrav/quencing_data/quencing_data_0_20_3_11_12/"
path2 = "C:/Users/yanic/Documents/GitHub/IsingModelQuantumGrav/IsingModelQuantumGrav/quencing_data/quencing_data_0_10_3_11_12/"
path3 = "C:/Users/yanic/Documents/GitHub/IsingModelQuantumGrav/IsingModelQuantumGrav/quencing_data/quencing_data_0_10_3_7_14_close/"
path4 = "C:/Users/yanic/Documents/GitHub/IsingModelQuantumGrav/IsingModelQuantumGrav/quencing_data/quencing_data_0_20_3_11_12/"
path5 = "C:/Users/yanic/Documents/GitHub/IsingModelQuantumGrav/IsingModelQuantumGrav/quencing_data/quencing_data_0_20_3_11_12_long/"
paths = [path3] #, path2]
# paths = [path4] #, path2]


def get_data(path):
    free_energies = []
    files = os.listdir(path)
    for file in files:
        with open(f"{path}{file}", "r") as doc:
            lines = doc.readlines()
        nodes = int(lines[5][:-1].split(";")[1])
        energy = float(lines[8][:-1].split(";")[1])
        j = float(lines[9].split(";")[1])
        h = float(lines[12].split(";")[1])
        sides = int(lines[4].split(";")[1])
        f0 = -nodes * math.log(2)
        betas, es, _ = tuple(zip(*[line.split(";") for line in lines[19:]]))
        betas = [float(beta) for beta in betas]
        es = [float(e) for e in es]
        integral = np.trapz(es, x=betas) + f0
        free_energies.append((energy, integral))

    free_energies.sort(key=lambda x: x[0])
    return tuple(zip(*free_energies)), j, h, sides


for path in paths:
    (energies, integrals), j, h, sides = get_data(path)
    plt.plot((j - np.array(energies)) / h, np.array(integrals) / h / sides, "x-")  # energies,

plt.xlabel("$J_{eff} / h$")
plt.ylabel("$f / h$")
plt.grid()
plt.show()

for path in paths:
    energies, integrals = get_data(path)
    resolves = - np.exp(- (np.array(integrals)))
    plt.plot(energies, resolves, "x-")

plt.xlabel("$e \\in [-30*j, 10*j]$")
plt.ylabel("Resolvent")
plt.grid()
plt.show()