import os
import numpy as np
import matplotlib.pyplot as plt
import math

path = "quencing_data/quencing_data_0_10_3_7_14/"
paths = [path]


def get_data(path):
    free_energies = []
    free_energies2 = []
    files = os.listdir(path)
    for file in files:
        with open(f"{path}{file}", "r") as doc:
            lines = doc.readlines()
        nodes = int(lines[5][:-1].split(";")[1])
        energy = float(lines[8][:-1].split(";")[1])
        j = float(lines[9].split(";")[1])
        h = float(lines[12].split(";")[1])
        sides = int(lines[4].split(";")[1])
        beta_stop = int(lines[7].split(";")[1])
        f0 = -nodes * math.log(2)
        betas, es, _ = tuple(zip(*[line.split(";") for line in lines[19:]]))
        betas = [float(beta) for beta in betas]
        es = [float(e) for e in es]
        integral = np.trapz(es, x=betas) + f0
        integral /= beta_stop
        free_energies.append((energy, integral))
        free_energies2.append((energy, es[-1]))

    free_energies.sort(key=lambda x: x[0])
    free_energies2.sort(key=lambda x: x[0])
    return tuple(zip(*free_energies)), j, h, sides, tuple(zip(*free_energies2))


for path in paths:
    (energies, integrals), j, h, sides, (energies2, integrals2) = get_data(path)
    plt.plot((j - np.array(energies)) / h, np.array(integrals) / h / sides, "x-")  # energies,
    plt.plot((j - np.array(energies2)) / h, np.array(integrals2) / h / sides, "x-")  # energies,

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