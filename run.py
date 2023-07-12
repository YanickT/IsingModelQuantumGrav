from hyperbolic_model import HyperbolicIsingModel
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle

"""
Boundary length + Energy aufzeichnen
es = [-j, 10*j, 11]
beta = [1, 10, 20, 30, 40, 50]
100_000_000 für thermalization

MC samples: 1000
Updates dazwischen: 100_000
"""


beta = 1
model = HyperbolicIsingModel(3, 7, 7, beta=1)
j = model.j
print(f"Total length: {len(model)}")
# model.plot("Initialization", ignore_last_n_layers=3)
model.update(1)  # jit compile stuff


betas = [1, 10, 20, 30, 40, 50]
es = [i * j - j for i in range(11)]


# some of the parameters
thermalization_time = 100_000_000
mc_samples = 1000
mc_updates = 100_000

j = model.j
es = np.linspace(-j, 10 * j, 10)

for beta in betas:
    model.set_beta(beta)
    for e in es:
        model.rest()
        model.set_e(e)
        energies = [model.energy()]
        boundary_lengths = [model.boundary_length()]
        states = [model.get_states()]

        # thermalization
        print(f"Beta = {beta}; E = {e}: Start thermalization", end="")
        t1 = time.time()
        model.update(thermalization_time)
        t2 = time.time()
        states.append(model.get_states())

        # MC sampling
        print(f"\rBeta = {beta}; E = {e}; Thermalization took {t2 - t1}s: Start sampling", end="")
        t3 = time.time()
        for i in range(mc_samples):
            model.update(mc_updates)
            energies.append(model.energy())
            boundary_lengths.append(model.boundary_length())
            states.append(model.get_states())
        t4 = time.time()
        print(f"\rBeta = {beta}; E = {e}; Thermalization took {t2 - t1}s: Sampling took {t4 - t3}s")

        # save data
        path = "C:/Users/yanic/Documents/GitHub/IsingModelQuantumGrav/IsingModelQuantumGrav/data"
        f_name = f"{path}/mc_sampling_b={beta}_e={e}"
        with open(f"{f_name}.csv", "w") as doc:
            doc.writelines([f"MC Hyperbolic Ising model\n",
                            f"p:;{model.p}\n",
                            f"q:;{model.q}\n",
                            f"n:;{model.n}\n",
                            f"Active nodes:;{len(model)}\n",
                            f"Total nodes:;{model.dn_lsi}\n",
                            f"Beta:;{beta}\n",
                            f"E:;{e}\n",
                            f"J:;{model.j}\n",
                            f"Gamma:;{model.gamma}\n",
                            f"Epsilon:;{model.epsilon}\n",
                            f"h:;{model.h}\n",
                            f"Thermaliatzion time [its]:;{thermalization_time}\n",
                            f"MC update time [its]:;{mc_updates}\n",
                            f"MC samples:;{mc_samples}\n\n",
                            f"Time for Thermalization [s]:;{t2 - t1}\n\n",
                            f"Time for Sampling [s]:;{t4 - t3}\n\n",
                            f"Energy; Boundary_length\n"] + [f"{e};{b}\n" for e, b in zip(energies, boundary_lengths)])

        # states
        with open(f"{f_name}_states.pkl", "wb") as doc:
            pickle.dump(states, doc)
