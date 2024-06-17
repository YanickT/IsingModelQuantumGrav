from hyperbolic_model import HyperbolicIsingModel
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pickle


# some of the parameters
thermalization_time = 1_000_000
mc_updates = 10_000 # 100_000_000
beta_start = 0
beta_stop = 10
p, q, n = 3, 11, 14

model = HyperbolicIsingModel(p, q, n, beta=beta_start)
# model.plot()
print(f"Using model of length {len(model)}")

j = model.j

es = np.linspace(-5 * model.j, 5 * model.j, 10, True)# [51:41:-1]  # DEBUG: testing for boundary [4:]
# print(es)
# es = np.linspace(-8.57649672e-01, 1.97568366e+00, 100, True)
# es = [0.15]  # -0.8, -0.05, 0.05, 0.1, 0.15
# es = np.linspace(-30*j, 10*j, 100, True)  # dividable by 5
# es = np.linspace(-1.5, 1.5, 100, True)  # dividable by 5
# print(f"Energies: {es}")

for e in es:

    delta_beta = (beta_stop - beta_start) / mc_updates
    beta = beta_start

    # prepare model
    model.set_e(e)
    model.rest()

    print(f"Start with energy e = {e} with delta_beta = {delta_beta}:")

    # thermalize model
    t1 = time.time()
    model.update(thermalization_time)
    t2 = time.time()
    print(f"\tThermalization took {t2 - t1} s")

    energies = [model.energy()]
    boundary_lengths = [model.boundary_length()]
    betas = [beta]
    for i in range(mc_updates):
        model.update(len(model))
        betas.append(beta)
        energies.append(model.energy())
        boundary_lengths.append(model.boundary_length())

        beta += delta_beta
        model.set_beta(beta)
    print(f"\tMeasurement took {time.time() - t2} s")

    # model.plot()
    # for finite size measurement
    if model.reaches_boundary():
        print(f"BOUNDARY REACHED FOR {e}")
        # model.plot()
        continue
    else:
        print(f"BOUNDARY NOT!!!! REACHED FOR {e}")
        # model.plot()
        continue

    #model.plot(ignore_last_n_layers=4, title=e)
    #exit("FORCE EXIT HERE <<<<<")

    # path = f"C:/Users/yanic/Documents/GitHub/IsingModelQuantumGrav/IsingModelQuantumGrav/quencing_data/quencing_data_{beta_start}_{beta_stop}_{p}_{q}_{n}"
    path = f"C:/Users/yanic/Documents/GitHub/IsingModelQuantumGrav/IsingModelQuantumGrav/quencing_data/quencing_data_{beta_start}_{beta_stop}_{p}_{q}_{n}_close"
    if not os.path.isdir(path):
        os.mkdir(path)
    f_name = f"{path}/e={e}"
    with open(f"{f_name}.csv", "w") as doc:
        doc.writelines([f"MC Hyperbolic Ising model\n",
                        f"p:;{model.p}\n",
                        f"q:;{model.q}\n",
                        f"n:;{model.n}\n",
                        f"Active nodes:;{len(model)}\n",
                        f"Total nodes:;{model.dn_lsi}\n",
                        f"Beta (start):;{beta_start}\n",
                        f"Beta (stop):;{beta_stop}\n",
                        f"E:;{e}\n",
                        f"J:;{model.j}\n",
                        f"Gamma:;{model.gamma}\n",
                        f"Epsilon:;{model.epsilon}\n",
                        f"h:;{model.h}\n",
                        f"Thermaliatzion time [its]:;{thermalization_time}\n",
                        f"MC update time [its]:;{mc_updates}\n\n",
                        f"Time for Thermalization [s]:;{t2 - t1}\n\n",
                        f"Beta;Energy;Boundary_length\n"] + [f"{beta};{e};{b}\n" for beta, e, b in zip(betas, energies, boundary_lengths)])
