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
print(f"Using model of length {len(model)}")
j = model.j
es = np.linspace(-5 * model.j, 5 * model.j, 100, True)

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
        print(f"Boundary reached for {e}")
    else:
        print(f"Boundary NOT reached for {e}")

    path = f"data/quencing_data_{beta_start}_{beta_stop}_{p}_{q}_{n}_close"
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
