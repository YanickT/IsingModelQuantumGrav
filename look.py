import os
import pickle
from hyperbolic_model import HyperbolicIsingModel

path = "C:/Users/yanic/Documents/GitHub/IsingModelQuantumGrav/IsingModelQuantumGrav/data"
# path = "C:/Users/yanic/Documents/GitHub/IsingModelQuantumGrav/IsingModelQuantumGrav"
model = HyperbolicIsingModel(3, 7, 7, beta=1)


for file in [name for name in os.listdir(path) if name[-4:] == ".pkl"]:
    energy_part = float(file.split("_")[3].split("=")[1])
    if not (energy_part == -0.37267799624996495):
        continue

    with open(f"{path}/{file}", "rb") as doc:
        states = pickle.load(doc)

    model.states = states[1]
    b = file.split("_")[2].split("=")[1]
    model.plot(ignore_last_n_layers=model.dn - 1, save=file[:-4]+ ".png", title=f"$\\beta = {b}$")