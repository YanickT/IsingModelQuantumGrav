import os
import pickle
from hyperbolic_model import HyperbolicIsingModel


path = "C:/Users/yanic/Documents/GitHub/IsingModelQuantumGrav/IsingModelQuantumGrav/data"
model = HyperbolicIsingModel(3, 7, 7, beta=1)


for file in [name for name in os.listdir(path) if name[-4:] == ".pkl"]:
    with open(f"{path}/{file}", "rb") as doc:
        states = pickle.load(doc)

        model.states = states[-1]
        model.plot(ignore_last_n_layers=3)