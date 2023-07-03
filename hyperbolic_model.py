import matplotlib.pyplot as plt
import numpy as np
from hypertiling.core import HyperbolicGraph, GraphKernels
from helper import plot_graph


graph = HyperbolicGraph(3, 7, 4, kernel=GraphKernels.GenerativeReflectionGraphStatic)
graph.check_integrity()
llsi = graph._sector_lengths_cumulated[-2]  # last_layer_start_index
print(f"Expect all neighbors till Node {llsi}")
nbrs_relations = graph.get_nbrs_list()

# setup boundary conditions
states = np.full((len(nbrs_relations),), -1, dtype=np.int32)
states[llsi:] = 1

# create list for updatable spins
update_spins = np.arange(graph._sector_lengths_cumulated[-3], llsi)
print(f"Updatable spins: {update_spins}")

# for plotting
plot_graph(nbrs_relations, graph.center_coords, graph.p, states)
plt.title("Initialization")
plt.show()


def update(n, graph, update_spins, states):
    for i in range(n):
        index = np.random.choice(update_spins)
        print(index)


update(10, graph, update_spins, states)