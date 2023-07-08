import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import random
from hypertiling.core import HyperbolicGraph, GraphKernels
from helper import plot_graph


@nb.njit()
def boundary_length_fast(states, update_spins):
    sum_ = 0
    for spin in update_spins:
        if states[spin] == 1:
            sum_ += 1
    return sum_


@nb.njit()
def accept_spinflip_fast(states, update_spin, nbr_relations, h, je, beta):
    """

    :param states:
    :param update_spin:
    :param nbr_relations:
    :param h:
    :param je:
    :return:
    """
    # critical combination # FIXME: this is not entirely true sometimes it should flip
    same = 0
    for nbr in nbr_relations[update_spin]:
        if states[update_spin] * states[nbr] == 1:
            same += 1
            if same == 2:
                return False

    # Assuming chi = 1 is ensured by the update process, the energy difference is given by the magnetic alignment and
    # the nearest neighbors only
    state = states[update_spin]
    energy_delta = h if state == 1 else -h
    for nbr in nbr_relations[update_spin]:
        if state * states[nbr] == 1:
            energy_delta += je

    return energy_delta < 0 or random.random() < np.exp(- beta * energy_delta)


def update_fast(states, nbr_relations, update_spins, n, h, je, beta, llsi):
    for i in range(n):
        index = random.choice(update_spins)
        if accept_spinflip_fast(states, index, nbr_relations, h, je, beta):
            states[index] = -1 if states[index] == 1 else 1
            for nbr in nbr_relations[index]:
                # add new spins
                if states[index] != states[nbr] and not (nbr in update_spins) and nbr < llsi and nbr != 0:
                    update_spins.append(nbr)

                # check if spin is still in boundary list
                else:
                    if nbr in update_spins:
                        array_pos_index = update_spins.index(nbr)
                        if np.all([states[nbr_] == states[nbr] for nbr_ in nbr_relations[nbr]]):
                            del update_spins[array_pos_index]


def construct_path(steps, paths, nbr_relations, node0):
    current_paths = 1
    for step in range(steps):
        step_m1 = step - 1
        step_p1 = step + 1
        for path_i in range(current_paths):
            path1_k = True  # path keep flags
            for nbr in nbr_relations[paths[path_i, step]]:
                if nbr == node0 or (step_m1 >= 0 and nbr == paths[path_i, step_m1]):
                    continue
                if path1_k:
                    paths[path_i, step_p1] = nbr
                    path1_k = False
                else:
                    paths[current_paths, :step_p1] = paths[path_i, :step_p1]
                    paths[current_paths, step_p1] = nbr
                    current_paths += 1
    return paths


def bidirectional_search(nbr_relations, node0, node1, node2, q):
    # number of steps, i.e. c = np.ceil((q - 3) / 2) with branching factor 2 => 2^c values
    steps = int(np.ceil((q - 1) // 2)) + 1
    n_path = int(np.power(2, steps + 1))

    paths1 = np.empty((n_path, steps), dtype=np.uint32)
    paths1[:, 0] = node1
    paths2 = np.empty((n_path, steps), dtype=np.uint32)
    paths2[:, 0] = node2

    paths1 = construct_path(steps, paths1, nbr_relations, node0)
    paths2 = construct_path(steps, paths2, nbr_relations, node0)

    # check in last column which matches
    # add node 0
    # return circle

class HyperbolicIsingModel:

    def __init__(self, p, q, n, gamma=100, beta=5, e=0, check_integrity=True):
        # technical attributes
        self.p = p
        self.q = q
        self.n = n
        self.gamma = gamma
        self.beta = float(beta)
        self.epsilon = (2 - q / 3) * np.pi
        self.h = q / 12 - 0.5
        self.j = np.sqrt((q - 6) * (q - 2)) / 12
        self.e = e
        self.je = (self.j - self.e)

        # graph attributes
        self.graph = HyperbolicGraph(p, q, n, kernel=GraphKernels.GenerativeReflectionGraphStatic)
        self.llsi = self.graph._sector_lengths_cumulated[-2]
        if check_integrity:
            self.graph.check_integrity()
            print(f"Expect all neighbors till Node {self.llsi}")
        self.nbrs_relations = np.array([e for e in self.graph.get_nbrs_list() if len(e) == 3],
                                       dtype=np.uint32)  # exclude last layer for boundary

        # create loops
        self.loop_array = np.empty((self.nbrs_relations.shape[0], 3, self.q - 1), dtype=np.uint32)  # self.p == 3!
        """
        bidirectional search for loops:
        1. select two neighbors (3 already specified)
        2. start bidirectional search
        3. terminate if in both lists the same nodes appear (loop closed)
        4. insert loop into every included node set (by rotation)
        """
        for node_index in range(self.llsi):
            for nbr_index, nbr in enumerate(self.nbrs_relations[node_index]):
                complement_nbr = self.nbrs_relations[node_index, (nbr_index + 2) % 3]
                loop = bidirectional_search(self.nbrs_relations, node_index, nbr, complement_nbr, self.q)
                print(nbr, loop)

        # setup boundary conditions
        self.states = np.full((len(self.graph),), 1, dtype=np.int8)
        self.states[self.llsi:] = -1

        # create array for updatable spins
        self.update_spins = list(range(self.graph._sector_lengths_cumulated[-3], self.llsi))

    def plot(self, title=""):
        # for plotting
        plot_graph(self.graph.get_nbrs_list(), self.graph.center_coords, self.p, self.states)
        plt.title(title)
        plt.show()

    def boundary_length(self):
        return boundary_length_fast(self.states, self.update_spins)

    def spin_ups(self):
        return np.count_nonzero(self.states == 1)

    def update(self, n):
        update_fast(self.states, self.nbrs_relations, self.update_spins, n, self.h, self.je, self.beta, self.llsi)


if __name__ == "__main__":
    import time

    model = HyperbolicIsingModel(3, 7, 5)
    print(model.loop_array)
    model.plot("Initialization")
    t1 = time.time()
    model.update(1000)
    print(f"Took: {time.time() - t1}s")
    model.plot("After run")

    set(a)