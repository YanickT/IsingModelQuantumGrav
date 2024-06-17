import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import random
from hypertiling.core import HyperbolicGraph, GraphKernels
from helper import plot_graph

random.seed(100)


@nb.njit()
def boundary_length_fast(states, update_spins):
    sum_ = 0
    for spin in update_spins:
        if states[spin] == 1:
            sum_ += 1
    return sum_


@nb.njit()
def accept_spinflip_fast(states, update_spin, nbr_relations, h, je, beta, loop_array, q1):
    """

    :param states:
    :param update_spin:
    :param nbr_relations:
    :param h:
    :param je:
    :return:
    """
    # combinations of the neighbors (s = same, d = different)
    # [d, d, s] -> flip ok, no topology change possible
    # [d, s, s] -> topology change possible
    # [s, s, s] -> flip forbidden, topology change inevitable (not possible to to update_list)
    same_nbrs = np.count_nonzero(states[nbr_relations[update_spin]] == states[update_spin])
    if same_nbrs == 2:
        # [d, s, s] case
        # get indices of same neighbors
        indices = nbr_relations[update_spin, np.where(states[nbr_relations[update_spin]] == states[update_spin])[0]]

        # search for loop with the indices
        for i in range(3):
            if indices[0] in loop_array[update_spin, i] and indices[1] in loop_array[update_spin, i]:
                # check if loop is closed or opend
                if loop_array[update_spin, i, 2] == 4294967295:
                    print(loop_array)
                    break
                elif np.all(states[loop_array[update_spin, i]] == states[update_spin]):
                    break
                return False

    # Assuming chi = 1 is ensured by the update process, the energy difference is given by the magnetic alignment and
    # the nearest neighbors only
    state = states[update_spin]
    energy_delta = h if state == 1 else -h
    for nbr in nbr_relations[update_spin]:
        if state * states[nbr] == 1:
            energy_delta += je
        else:
            energy_delta -= je

    return energy_delta < 0 or random.random() < np.exp(- beta * energy_delta)


@nb.njit()
def update_fast(states, nbr_relations, update_spins, n, h, je, beta, llsi, loop_array, q):
    for i in range(n):
        # print(update_spins)
        # index = random.choice(update_spins)
        index = update_spins[random.randint(0, len(update_spins) - 1)]
        if accept_spinflip_fast(states, index, nbr_relations, h, je, beta, loop_array, q):
            states[index] = -1 if states[index] == 1 else 1
            for nbr in nbr_relations[index]:
                # add new spins
                if states[index] != states[nbr] and not (nbr in update_spins) and nbr < llsi:
                    update_spins.append(nbr)

                # check if spin is still in boundary list
                else:
                    if nbr in update_spins:
                        all = True
                        for nbr_ in nbr_relations[nbr]:
                            if not states[nbr_] == states[nbr]:
                                all = False
                                break

                        if all:
                            # if np.all([states[nbr_] == states[nbr] for nbr_ in nbr_relations[nbr]]):
                            array_pos_index = update_spins.index(nbr)
                            del update_spins[array_pos_index]


def construct_path(steps, paths, nbr_relations, node0):
    current_paths = 1
    length = nbr_relations.shape[0]
    active = np.full((paths.shape[0],), 1, dtype=bool)
    for step in range(steps - 1):
        step_m1 = step - 1
        step_p1 = step + 1
        for path_i in range(current_paths):
            path_k = True  # path keep flags
            if not active[path_i]:
                continue

            for nbr in nbr_relations[paths[path_i, step]]:
                if nbr == node0 or (step_m1 >= 0 and nbr == paths[path_i, step_m1]):
                    continue

                if nbr >= length:
                    if path_k:
                        active[path_i] = False
                    else:
                        active[current_paths] = False
                    continue

                if path_k:
                    paths[path_i, step_p1] = nbr
                    path_k = False
                else:
                    paths[current_paths, :step_p1] = paths[path_i, :step_p1]
                    paths[current_paths, step_p1] = nbr
                    current_paths += 1

    active[current_paths:] = False
    return paths[active]


def bidirectional_search(nbr_relations, node0, node1, node2, q):
    # number of steps, i.e. c = np.ceil((q - 3) / 2) with branching factor 2 => 2^c values
    steps = int(np.ceil((q - 1) // 2)) + 1
    steps_p1 = steps + 1
    steps_m1 = steps - 1
    n_path = int(np.power(2, steps_m1))
    loop = np.empty((q,), dtype=np.uint32)

    paths1 = np.empty((n_path, steps), dtype=np.uint32)
    paths1[:, 0] = node1
    paths2 = np.empty((n_path, steps_m1), dtype=np.uint32)
    paths2[:, 0] = node2

    paths1 = construct_path(steps, paths1, nbr_relations, node0)
    paths2 = construct_path(steps_m1, paths2, nbr_relations, node0)

    index_candidate = np.nonzero(np.in1d(paths1[:, -1], paths2[:, -1]))[0]
    if index_candidate.shape[0] == 0:
        return False

    index1 = index_candidate[0]
    index2 = np.nonzero(paths2[:, -1] == [paths1[index1, -1]])[0][0]
    loop[0] = node0
    loop[1:steps_p1] = paths1[index1]
    loop[steps_p1:steps_p1 + steps_m1] = paths2[index2][-2::-1]
    return loop


class HyperbolicIsingModel:

    def __init__(self, p, q, n, gamma=100, beta=10, e=0.2, check_integrity=True):
        # technical attributes
        self.p = p
        self.q = q
        self.n = n
        self.gamma = gamma
        self.beta = float(beta)
        self.epsilon = (2 - q / 3) * np.pi
        self.h = q / 12 - 0.5
        self.j = np.sqrt((q - 6) * (q - 2)) / 12

        # graph attributes
        # create additional layers for loop detection
        self.dn = int(np.ceil(q / 2) - 1)
        self.graph = HyperbolicGraph(p, q, n + self.dn, kernel=GraphKernels.GenerativeReflectionGraphStatic)
        self.llsi = self.graph._sector_lengths_cumulated[- self.dn - 1]
        self.dn_lsi = self.graph._sector_lengths_cumulated[- self.dn]
        if check_integrity:
            self.graph.check_integrity()
            print(f"Expect all neighbors till Node {self.llsi}")

        nbr_list = self.graph.get_nbrs_list()
        self.nbrs_relations = np.array([e for e in nbr_list if len(e) == 3],
                                       dtype=np.uint32)  # exclude last layer for boundary

        # create loops
        self.nbr_array = np.array([e if len(e) == 3 else e + [4294967295] * (3 - len(e)) for e in nbr_list])
        self.loop_array = np.empty((self.llsi, 3, self.q - 1), dtype=np.uint32)  # self.p == 3!
        for node_index in range(self.llsi):
            for nbr_index, nbr in enumerate(self.nbrs_relations[node_index]):
                complement_nbr = self.nbr_array[node_index, (nbr_index + 2) % 3]
                loop = bidirectional_search(self.nbr_array, node_index, nbr, complement_nbr, self.q)

                # FIXME: use permutation instead of multiple calculation (fast & dirty)
                self.loop_array[node_index, nbr_index] = loop[1:]

        self.states = None
        self.update_spins = None

        self.set_e(e)
        self.rest()

    def __len__(self):
        return self.llsi

    def set_e(self, e):
        self.e = e
        self.je = (self.j - self.e)

    def set_beta(self, beta):
        self.beta = beta

    def get_states(self):
        return self.states[:self.dn_lsi]

    def plot(self, title="", ignore_last_n_layers=0, save=None):
        # for plotting
        lengths = self.graph._sector_lengths_cumulated[- (ignore_last_n_layers + 1)]
        plot_graph(self.graph.get_nbrs_list()[:lengths], self.graph.center_coords[:lengths], self.p,
                   self.states[:lengths])
        plt.title(title)
        if not (save is None):
            plt.savefig(save)
        plt.show()

    def energy(self):
        energy = - (self.gamma + 1 - self.epsilon) - self.h * np.sum(self.states + 1) / 2
        # print("MIT ODER OHNE BOUNDARY? MOMENTAN OHNE")
        for node_index in range(self.dn_lsi):
            for nbr in self.nbr_array[node_index]:
                if nbr >= self.dn_lsi:
                    continue
                if self.states[nbr] != self.states[node_index]:
                    energy += self.je
        return energy

    def boundary_length(self):
        return boundary_length_fast(self.states, self.update_spins)

    def spin_ups(self):
        return np.count_nonzero(self.states == 1)

    def rest(self):
        # setup boundary conditions
        self.states = np.full((len(self.graph),), 1, dtype=np.int8)
        self.states[self.llsi:] = -1

        # create array for updatable spins
        self.update_spins = nb.typed.List(list(range(self.graph._sector_lengths_cumulated[- self.dn - 2], self.llsi)))
        # self.update_spins = list(range(self.graph._sector_lengths_cumulated[- self.dn - 2], self.llsi))

    def update(self, n):
        update_fast(self.states, self.nbrs_relations, self.update_spins, n, self.h, self.je, self.beta, self.llsi,
                    self.loop_array, self.q - 1)

    def reaches_boundary(self):
        # print(self.graph._sector_lengths_cumulated[self.llsi])
        return (np.max(self.states[self.graph._sector_lengths_cumulated[- self.dn - 2]: self.llsi]) == 1)


if __name__ == "__main__":
    import time

    model = HyperbolicIsingModel(3, 7, 7, beta=5)
    print(f"Total length: {len(model)}")
    model.plot("Initialization", ignore_last_n_layers=3)

    # some of the parameters
    #model.update(1)
    #e = model.j
    #model.rest()
    #model.set_e(e)

    t1 = time.time()
    model.update(1_000_000)
    print(f"Took: {time.time() - t1}s")
    model.plot(f"E = {e: .4f}", ignore_last_n_layers=3)
