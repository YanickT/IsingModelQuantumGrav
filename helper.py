import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt


def plot_graph(adjacent_matrix, center_coords, p, states, colors=None):
    fig_ax = plt.subplots()
    fig_ax[1].set_xlim(-1, 1)
    fig_ax[1].set_ylim(-1, 1)
    fig_ax[1].set_box_aspect(1)
    if colors is None:
        colors = ["#000000", "#FF0000AA", "#0000FFAA"]
    graph = nx.Graph()
    for y in range(len(adjacent_matrix)):
        if y >= center_coords.shape[0]:
            sector = (y - 1) // (center_coords.shape[0] - 1)
            index = (y - 1) % (center_coords.shape[0] - 1)
            index += 1
            rot = center_coords[index] * np.exp(1j * sector * np.pi * 2 / p)
            x_ = np.real(rot)
            y_ = np.imag(rot)
        else:
            x_ = np.real(center_coords[y])
            y_ = np.imag(center_coords[y])

        graph.add_node(y, pos=(x_, y_), node_color=colors[states[y]])

    for y, row in enumerate(adjacent_matrix):
        for index in row:
            if index >= len(adjacent_matrix):
                continue
            graph.add_edge(y, index)
    nx.draw_networkx(graph, pos=nx.get_node_attributes(graph, 'pos'),
                     node_color=list(nx.get_node_attributes(graph, 'node_color').values()))
