# IsingModelQuantumGrav
 
Repository to [Paper](https://arxiv.org/abs/2407.06266).

## Abstract
Inspired by the program of discrete holography, we show that Jackiw-Teitelboim (JT) gravity on a hyperbolic tiling of Euclidean AdS2 gives rise to an Ising model on the dual lattice, subject to a topological constraint. The Ising model involves an asymptotic boundary condition with spins pointing opposite to the magnetic field. The topological constraint enforces a single domain wall between the spins of opposite direction, with the topology of a circle. The resolvent of JT gravity is related to the free energy of this Ising model, and the classical limit of JT gravity corresponds to the Ising low-temperature limit. We study this Ising model through a Monte Carlo approach and a mean-field approximation. For finite truncations of the infinite hyperbolic lattice, the map between both theories is only valid in a regime in which the domain wall has a finite size. For the extremal cases of large positive or negative coupling, the domain wall either shrinks to zero or touches the boundary of the lattice. This behavior is confirmed by the mean-field analysis. We expect that our results may be used as a starting point for establishing a holographic matrix model duality for discretized gravity. 

## Usage

Use ```quencing.py``` to collect the data. They will be stored in multiple .csv files.
Use ```visualize.py``` to plot the free energy.