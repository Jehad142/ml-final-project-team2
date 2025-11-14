import networkx as nx
import matplotlib.pyplot as plt
from jarvis.core.atoms import Atoms
import numpy as np

def atoms_to_graph(atoms_dict, cutoff=3.5):
    """
    Convert a Jarvis Atoms dictionary to a NetworkX graph.

    Parameters
    ----------
    atoms_dict : dict
        Dictionary from Jarvis containing 'elements' and 'coordinates'.
    cutoff : float
        Distance cutoff (in Angstrom) to define a bond.

    Returns
    -------
    G : networkx.Graph
        Graph with nodes as atoms and edges as bonds.
    """
    # Create Atoms object
    atoms = Atoms(
        symbols=atoms_dict['elements'],
        positions=atoms_dict['coordinates']
    )

    G = nx.Graph()
    
    # Add nodes
    for i, symbol in enumerate(atoms.elements):
        G.add_node(i, atomic_number=atoms.atomic_numbers[i], symbol=symbol)
    
    # Add edges based on cutoff distance
    positions = atoms.get_positions()
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= cutoff:
                G.add_edge(i, j, distance=dist)
    
    return G

def visualize_graph(G, positions=None):
    """
    Visualize a NetworkX graph with node labels as element symbols.
    If positions are not provided, use spring_layout.
    """
    if positions is None:
        positions = nx.spring_layout(G, seed=42)
    
    atomic_numbers = [G.nodes[n]['atomic_number'] for n in G.nodes()]
    labels = {n: G.nodes[n]['symbol'] for n in G.nodes()}
    
    plt.figure(figsize=(6,6))
    nx.draw(
        G,
        pos=positions,
        labels=labels,
        with_labels=True,
        node_color=atomic_numbers,
        cmap=plt.cm.tab20,
        node_size=500
    )
    plt.show()
