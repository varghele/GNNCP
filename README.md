# GNNCP
Two quick notebooks that employ GNNs (graph neural networks) to calculate conformation probabilities from the frames of a molecular trajectory. 

## Input:
Inputs consist of the trajectory `shooting_points.h5`, as well as the targets `shooting_results.npz`. The trajectory here is from a LiCl system in a PIP3 water environment. There are a total of 1112 atoms, and the trajectory is 1000 frames long. In `shooting_results.npz`, the number of achieved conformations, when starting a trajectory from this point on, are archieved. For each frame there are results of 500 offshoots. \
\
This has to be fed into the notebook `1_MAKE_GRAPH_FROM_H5.ipynb` first, to convert every frame of the trajectory into a fully connected graph. This graph is structured the following way:\
\
**EDGES:**\
Edges represent the connections between the atoms. The graph is fully connected, so each edge represents a connection between two atoms with the following attributes: `[d, 1, 0]`, or `[d, 0, 1]`, where `d` represents the distance between the two connected nodes. The rest is a one-hot encoding, whether the edge represents a bond, or not.\
\
**NODES**\
Each node represents an atom with the following attributes: `[e, s, x, y, z]`, where e(epsilon) and s(sigma) are molecular descriptors from the OpenMM Forcefield (<https://github.com/openmm/openmmforcefields/blob/ddea4f61d508f0cc06ad09dd29d0943721890038/amber/ffxml/tip3p_standard.xml#L261>), and x,y,z the coordinates of the atom.\
\
**GLOBAL**\
The global attributes of the graph represent `[na, nb]`, the total number of achieved conformation A or B when running the offshoot from the frame. This is then used in the binomial loss of the prediction.

## Output of `1_MAKE_GRAPH_FROM_H5.ipynb`:
Output of `1_MAKE_GRAPH_FROM_H5.ipynb` is a graph for each frame of the trajectory. Since those graphs are fully connected, they become quite large, and one frame takes 30MB of space. Once the graphs are prepared, you can run the notebook `2_GNN.ipynb` to train a GNN to predict the shooting result of the frame.

## GNN
In its current form, the GNN in the notebook features a node and edge encoding, scatter_add as aggregation, and one layer MLPs as update functions, with one round of message-passing. THe GNN predicts `qb`, the logarithmic possibility of one conformation, and is trained to minimize the binomial loss between `qb` and `na` and `nb`.\
Logging is currently implemented with `tensorboard`.



