import numpy as np
from scipy.sparse import lil_matrix


def pi_x (i,V,theta,G,B):


    V = V.tocsr()
    theta = theta.tocsr()
    G = G.tocsr()
    B = B.tocsr()

    # Calculate delta_theta matrix
    delta_theta = theta[i, 0] - theta.toarray().flatten()

    # Compute P_i  using dot product
    V_i = V[i, 0]
    V_all = V.toarray()
    G_row = G.getrow(i).toarray().flatten()
    B_row = B.getrow(i).toarray().flatten()

    P_i = V_i * np.dot(G_row * np.cos(delta_theta) + B_row * np.sin(delta_theta),V_all.flatten())


    return P_i



V = lil_matrix([[1.0], [1.02], [0.98]])  # Voltage magnitudes
theta = lil_matrix([[0.0], [-0.1], [0.2]])  # Voltage angles in radians
G = lil_matrix((3, 3))
G[0, 1] = 0.1;
G[0, 2] = 0.05
G[1, 0] = 0.1;
G[1, 2] = 0.02
G[2, 0] = 0.05;
G[2, 1] = 0.02

B = lil_matrix((3, 3))
B[0, 1] = -0.2;
B[0, 2] = -0.1
B[1, 0] = -0.2;
B[1, 2] = -0.05
B[2, 0] = -0.1;
B[2, 1] = -0.05

i = 1  # Bus index (0-based)
P_i = pi_x(i, V, theta, G, B)
print(P_i)