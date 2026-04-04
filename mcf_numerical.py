import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def plot_surface_triangle(nodes_3d, triangles, title="Surface mesh"):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    faces = nodes_3d[triangles]

    mesh = Poly3DCollection(faces, alpha=0.6)
    mesh.set_facecolor('steelblue')
    mesh.set_edgecolor('white')
    mesh.set_linewidth(0.2)
    ax.add_collection3d(mesh)

    # scaling axes
    mins = nodes_3d.min(axis=0)
    maxs = nodes_3d.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return fig

def plot_surface(triangles, title='Surface mesh'):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    faces = triangles
    mesh = Poly3DCollection(faces, alpha=0.6)
    mesh.set_facecolor('steelblue')
    mesh.set_edgecolor('white')
    mesh.set_linewidth(0.2)
    ax.add_collection3d(mesh)

    # scaling axes
    mins = np.min(triangles, axis=(0, 1))
    maxs = np.max(triangles, axis=(0, 1))

    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])


    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return fig

def compute_grad_T(i, nu_T, nodes, edge_vectors):
    if i == 1:
        numerator = np.cross((nodes[2] - nodes[1]), nu_T)
        denominator = np.cross(edge_vectors[0], edge_vectors[1])
        denom = np.linalg.norm(denominator)
        return numerator/denom
    elif i == 2:
        numerator = np.cross((nodes[0] - nodes[2]), nu_T)
        denominator = np.cross(edge_vectors[0], edge_vectors[1])
        denom = np.linalg.norm(denominator)
        return numerator/denom

    else:
        numerator = np.cross((nodes[1] - nodes[0]), nu_T)
        denominator = np.cross(edge_vectors[0], edge_vectors[1])
        denom = np.linalg.norm(denominator)
        return numerator/denom


def compute_nu_T(e1, e2):
    num = np.cross(e1, e2)
    denom = np.linalg.norm(num)
    return num/denom

def compute_area_T(e1, e2):
    inner = np.cross(e1, e2)
    result = 0.5*np.linalg.norm(inner)
    return result


# def compute_Sij(N, nodes, tri_indices):
#     Sij = np.zeros((N,N,3))
#     Sij_T = [] # storing local stiffness matrices
#     
#     for t, tri in enumerate(tri_indices):
#         T = nodes[tri]
#         e1 = T[1] - T[0]
#         e2 = T[2] - T[0]
#         edge_vectors = [e1,e2]
#         nu_T = compute_nu_T(e1, e2)
#         area_T = compute_area_T(e1, e2)
#     
#         current_Sij_T = np.zeros((3,3))
#     
#         for i in range(T.shape[0]):
#             for j in range(T.shape[1]):
#                 # compute Sij
#     
#                 doti = compute_grad_T(i, nu_T, T, edge_vectors)
#                 dotj = compute_grad_T(j, nu_T, T, edge_vectors)
#                 dot_result = np.dot(doti, dotj)
#                 Sij_T_result = np.dot(dot_result, area_T)
#                 current_Sij_T[i][j] = Sij_T_result
#     
#         Sij_T.append(current_Sij_T)
# 
# 
#     # building global Sij
#     N_nodes = N*N
#     Sij_global = np.zeros((N_nodes, N_nodes))
#     
#     for t, tri in enumerate(tri_indices):
#         for i in range(3):
#             for j in range(3):
#                 Sij_global[tri[i], tri[j]] += Sij_T[t][i][j]
# 
#     return Sij_global
def compute_Sij(N, nodes, tri_indices):
     T_all = nodes[tri_indices]             # (M, 3, 3)
     e1 = T_all[:, 1] - T_all[:, 0]        # (M, 3)
     e2 = T_all[:, 2] - T_all[:, 0]        # (M, 3)

     cross = np.cross(e1, e2)               # (M, 3)
     denom = np.linalg.norm(cross, axis=1)  # (M,)
     nu = cross / denom[:, None]            # (M, 3)
     area = denom / 2                       # (M,)

     edge0 = T_all[:, 1] - T_all[:, 0]
     edge1 = T_all[:, 2] - T_all[:, 1]
     edge2 = T_all[:, 0] - T_all[:, 2]

     grad0 = np.cross(edge0, nu) / denom[:, None]  # (M, 3)
     grad1 = np.cross(edge1, nu) / denom[:, None]
     grad2 = np.cross(edge2, nu) / denom[:, None]

     grads = np.stack([grad0, grad1, grad2], axis=1)  # (M, 3, 3)
     S_local = np.einsum('mik,mjk->mij', grads, grads) * area[:, None, None]  # (M, 3, 3)

     N_nodes = N * N
     Sij_global = np.zeros((N_nodes, N_nodes))
     for li in range(3):
         for lj in range(3):
             np.add.at(Sij_global, (tri_indices[:, li], tri_indices[:, lj]), S_local[:, li, lj])

     return Sij_global


def compute_Mij(N, nodes, tri_indices):
      T_all = nodes[tri_indices]
      e1 = T_all[:, 1] - T_all[:, 0]
      e2 = T_all[:, 2] - T_all[:, 0]
      area = np.linalg.norm(np.cross(e1, e2), axis=1) / 2  # (M,)

      M_local = (area[:, None, None] / 12) * np.array([[2,1,1],[1,2,1],[1,1,2]])  # (M, 3, 3)

      N_nodes = N * N
      Mij_global = np.zeros((N_nodes, N_nodes))
      for li in range(3):
          for lj in range(3):
              np.add.at(Mij_global, (tri_indices[:, li], tri_indices[:, lj]), M_local[:, li, lj])

      return Mij_global


# def compute_Mij(N, nodes, tri_indices):
#     N_nodes = N*N
#     Mij = np.zeros((N_nodes,N_nodes))
#     
#     for t, tri in enumerate(tri_indices):
#         T = nodes[tri]
#         e1 = T[1] - T[0]
#         e2 = T[2] - T[0]
#         area_T = compute_area_T(e1, e2)
#     
#         Mij_T = (area_T/12)*np.array([[2, 1, 1],
#                                       [1, 2, 1],
#                                       [1, 1, 2]])
#         for i in range(3):
#             for j in range(3):
#                 Mij[tri[i], tri[j]] += Mij_T[i][j]
# 
# 
#     return Mij
