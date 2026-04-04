
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import io
from mcf_numerical import plot_surface_triangle, plot_surface
from mcf_numerical import compute_Sij, compute_Mij


# def sphere(u,v):
#     x = np.sin(v)*np.cos(u)
#     y = np.sin(v)*np.sin(u)
#     z = np.cos(v)
#     return np.array([x, y, z])
def sphere(u,v):
    x = u
    y = v
    z = np.sin(np.sqrt(u**2 + v**2))
    return np.array([x,y,z])


N = 20
u_range = np.linspace(0, 2*np.pi, N, endpoint=True)
v_range = np.linspace(0, 2*np.pi, N) # poles
U, V = np.meshgrid(u_range, v_range)

f_mesh = np.zeros((N, N, 3))
for i in range(N):
    for j in range(N):
        u_val = u_range[i]
        v_val = v_range[j]
        f_mesh[i][j] = sphere(u_val, v_val).astype(float).flatten()

nodes = f_mesh.reshape(N*N, 3) # global node list

tri_indices = []
for i in range(N): # rows
    i_next = (i + 1) % N
    for j in range(N-1): # columns (will wrap around)
        a = i*N + j
        b = i*N + (j+1)
        c = i_next*N + (j+1)
        d = i_next*N + j

        tri_indices.append([a,b,d])
        tri_indices.append([b,c,d])

tri_indices = np.array(tri_indices)

Sij_global = compute_Sij(N, nodes, tri_indices)
Mij_global = compute_Mij(N, nodes, tri_indices)

nodes_current = nodes
Sij_current = Sij_global
Mij_current = Mij_global
iteration_counter = 0
frame_freq = 1

plotly_frames = []

while True:
    # add to plotly frames
    if iteration_counter % frame_freq == 0:
       frame = go.Frame(
                data = go.Mesh3d(
                    x=nodes_current[:, 0],
                    y=nodes_current[:,1],
                    z=nodes_current[:,2],
                    i = tri_indices[:,0],
                    j = tri_indices[:,1],
                    k = tri_indices[:,2],
                    color='steelblue',
                    opacity=0.7
                ),
                name=str(iteration_counter)
        )
       plotly_frames.append(frame)

    edges = nodes_current[tri_indices[:, 1]] - nodes_current[tri_indices[:,0]]
    h_current = np.min(np.linalg.norm(edges, axis=1))
    tau = 0.01*h_current**2

    b = Mij_current @ nodes_current
    new_nodes = np.linalg.solve(Mij_current + tau*Sij_current, b)

    displacement = np.max(np.linalg.norm(new_nodes - nodes_current, axis=1))
    print("iteration: ", iteration_counter, " displacement: ", displacement)

    nodes_current = new_nodes

    # update Sij and Mij
    Sij_current = compute_Sij(N, new_nodes, tri_indices)
    Mij_current = compute_Mij(N, new_nodes, tri_indices)
    iteration_counter += 1

    if displacement < 1e-1:
        break

dur = 100
# plotly animation:
fig = go.Figure(
        data = go.Mesh3d(
            x=nodes[:,0],
            y=nodes[:,1],
            z=nodes[:,2],
            i=tri_indices[:,0],
            j=tri_indices[:,1],
            k=tri_indices[:,2],
            colorscale='Viridis',
            cmin=-1,cmax=1
        ),
        frames = plotly_frames
    )

fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0,
                x=0.5,
                xanchor='center',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(frame=dict(duration=dur, redraw=True), fromcurrent=True)]
                        )
                    ]
                )
            ]
        )

fig.update_layout(
        sliders=[
            dict(
                steps=[
                    dict(
                        method='animate',
                        args=[[frame.name], dict(mode='immediate', frame=dict(duration=dur, redraw=True))],
                        label=str(i*10)
                        )
                    for i, frame in enumerate(plotly_frames)
                    ],
                currentvalue=dict(prefix='Iteration: '),
                x=0.1,
                len=0.9
                )
            ]
        )

fig.show()
