import numpy as np
import matplotlib.pyplot as plt
from minimizer_numerical import plot_surface, run_minimizer, run_minimizer_animation, run_minimizer_blender
import plotly.graph_objects as go
from PIL import Image
import io
import trimesh
from scipy.interpolate import griddata

# sort of an arbitrary surface example:
def surface1(dur):
    N = 25
    u_range = np.linspace(0, 2*np.pi, N)
    v_range = np.linspace(0, 2*np.pi, N)

    du = u_range[1] - u_range[0]
    dv = v_range[1] - v_range[0]
    U, V = np.meshgrid(u_range, v_range, indexing='ij')
    f_mesh = np.zeros((N,N,3))
    H_mesh = np.zeros((N,N))
    nu_mesh = np.zeros((N,N,3))

    for i in range(N):
        for j in range(N):
            x_val = u_range[i]
            y_val = v_range[j]
            z_val = np.sin(np.sqrt(u_range[i]**2 + v_range[j]**2))
            f_mesh[i][j] = np.array([x_val, y_val, z_val]).astype(float).flatten()


    original_fig = plot_surface(U, V, f_mesh)
    tol_eps = 1e-3
    eps = (min(du, dv)**2)/4
    f_mesh_min, H_mesh_min, nu_mesh_min, plotly_frames = run_minimizer_animation(N, u_range, v_range, du, dv,
                                                        U, V, f_mesh, H_mesh, nu_mesh,
                                                        tol_eps, eps, 15, fix_u_boundary=True, fix_v_boundary=True, periodic=False)

    # blender_frames = np.array(blender_frames)
    # np.save("surface1_frames.npy", blender_frames)

    # CODE FOR PLOTLY ANIMATION:
    fig = go.Figure(
            data = go.Surface(
                x=f_mesh[:,:,0],
                y=f_mesh[:,:,1],
                z=f_mesh[:,:,2],
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


    # new_fig = plot_surface(U, V, f_mesh)
    # original_fig.show()
    # new_fig.show()
    # plt.show()

# right helicoid (already a minimal surface) example:
def right_helicoid():
    N = 25
    u_range = np.linspace(-1, 1, N)
    v_range = np.linspace(-np.pi, np.pi, N)

    du = u_range[1] - u_range[0]
    dv = v_range[1] - v_range[0]
    U, V = np.meshgrid(u_range, v_range, indexing='ij')
    f_mesh = np.zeros((N,N,3))
    H_mesh = np.zeros((N,N))
    nu_mesh = np.zeros((N,N,3))

    for i in range(N):
        for j in range(N):
            x_val = u_range[i]
            y_val = v_range[j]
            z_val = np.sin(np.sqrt(u_range[i]**2 + v_range[j]**2))
            f_mesh[i][j] = np.array([x_val, y_val, z_val]).astype(float).flatten()



    # plot my original surface
    original_fig = plot_surface(U, V, f_mesh)

    tol_eps = 1e-6
    eps = (min(du, dv)**2)/4

    f_mesh_min, H_mesh_min, nu_mesh_min = run_minimizer(N, u_range, v_range, du, dv,
                                                        U, V, f_mesh, H_mesh, nu_mesh,
                                                        tol_eps, eps)
    
    new_fig = plot_surface(U, V, f_mesh)
    original_fig.show()
    new_fig.show()
    plt.show()



def sphere(dur):
    # # 1 - original mesh generation
    # N = 25
    # z_range = np.linspace(-1, 1, N) # z
    # print('z range: ', z_range)
    # theta_range = np.linspace(0, 2*np.pi, N, endpoint=False) # theta

    # dz = z_range[1] - z_range[0]
    # dtheta = theta_range[1] - theta_range[0]
    # T, Z = np.meshgrid(theta_range, z_range, indexing='ij')

    # f_mesh = np.zeros((N,N,3))
    # H_mesh = np.zeros((N,N))
    # nu_mesh = np.zeros((N,N,3))

    # r = 1

    # for i in range(N):
    #     for j in range(N):
    #         theta_val = theta_range[i]
    #         z_val = z_range[j]
    #         x_val = np.sqrt(1 - z_val**2)*np.cos(theta_val)
    #         y_val = np.sqrt(1 - z_val**2)*np.sin(theta_val)
    #         z_val = z_val
    #         f_mesh[i][j] = np.array([x_val, y_val, z_val]).astype(float).flatten()



    # # plot my original surface
    # original_fig = plot_surface(T, Z, f_mesh)
    # # original_fig.show()
    # # plt.show()


    # 2 - triangular mesh generation
    sphere_tm = trimesh.creation.icosphere(subdivisions=3)
    verts = sphere_tm.vertices

    theta_ico = np.arctan2(verts[:,1], verts[:,0])
    z_ico = verts[:,2]

    N = 25
    z_range = np.linspace(-1, 1, N) # z
    theta_range = np.linspace(-np.pi, np.pi, N, endpoint=False) # theta

    dz = z_range[1] - z_range[0]
    dtheta = theta_range[1] - theta_range[0]
    T, Z = np.meshgrid(theta_range, z_range, indexing='ij')

    f_mesh = np.zeros((N,N,3))
    points = np.stack([theta_ico, z_ico], axis=1)
    for k in range(3):
        f_mesh[:, :, k] = griddata(points, verts[:,k], (T,Z), method='linear')

    H_mesh = np.zeros((N,N))
    nu_mesh = np.zeros((N, N, 3))

    tol_eps = 1e-10
    eps = ((min(dtheta, dz)**2)/4)

    f_mesh_min, H_mesh_min, nu_mesh_min, plotly_frames, had_singularity, det_gij_min_idx = run_minimizer_animation(N, theta_range, z_range, dtheta, dz,
                                                        T, Z, f_mesh, H_mesh, nu_mesh,
                                                        tol_eps, eps, 1, fix_u_boundary=False, fix_v_boundary=False, periodic=True)
    

    fig = go.Figure(
            data = go.Surface(
                x=f_mesh[:,:,0],
                y=f_mesh[:,:,1],
                z=f_mesh[:,:,2],
                colorscale='Viridis',
                cmin=-1,cmax=1
            ),
            frames = plotly_frames
        )

    fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-1, 1]),
                yaxis=dict(range=[-1,1]),
                zaxis=dict(range=[-1,1]),
                aspectmode='data'
            )
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
    # new_fig = plot_surface(U, V, f_mesh)
    # original_fig.show()
    # new_fig.show()
    # plt.show()

   
def main():
    sphere(100)


if __name__ == '__main__':
    main()
