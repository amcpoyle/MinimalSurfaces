import numpy as np
import matplotlib.pyplot as plt
from minimizer_numerical import plot_surface, run_minimizer, run_minimizer_animation
import plotly.graph_objects as go

# sort of an arbitrary surface example:
def surface1():
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
    tol_eps = 1e-6
    eps = (min(du, dv)**2)/4
    f_mesh_min, H_mesh_min, nu_mesh_min, plotly_frames = run_minimizer_animation(N, u_range, v_range, du, dv,
                                                        U, V, f_mesh, H_mesh, nu_mesh,
                                                        tol_eps, eps)

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
                            args=[None, dict(frame=dict(duration=16, redraw=True), fromcurrent=True)]
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
                            args=[[frame.name], dict(mode='immediate', frame=dict(duration=16, redraw=True))],
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

   
def main():
    surface1()

if __name__ == '__main__':
    main()
