import numpy as np
import matplotlib.pyplot as plt
from minimizer_numerical import plot_surface, run_minimizer, run_minimizer_animation
import plotly.graph_objects as go
from PIL import Image
import io

r0 = 0.5 # 1 = cylinder, 0 = already singularity

def r(z):
    global r0

    # defining r(z) function
    value = r0 + (r0 - 1)*(z**4) - 2*(r0 - 1)*(z**2)
    return value



def quartic_eqn(dur):
    N = 25
    theta_range = np.linspace(0, 2*np.pi, N)
    z_range = np.linspace(-1,1, N)

    dtheta = theta_range[1] - theta_range[0]
    dz = z_range[1] - z_range[0]
    T, Z = np.meshgrid(theta_range, z_range, indexing='ij')
    f_mesh = np.zeros((N,N,3))
    H_mesh = np.zeros((N,N))
    nu_mesh = np.zeros((N,N,3))

    for i in range(N):
        for j in range(N):
            theta_val = theta_range[i]
            z_val = z_range[j]
            
            u_val = r(z_val)*np.cos(theta_val)
            v_val = r(z_val)*np.sin(theta_val)
            w_val = z_val
            f_mesh[i][j] = np.array([u_val, v_val, w_val]).astype(float).flatten()


    original_fig = plot_surface(T, Z, f_mesh)
    original_fig.show()
    plt.show()
    tol_eps = 1e-6
    eps = (min(dtheta, dz)**2)/4
    f_mesh_min, H_mesh_min, nu_mesh_min, plotly_frames = run_minimizer_animation(N, theta_range, z_range, dtheta, dz,
                                                        T, Z, f_mesh, H_mesh, nu_mesh,
                                                        tol_eps, eps, 10)

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

def main():
    quartic_eqn(100)


if __name__ == '__main__':
    main()
