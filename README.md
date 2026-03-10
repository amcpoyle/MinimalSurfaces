# Minimal Surfaces
minimizer_numerical.py takes a surface element $f: U \rightarrow \mathbb{R}^{3}$ and optimizes it to a minimal surface ($H = 0$ everywhere in $U$). run_minimizer_animation runs the minimizer and creates the plotly frames required to create a 3d surface plot animation of the evolution of the surface optimization.

minimizer.py was an attempt to perform surface minimization symbolically, but it was too slow so I switched to numerical methods.

An example animation of an arbitrary surface that optimizes into a minimal surface via iterating normal variation:

https://github.com/user-attachments/assets/496d8d45-88f6-4b1e-8bab-8ba37efaec98

