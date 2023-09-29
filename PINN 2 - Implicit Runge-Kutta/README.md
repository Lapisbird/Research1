9/28/2023

My PyTorch implementation of another model from the following paper:

“Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations” by Maziar Raissi, Paris Perdikaris, and George Em Karniadakis.

This PINN is the one described in section "3. Discrete Time Models" (in particular the first one on Burger's equation).

The basic idea is to use a neural network to approximate the intermediate values of a very high-order implicit Runge-Kutta to enable high accuracy and large step values.

All code is my own work UNLESS I otherwise explicitly note on comments. Although the authors have completed models on their Github, it is written via Tensorflow, whereas my implementation will be via PyTorch. I will attempt to refrain from viewing their code as much as possible, although since I will be using some of their raw data (in particular the labels and the Butcher Tableau), I do plan on using their code for the extraction of that data (as their raw data is formatted in the manner they choose).