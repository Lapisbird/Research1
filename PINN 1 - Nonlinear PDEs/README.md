In this folder lies my implementation of the PINN described in section 2 of the following paper:

“Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations” by Maziar Raissi, Paris Perdikaris, and George Em Karniadakis.

https://arxiv.org/abs/1711.10561

I am writing this code for practice. It will be written in PyTorch. The original code from the paper, which I have not and will not see, was written using Tensorflow.

Although all code is my own implementation, I will make explicit via comments any code I have written which is not a direct implementation of a technique or process described in the sections of the paper I am referencing.



UPDATE 9/22/2023:

I had significant difficulty with the proper convergence of my original code and all variations of it. It seemed that my model would always either fail to converge or converge into an approximate of a flat plane - far from the desired result. After an eternity of troubleshooting and attempted fixes, I consulted the following implementation of the model in PyTorch (which I was not previously aware of) to try and find my issue:

https://github.com/teeratornk/PINNs-2/blob/master/Burgers%20Equation/Burgers%20Inference%20(PyTorch).ipynb

This model, although written to implement the same paper and in PyTorch, followed a significantly different programming style and architecture than I did. After a week of attempts, I was still unable to diagnose my issue.

So, I decided to completely re-build my model, still following my initial style of programming and architecture, not the reference’s. After slowly rebuilding my model from the ground up, implementing features one-at-a-time, I ended up creating a functioning model.

My earlier failed attempts at a model are in the “Earlier Attempts” directory, while my new functioning model is in the “PINN From the Ground Up” directory and is specifically the “GroundUpPINN4.ipynb” file.

In the end, the only code which I directly used from the PyTorch implementation referenced was the parameter values for LBFGS, which I also credited to them in my code at that section. Apart from that, I also drew inspiration from their code in the following ways:

* Removal of my dataloader due to their lack of one
* Implementation of a self-written MSE_u function. Although this was later reverted to nn.MSELoss
* Separation of the t and x components of my collocation point array prior to utilization

These changes were all implemented during my attempt to try and fix my original code. Although I have kept them (except for the MSELoss), I do not strictly know if they are necessary changes.

Apart from these minor inspirations, the rest of the code in my final GroundUpPINN4.ipynb file is wholly my own implementation, and follows the same structure laid out in my original attempts.

Even now, with the long-eluded goal of a functional model achieved, I do not know precisely why my original implementations failed. However, having already spent many dozens of hours attempting to fix it, I have decided to no longer pursue an exact answer. My assumption is that it stems from some trivial error in the code that I have simply failed to catch.


UPDATE 9/22/2023:

I ended up spending some time trying to diagnose the reasoning behind my original PINN's failure by systemically replacing componenets with code from my new, working PINN. And I found the issue.

What was it?

The issue was that the labels had shape [n] instead of [n, 1].

For some reason, with shape [n], the code still ran without exceptions, but result in the previously discussed failure to properly converge.

I am certainly less than pleased that dozens of hours of work had to be spent to diagnose such a trivial error. However, I am also glad to have had the experience. Not only have I now learned to respect the subtly and fragility of the training process; code without exceptions does not equate with code without errors. But I have also learned and gained familiarility with much of PyTorch in a deeper way through my hunt for the issue.