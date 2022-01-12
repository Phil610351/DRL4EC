# Computation Offloading in Edge Computing of Industrial Internet based on Deep Reinforcement Learning
## Introduction - Thesis
This project contains codes of `Yifan Zhu`'s undergraduate thesis.

This thesis focuses on multi-objective modeling of the computational offloading problem
in edge computing with the main demands of industrial Internet applications, and adopts deep
reinforcement learning as a solution method to optimize the resource allocation of edge
computing and provide a comprehensive optimal solution for task scheduling in the context of
industrial Internet. Currently, industrial Internet applications have a high demand for low
latency and low cost: most applications require the shortest possible response time due
to QoS considerations; the more resource-rich servers have higher cost per unit time.
Therefore, under the consideration of cost control and low latency, this thesis proposes
a multi-objective optimization problem with minimizing both as the optimization objective
and takes the set of execution locations of tasks as the decision variables. Considering the
difficulty of solving the multi-objective optimization problem, this study solves the problem
with a deep reinforcement learning algorithm and proposes to self-learn to get the optimal
strategy. Considering the convergence speed and the robustness of the results, the proposed
weighted summation method is used to decompose the multi-objective problem into
multiple scalar optimization subproblems, while the subproblems are modeled using a
pointer network and the actor-critic method is used to train the model parameters in order to
learn to obtain the best optimization strategy and obtain the Pareto front. Three deep
reinforcement learning models are trained in the experiments, and the neighborhood
parameter transfer method, generalization performance, and convergence and distributivity
of the Pareto frontier are analyzed and evaluated respectively.

## Introduction - project
1. To train the model, run main.py
2. Trained model is available in these directories:
  - `ec_no_transfer`: model trained without parameter transfer strategy.
  - `ec_transfer-10ec-20task`: model trained to solve problems with 10 servers and 20 tasks.
  - `ec_transfer-10ec-40task`: model trained to solve problems with 10 servers and 40 tasks.
3. To test trained model, modify the model directory and run test.py

**Please note that many codes are inherited from https://github.com/mveres01/pytorch-drl4vrp**
