# 3D Drone Path Simulation

This repository contains a minimal 3D drone simulation built in Python with the aim of illustrating drone navigation and physics modeling. The endgoal is to demonstrate a simple pipeline for simulating a drone moving toward a target under noisy sensing, imperfect thrust actuation, and basic physical constraints.

The project includes:

- A lightweight physics model integrated with Runge-Kutta 4 integration scheme
- A PD controller driving the drone toward a goal  
- An actuator model with thrust limits and noise  
- Basic obstacle representations  
- Tools for visualizing trajectories and generating 3D animations  

The notebook `blog/drone_simulation.ipynb` explains and implements the details of the simulation.

## Example result

Here is a small video of the result of the simulation

<video src="./assets/hero.mp4">