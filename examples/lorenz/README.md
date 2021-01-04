# Lorenz attractor estimation example
This example shows nonlinear filters for a chaotic Lorenz attractor system.  An extend Kalman filter (EKF) and particle filter (PF) are used.  Options to create a video animation for the particle filter are included.

To run the example, build and install the C++ libraries from the main README instructions.  Start the docker container.
```
cd libsia/examples/lorenz
python lorenz.py --help
python lorenz.py
```
