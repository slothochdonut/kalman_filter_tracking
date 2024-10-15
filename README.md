# bayesian-filtering-beetles
Master thesis

a project which uses kalman filter to filter out the noise of a video tracked beetle trajectory and predict the true one. By doing this the state of the beetle movement is first assumed as only trajectories, then added velocity. A problem concerned is the choice of parameters in the underlying Guassion noise, the mean and variance. 

There are two stages in basic kalman filter, the observation and the estimation of the true state. This means the process is iterative updating process.