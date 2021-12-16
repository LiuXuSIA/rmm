## random mapping method (RMM) for terrain modeling
A simple demo of random mapping method for terrain modeling published in AAAI22. <br>
__scripts__ :  A simple implementation codes for RMM with Python 3.8, numpy 1.21.2, and scipy 1.6.2 <br> 
__datasets__ : The used datasets comprising *planet*, *quarry*, and *mountain*. <br>
The underlying mathematical proof regarding the linear property of RMM can be found in our paper.
## What's RMM
RMM is a feature mapping method, based on the fast random construction of base functions,
which can efficiently project the messy points in the low-dimensional space into
the high-dimensional space where the points are approximately linearly distributed.  
## Why RMM
In the context of machine learning, by efficiently generating approximately linearly separable
or distributed space at random, the RMM can accelerate the training process. 
## How RMM
For an arbitrary set of samples $\boldsymbol X = \{\boldsymbol x_i,t_i\}_{i=1}^{L}$, where $\boldsymbol x_i \in {\boldsymbol R}^{N\times 1}$ is the feature vector <br>
and $t_i \in {\boldsymbol R}^m$ is the target value. RMM generate a feature mapping function $\phi (\boldsymbol x_i)$ for the vector $\boldsymbol x_i$ as follows:
$$\phi(\boldsymbol x_i)=g(\boldsymbol {Wx_i}+\boldsymbol b)=g(\boldsymbol v_i)=\boldsymbol s_i$$
where $\boldsymbol W$ is a $M \times N$ matrix denoting a linear transformation, and $b$ is a $M \times 1$ bias vector. 
Particularly, the elements of $W$ and $b$ are generated at random from a probability distribution, such as a uniform distribution.
Then, for the $\boldsymbol X$ that is not linearly distributed, we can obtain a approximately linearly distributed set as follows
$$\boldsymbol S=\boldsymbol G(\boldsymbol{WX})=\boldsymbol G(\boldsymbol V)$$

## What's terrain modeling
Assume that a robot has captured a data set $\boldsymbol D=\{\boldsymbol x_i, t_i\}_{i=1}^{L}$ by the end points of a laser ranger finder or depth cameras when moving in the environments, where $\boldsymbol x_i$ is a 2D location and $t_i$ is its elevation. Our idea for for the terrain modeling is to build a linear regression model between $\{\boldsymbol x_i\}_{i=1}^{L}$ and $\{t_i\}_{i=1}^{L}$.
## How terrain modeing with RMM
Two points. Firlrly, RMR can be effective for regression tasks. Secondly, treat the terrain modeling as a regression problem.
Oegression model: $$y=f(\boldsymbol x, \boldsymbol\beta)=\boldsymbol\beta^T\boldsymbol g(\boldsymbol {Wx})+b=\boldsymbol\beta^T\boldsymbol s+b=\boldsymbol\beta^{T}\boldsymbol s$$
Objective funtion: 
$$\nabla J(\boldsymbol \beta)=\boldsymbol {SS}^T\boldsymbol \beta - \boldsymbol{ST}^T+\alpha \boldsymbol \beta$$
The solution can be acquired through using Cholesky decomposition, singular value decomposition, Penrose-Moore generalized inverse, or stochastic gradient descent.

## Why terrain modeing with RMM
In the context of robotics mapping, the vast amount of data captured by robots in large-scale environments
brings the computing and storage bottlenecks to the typical methods of modeling the spaces the robots travel in.
- The randomness and closed-form solutions make RMR very time-efficient, accelerating the terrain modeling process.
- The limited parameters of RMR make RMR very memory-efficient, reducing the occupied stroage space.
- The accurate intepolation ability of RMR make RMR can fill the terrin gaps, generating more complete and detailed terrain maps.

## Tips
- The RMM is sensitive to the data scale.
It is better to scale or normalize the data.
In our implemantation, we first generate all the random weights $W$ from the uniform distribution between [-1, 1],
and the we scale the generated weight using a scaling factor $\alpha$, that is the used weight matrix $W'=\alpha W$.
Particularly, we treat the scaling factor  $\alpha$ as another hyperparameter, i.e., the *scaleRate*.
In the future, we will first study the relationship between the results of RMR and the data scale. 
- The RMM can also be used for classification task.
Indeed, classification task is a simple version of regression task. Refer to our paper for the derivation.
- Terrain modeling with RMR is required to extract the terrain surface first.
