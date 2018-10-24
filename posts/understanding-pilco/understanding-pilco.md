# Understanding PILCO
[Email](mailto:hjzh578@gmail.com) me or fire an issue [here](https://github.com/hjzh4/hjzh4.github.io/issues) if you have any question.
## Introduction
PILCO is a famous model-based reinforcement learning algorithm, which leverages gaussian process to model robotic system. Model-based reinforcement learning algorithms are considered to be data-efficient as they optimize the controller policy via the prediction of consquential states following current policy, instead of exploiting data which only comes from interacting with environment. However, a big issue in model-based reinforcement learning is that humans actually can't get a "right" model of a dynamic system (think about Laplace's Daemon), owing to uncertainty introduced by noisy measurements,  unobservable internal properties of the dynamic system,  etc. in which case the model bias is inevitably introduced when people assume that the learned dynamics model sufficiently accurately resembles the real environment. To capture the uncertainty of models, PILCO uses Gaussian Process to model the dynamic system. Moreover, Gaussian Process can also take as input a distribution and output another distribution, so that we can keep uncertainty of models, prediction through the whole algorithm pipeline.

Gaussian Process is a very powerful machine learning tool, while it's also way less popular than neural networks in current robotics community, because it's way more analytically harder than neural networks though mathmatically easier. Owing to this, PILCO is also very hard to understand for a beginner like me (It seems there are also a lot of overloadings of notions in paper...). In this blog, I will derivate the whole framework of PILCO for my own learning purpose. I also hope that it can help other beginners like me to understand PILCO easier.

In this blog, I just assume you have a basic linear algebra and calculus and reinforcement learning background.

## Gaussian Process
Understanding Gaussian Process is the basis for understanding PILCO. Here are some resources that introduce me to this field.
- [Gaussian Process for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
- [Gaussian Process Summer School 2018](http://gpss.cc/gpss18/day-1.html)
- [Machine Learning Summer School 2012](https://www.youtube.com/watch?v=BS4Wd5rwNwE)

## PILCO
Let's start from pseudo code.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**init**: Sample controller parameters $\theta \sim \mathcal{N}(\bm{0}, \mathcal{I})$. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**repeat** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Apply random control signals and record data. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Learn Gaussian Process dynamics model using all data. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**repeat** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Simulate system with current controller $\pi(\theta)$ and get expected cost $J^\pi(\theta)$. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Improve controller paramters to $\theta^*$. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**until** convergence \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set controller parameters to $\theta^*$. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**until** task learned

From the pseudo code, we can see there are three key components of PILCO:
- Dynamics Model Learning
- Policy Evaluation
- Policy Improvement

### Dynamics Model Learning
In PILCO, the input of dynamics model is a vector $\overset\sim{x}:=[x_{(1)}^T, \cdots, x_{(D)}^T, u_{(1)}^T, \cdots, u_{(F)}^T]^T$, where vector $x\in \mathbb{R^D}$ is our observation of the dynamic system, vector $u \in \mathbb{R^F}$ is the control signal from controller, and the output of dynamics model is a vector $\Delta$ representing the increment of next state with reference to current state. 

Predicting relavant state increment instead of absolute state can make the prediction more smooth as the difference of the state increment is very small. Moreover, to predict a multi-dimensional state target, we train conditionally independent GPs for each state target dimension. So following, we just consider a single GP for one dimension of state target. Then we use 
$$
\Delta_{t-1}^{(a)} = f(\overset\sim{x_{t-1}}) + \epsilon, \epsilon \sim \mathcal{N}(0, \mathcal{\delta_\epsilon^2\mathcal{I}})
$$
to represent a distribution we want to compute. It is actually similar with Gaussian Process Regression, where we represent underlying true system dynamics as a laten function $f(\overset\sim{x_{t-1}})$ and represent uncertainty as $\epsilon$.

The key of Gaussian Process Prediction is to combine a prior distribution and likelihood of data to predict a posterior distribution of the random variable $f_*$ at an unknown input $\overset\sim{x_*}$. Here we represent our data as $\overset\sim{X}=[\overset\sim{x_1}, \cdots, \overset\sim{x_n}]^T, \bm{y} = [\Delta_1^{(a)}, \cdots, \Delta_n^{(a)}]^T$. So what we need to compute is the distribution of the random variable $f_*$ given $\overset\sim{X}, \bm{y}$ and a new input $\overset\sim{x_*}$. Following, I will show how to compute this conditional distribution.

In the paper of PILCO, for Gaussian Process, authors only consider a prior mean function $m \equiv 0$ and a prior covariance function $k(\overset\sim{x}, \overset\sim{x}')=\alpha^2exp(-\frac{1}{2}(\overset\sim{x}-\overset\sim{x}')^T\Lambda^{-1}(\overset\sim{x}-\overset\sim{x}'))$ with $\Lambda := diag([l_1^2, \cdots, l_{D + F}^2])$. 

To compute the conditional distribution of prediction $f_*$, we need to get a prior joint distribution of the data $\bm{y}$ and $f_*$ first. we know a prior distribution of $\bm{y}$:
$$
\bm{y} \sim N(\bm{0}, K(\overset\sim{X}, \overset\sim{X}) + \delta_\epsilon^2\mathcal{I})
$$
Also, we know that $f_*$ is distributed as a normal variable, because of Gaussian Process assumption. So we know the joint distribution of $\bm{y}$ and $f$ is also normal distribution which can be written as:
$$
\begin{bmatrix} 
f_* \\ 
\bm{y}
\end{bmatrix} \sim N(\begin{bmatrix}0 \\ \bm{0}\end{bmatrix}, \begin{bmatrix} k(\overset\sim{x_*}, \overset\sim{x_*}) &  K(\overset\sim{x_*}, \overset\sim{X}) \\ K(\overset\sim{X}, \overset\sim{x_*})   & K(\overset\sim{X}, \overset\sim{X}) + \delta_\epsilon^2\mathcal{I}\end{bmatrix})\\$$
Where $K(\overset\sim{X}, \overset\sim{x_*})$ is a $n \times 1$ vector computed from evaluation of covariance function.

Having a prior distribution, we can get posterior distribution using math. Now I will go through this. Instead of using Bayesian Rules, I will show a smarter method to do this. We first define $z = f_* + A\bm{y}$, where $A = -K(\overset\sim{x_*}, \overset\sim{X})(K(\overset\sim{X}, \overset\sim{X}) + \delta_\epsilon^2\mathcal{I}))^{-1}$. Now, we have 
$$cov(z, \bm{y}) = cov(f_*, \bm{y}) + cov(A\bm{y}, \bm{y}) = K(\overset\sim{x_*}, \overset\sim{X}) + Avar(\bm{y}) 
= K(\overset\sim{x_*}, \overset\sim{X}) - K(\overset\sim{x_*}, \overset\sim{X})(K(\overset\sim{X}, \overset\sim{X}) + \delta_\epsilon^2\mathcal{I}))^{-1}(K(\overset\sim{X}, \overset\sim{X}) + \delta_\epsilon^2\mathcal{I}))
=\bm{0} $$
Therefore $z$ and $\bm{y}$ are uncorrelated and since they are jointly normal, they are independent. To compute conditional mean we have
$$
E(f_* \mid \bm{y}) = E(z - A\bm{y} \mid \bm{y}) = E(z \mid \bm{y}) - E(A\bm{y} \mid \bm{y}) = E(z) - E(A\bm{y}) = E(z) - A\bm{y} = K(\overset\sim{x_*}, \overset\sim{X})(K(\overset\sim{X}, \overset\sim{X}) + \delta_\epsilon^2\mathcal{I}))^{-1}\bm{y}
$$
To compute variance, 
$$
var(f_* \mid \bm{y}) = var(z - A\bm{y} \mid \bm{y}) = var(z \mid \bm{y}) = var(z) 
$$
$$
var(f_* \mid \bm{y}) = var(z) = var(f_* + A\bm{y}) = var(f_*) + var(A\bm{y}) + cov(f_*, A\bm{y}) + cov(A\bm{y}, f_*) = var(f_*) + Avar(\bm{y})A^T + cov(\bm{y}, f_*)A^T + Acov(f_*, \bm{y})
$$
$$var(f_* \mid \bm{y}) = k(\overset\sim{x_*}, \overset\sim{x_*}) - K(\overset\sim{x_*}, \overset\sim{X})K(\overset\sim{X}, \overset\sim{X})^{-1}K(\overset\sim{X}, \overset\sim{x_*})$$



