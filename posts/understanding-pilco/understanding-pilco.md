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
- Approximate Inference for Long-Term Predictions
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
$$
cov(z, \bm{y}) = cov(f_*, \bm{y}) + cov(A\bm{y}, \bm{y}) = K(\overset\sim{x_*}, \overset\sim{X}) + Avar(\bm{y}) 
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
$$var(f_* \mid \bm{y}) = k(\overset\sim{x_*}, \overset\sim{x_*}) - K(\overset\sim{x_*}, \overset\sim{X})K(\overset\sim{X}, \overset\sim{X} + \delta_{\epsilon}^2\mathcal{I})^{-1}K(\overset\sim{X}, \overset\sim{x_*})$$

### Approximate Inference for Long-Term Predictions
The target of the algorithm is to learn an optimal policy which minimizes the expected cost $J^\pi(\theta)=\sum_{t=0}^TE_{x_t}[c(x_t)], x_0 \sim \mathcal{N}(\mu_0,\Sigma_0)$ of following $\pi$ for $T$ steps (policy evaluation), where $c(x_t)$ is the cost (negative reward) of being in state $x$ at time $t$. We assume that $\pi$ is a function parameterized by $\theta$. We also have $J^\pi(\theta)=\int_0^Tc(x_t)p(x_t)dx_t$ which means we need to compute $p(x_t), t=1, \cdots, T$. Here we have $\Delta_{t-1} = x_t - x_{t-1}+\epsilon, \epsilon \sim \mathcal{N}(\bm{0}, \Sigma_\epsilon), \Sigma_\epsilon = diag([\delta_{\epsilon_1}, \cdots, \delta_{\epsilon_D}])$, we can write following prediction equations:
$$
p(x_t \mid x_{t-1}, \mu_{t-1}) = \mathcal{N}(x_t \mid \mu_t, \Sigma_t), \\
\mu_t = x_{t-1} + \mathbb{E}_f[\Delta_{t-1}], \\
\Sigma_t = var_f[\Delta_{t-1}]
$$
To compute $p(x_t)$, we need this
$$
p(x_t) = \iint p(x_t \mid x_{t-1}, u_{t-1})p(u_{t-1}, x_{t-1})dx_{t-1}du_{t-1}=\iint p(x_t \mid x_{t-1}, u_{t-1})p(u_{t-1} \mid x_{t-1})p(x_{t-1})dx_{t-1}du_{t-1}, t=1, \cdots, T
$$
Here, comparing with the derivation we did in last section, we actually need to predict distribution of $x_t$ at uncertain input $(x_{t-1}, u_{t-1})$ as when we propagate state through GP models, we propagate distribution of state so we can keep uncertainty of state through GP. As stated in original paper, this predictive distribution is not Gaussian and unimodal. However, we can approximate the predictive distribution by a Gaussian that possesses the same mean and variance. From now on, I will show how to compute mean and variance. However I want to start with a simple case first, so we can go through the math easily. As we can see in the last section, the input to our GP dynamics model should be a distribution over $(x_{t-1}, u_{t-1})$. However, I want to show how we can predict $p(x_t)$ given $p(x_{t-1})$ first. Then it would be easy to extend this case to having $u$ which is the control signal. Moreover, let's start with univariate predictions which is to predict one dimension of $x_t \in \mathbb{R}^D$ given an uncertain $x_{t-1} \in \mathbb{R}^D$. Now let's consider this problem, predicting a function value $f(x_{t-1}), f: \mathbb{R}^D \rightarrow \mathbb{R}$, at an **uncertain** previous state input $x_{t-1} \sim \mathcal{N}(\mu, \Sigma)$, where $f \sim GP$ with an $SE$ covariance function plus a noise covariance function. Let's write this predictive distribution as:
$$
p(x_t | \mathcal{N}(x_{t-1} \mid \mu, \Sigma)) = p(f(x_{t-1})|\mu, \Sigma) = \int p(f(x_{t-1}) \mid x_{t-1})p(x_{t-1} \mid \mu, \Sigma)dx_{t-1}
$$
Genrally, the predictive distribution in equation above cannot be computed analytically while we can compute the mean and variance of this distribution and use a Gaussian distribution with the same mean and variance to approximate it. To compute the mean $\mu_t$, we have:
$$
\mu_t = \iint f(x_{t-1})p(f, x_{t-1})d(f, x_{t-1}) = \iint f(x_{t-1})p(f,x_{t-1})dfdx_{t-1} = \int \mathbb{E}[f(x_{t-1}) \mid x_{t-1}]p(x_{t-1})dx_{t-1} \\
= \mathbb{E}_{x_{t-1}}[\mathbb{E}_{f}[f(x_{t-1}) \mid x_{t-1}] \mid \mu, \Sigma]
$$
Here we can see $\mathbb{E}_{f}[f(x_{t-1}) \mid x_{t-1}]$ is a predictive distribution at determinent input which we already know in last section. Plugging the result we have from the last section, we have:
$$
\mu_{t} = E_{x_{t-1}}[m_f(x_{t-1}) \mid \mu, \Sigma] = \int m_f(x_{t-1})\mathcal{N}(x_{t-1} \mid \mu, \Sigma)dx_{t-1} = \int K(x_{t-1}, X)(K(X, X) + \delta_{\epsilon}^2\mathcal{I}))^{-1}\bm{y}\mathcal{N}(x_{t-1} \mid \mu, \Sigma)dx_{t-1} \\ 
= \int K(x_{t-1}, X)\bm{\beta}\mathcal{N}(x_{t-1} \mid \mu, \Sigma)dx_{t-1} \\
= \bm{\beta}^T \int K(x_{t-1}, X)^T \mathcal{N}(x_{t-1} \mid \mu, \Sigma)dx_{t-1} \\
= \bm{\beta}^T\bm{q}
$$
where $\bm{\beta} := (K(X, X) + \delta_\epsilon^2\mathcal{I})^{-1}\bm{y}$, $\bm{q} = [q_1, \cdots, q_n]^T \in \mathbb{R}^n$ with 
$$
q_i := \int k(x_{t-1}, x_i)\mathcal{N}(x_{t-1}|\mu, \Sigma)dx_{t-1} \\
= \alpha^2|\Sigma\Lambda^{-1} + I|^{-\frac{1}{2}}exp(-\frac{1}{2}(x_{i} - \mu)^T(\Sigma + \Lambda)^{-1}(x_i - \mu))
$$
Now let's take a look at variance. According to law of total variance we have
$$
\delta_{t}^2 = var_{x_{t-1}, f}[f(x_{t-1}) \mid \mu, \Sigma] = \mathbb{E}_{x_{t-1}}[var_f[f(x_{t-1}) \mid x_{t-1}] \mid \mu, \Sigma] + var_{x_{t-1}}[\mathbb{E}_f[f(x_{t-1})\mid x_{t-1}] \mid \mu, \Sigma] \\
= \mathbb{E}_{x_{t-1}}[\delta_f^2(x_{t-1}) \mid \mu, \Sigma] + (\mathbb{E}_{x_{t-1}}[m_f(x_{t-1})^2 \mid \mu, \Sigma] - \mathbb{E}_{x_{t-1}}[m_f(x_{t-1}) \mid \mu, \Sigma]^2)
$$
Using the result from last section for $\delta_f$ and $m_{f}$, we have
$$
\delta_t^2 = \int k(x_{t-1}, x_{t-1}) - K(x_{t-1}, X)(K(X, X) + \delta_\epsilon^2\mathcal{I})^{-1}K(X, x_{t-1})p(x_{t-1})dx_{t-1} + \int K(x_{t-1}, X)\beta\beta^TK(X, x_{t-1})p(x_{t-1})dx_{t-1} - (\bm{\beta}^T\bm{q})^2
$$
Equation above can be computed analytically. 

Awesome! Now we know how to solve univariate predictions on uncertain input. Now we assume the target dimension is $E$. In the multivariate case, the predictive mean vector $\bm{\mu}_{t-1}$ is the coolection of all independently predicted means computed according to result equation in univariate predictions case.
$$
\bm\mu_{t-1} \mid \bm{\mu}, \bm{\Sigma} = [\bm\beta_1^T\bm{q}_1, \cdots, \bm\beta_E^T\bm{q}_E]^T
$$
However unlike predicting at deterministic inputs where all the target dimensions are independent, the target dimensions now covary, so the corresponding predictive covariance matrix
$$
\bm{\Sigma} \mid \bm{\mu}, \bm{\Sigma} = \begin{bmatrix}var_{f, x_{t-1}}[f_1(x_{t-1}) \mid \bm{\mu}, \bm{\Sigma}] & \cdots & cov_{f, x_{t-1}}[f_1(x_{t-1}), f_E(x_{t-1}) \mid \bm\mu, \bm\Sigma] \\
\vdots & \ddots & \vdots \\
cov_{f, x_{t-1}}[f_E(x_{t-1}), f_1(x_{t-1}) \mid \bm\mu, \bm\Sigma] & \cdots &  var_{f, x_{t-1}}[f_E(x_{t-1}) \mid \bm{\mu}, \bm{\Sigma}] \end{bmatrix}
$$
is no longer diagonal. The cross-covariances are given by
$$
cov_{f, x_{t-1}}[f_a(x_{t-1}), f_b(x_{t-1}) \mid \bm\mu, \bm\Sigma] = \mathbb{E}_{f, x_{t-1}}[f_a(x_{t-1})f_b({x_{t-1}}) \mid \bm{\mu}, \bm{\Sigma}] - \bm\mu_{a}(x_{t-1})\bm\mu_{b}(x_{t-1}). 
$$
The equation above can also be analytically computed. So that we can get the exact mean $\bm\mu_{t}$ and the exact covariance $\bm\Sigma_{t}$ of the generally non-Gaussian predictive distribution $p(f(x_{t-1}) \mid \bm\mu, \bm\Sigma)$.

Now let's come back to approximate inference for long-term predictions in PILCO. This step can be decomposed into following 3 substeps:
- A distribution $p(u_{t-1}) = p(\pi(x_{t-1}))$ over actions is computed when mapping $p(x_{t-1})$ through the policy $\pi$.
- A joint Gaussian distribution $p(x_{t-1}, u_{t-1}) = p(x_{t-1}, \pi(x_{t-1}))$ is computed.
- The distribution $p(x_t)$ is computed by applying the results we just got.

Now let's consider about computation of a distribution over actions. First of all, the policy itself should have two properties:
- For a state distribution $p(x)$ we need to be able to compute a corresponding distribution over actions $p(u) = p(\pi(x))$.
- In a realistic application, the policy must be able to deal with constrained control signals.

During the forward simulation, the states are given by probability distribution $p(x_t), t = 0, \cdots, T$. The probability distribution of the state $x_t$ induces a predictive distribution over actions, even if the policy is deterministic. To get constrained control signals, we assume the control limits are such that $\bm{u} \in [-\bm{u}_{max}, \bm{u}_{max}]$. Let us consider a preliminary policy $\overset\sim{\pi}$ with an unconstrained amplitude. To model the control limits coherently during simulation, we squash that preliminary policy $\overset\sim{\pi}$ through a bounded and differentiable squashing function, e.g. sine, logistic, cumulative Gaussian. However there are sevel advantages for using sine function. One advantageous property of the sine is that it allows for an analytic computation of the mean and the covariance of $p(\overset\sim{\pi}(\bm{x}))$ if $\overset\sim{\pi}(\bm{x})$ is Gaussian distributed so that we can use this mean and covariance to approximate the predictive distribution with a Gaussian distribution.

Following, I will show you one possible representation of the preliminary policy $\overset\sim\pi$ that allow for a closed-form computation of mean and the covariance of $p(\overset\sim{\pi}(\bm{x}))$ when the state $\bm{x}$ is Gaussian distributed which is a nonlinear representation of $\overset\sim{\pi}$, namely radial basis function (RBF) network.

#### RBF Network
The preliminary RBF policy is given by
$$
\overset\sim{\pi}(x_{*}) = \Sigma_{s=1}^N\beta_sk_\pi(x_s, x_*)=\bm\beta_\pi^Tk_\pi(\bm{X}_\pi,x_*),
$$
where $x_*$ is a test input, $k_\pi$ is a kernel function, and $\bm\beta_\pi:=(K_\pi+\delta_\pi^2\mathcal{I})^{-1}\bm{y}_\pi$, $\bm{X}_\pi and \bm{y}_\pi$ are training data. This is the familiar with what we have seen before and we can see this RBF network as a deterministic GP, which means $var_{\pi}=0$. Then we can use the results we got before to compute this predictive distribution and use a Gaussian distribution with exactly the same mean and variance to approximate this predictive distribution. Then we need to squash the preliminary policy through the sine and compute an approximate Gaussian distribution of $p(\bm{u}_{max}sin(\overset\sim{\pi}(x_{t-1})))$ which can be also computed analytically. Then we will need to compute the joint distribution $p(\bm{x}_{t-1}, \bm{u}_{t-1})=p(x_{t-1}, \pi(x_{t-1}))$. It is also computed in two steps. First, we compute the distribution $p(x_{t-1}, \overset\sim{\pi}(x_{t-1}))$. Then we compute an approximate fully joint Gaussian distribution $p(x_{t-1}, \overset\sim\pi(x_{t-1}), \bm{u}_{max}sin(\overset\sim\pi(x_{t-1})))$ and marginalize $\overset\sim{\pi}(x_{t-1})$ out to obtain the desired joint distribution $p(x_{t-1}, u_{t-1})$.

Use above results we can compute the predictive state distribution $p(x_{t}), t=1, \cdots, T$ iteratively. Then we can also compute the expective cumulative cost of following these steps.

### Policy Learning
Here we need to use a gradient-based policy search method to optimize the policy parameters. This means, we aim to find a parameterized policy $\pi^*$ from a class of policies $\Pi$ with
$$
\pi^* \in arg min_{\pi \in \Pi}J^{\pi_\psi}(x_0).
$$

## REFERENCES
- [PILCO: A Model-Based and Data-Efficient Approach to Policy Search](https://spiral.imperial.ac.uk/bitstream/10044/1/11585/4/icml2011_final.pdf)
- [Efficient Reinforcement Learning using Gaussian Processes](https://pdfs.semanticscholar.org/c9f2/1b84149991f4d547b3f0f625f710750ad8d9.pdf)

