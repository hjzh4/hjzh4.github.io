# Understanding PILCO
[Email](mailto:hjzh578@gmail.com) me or fire an issue [here](https://github.com/hjzh4/hjzh4.github.io/issues) if you have any question.
## Introduction
PILCO is a famous model-based reinforcement learning (RL) algorithm, which leverages gaussian process (GP) to model robotic dynamics system. Model-based RL are considered to be data-efficient as they learn a dynamics model of the environment so that they can predict next state given the current state and the control signal. However, a big issue in model-based RL is that people used to assume that the learned dynamics model sufficiently accurately resembles the real environment which introduces model bias inevitably. However humans actually cannot get a "deterministic" model of a dynamic system (think about Laplace's Daemon), owing to uncertainty introduced by noisy measurements,  unobservable internal properties of the dynamic system,  etc. To capture the uncertainty of models, PILCO uses a GP to model the dynamic system. Moreover, GP can also take as input a probablistic distribution and output another probablistic distribution, so that we can propagate the uncertain state through the model.

GP is a very powerful machine learning tool like neural networks. Especially in the robotics community, data-efficiency is the key problem towards the feasible employment of learning method and model-based RL is a promising way to reduce data-inefficiency and that's why I'm so interested in PILCO and GP now. In this blog, I try to derivate the PILCO and go through the math. I also recommend everyone to read Marc Deisenroth's Ph.D. thesis where he explained PILCO more clearly than in his original paper.

## Gaussian Process
Understanding GP is the basis for understanding PILCO. Here are some resources that introduce me to this field.
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
- Policy Learning

In PILCO, we just do these three things repeadtedly.

### Dynamics Model Learning
In PILCO, the input of dynamics model is a vector $\overset\sim{\bm{x}}:=[x_{(1)}^T, \cdots, x_{(D)}^T, u_{(1)}^T, \cdots, u_{(F)}^T]^T$, where vector $\bm{x}\in \mathbb{R}^D$ is our observation of the dynamic system, vector $\bm{u} \in \mathbb{R}^F$ is the control signal from controller, and the output of dynamics model is a vector $\bm{\Delta}$ representing the increment of next state with reference to current state. 

Predicting relavant state increment instead of absolute state can make the prediction more smooth as the difference of the state increment is basically very small. Moreover, to predict a multi-dimensional state target (very common in robotics), we train conditionally independent GPs for each state target dimension. Following we will start from considering one single GP for one dimension of target state. As I said before uncertain state (probablistic distribution of state) can be propagated through the model in PILCo, this actually means we need to compute a predictive distribution $p(\bm{x}_{*})$ given another predictive distribution $p(\bm{x}_{t-1})$, for example. However I will start from computing the predictive distribution at a deterministic input and then I will show how to compute it at a deterministic input. This section is about computing the predictive distribution of a single dimension of the target state at a deterministic input. Assumining we have data $\overset\sim{\bm{X}}=[\overset\sim{\bm{x}_1}, \cdots, \overset\sim{\bm{x}_n}]^T, \bm{y} = [\Delta_1, \cdots, \Delta_n]^T$ and we want to compute a predictive distribution $f(\overset\sim\bm{x}_{t-1})$ given a deterministic input $\bm{x}_{t-1}$. We have a equation for collected data 
$$
\Delta_{*} = f(\overset\sim{\bm{x}_{*}}) + \epsilon, \epsilon \sim \mathcal{N}(0, \mathcal{\delta_\epsilon^2\mathcal{\bm{I}}}),
$$
where $f$ is an underlying dynamics function we want to predict, $\epsilon$ is the measurement uncertainty. Then we can use GP regression to regress $f$, we write
$$
p(f(\bm{x}_*) \mid \bm{x}_*, \overset\sim\bm{X}, \bm{y}).
$$
In the paper of PILCO, to define GP for $f$, authors only consider a prior mean function $m \equiv 0$ and a prior covariance function $k(\overset\sim\bm{x}, \overset\sim\bm{x}')=\alpha^2exp(-\frac{1}{2}(\overset\sim\bm{x}-\overset\sim\bm{x}')^T\bm\Lambda^{-1}(\overset\sim\bm{x}-\overset\sim\bm{x}'))$ with $\bm\Lambda := diag([l_1^2, \cdots, l_{D + F}^2])$. 

As GP regression, to compute the conditional distribution of prediction $f_*$, we need to get a prior joint distribution of the data $\bm{y}$ and $f_*$ first. we know a prior distribution of $\bm{y}$:
$$
\bm{y} \sim N(\bm{0}, K(\overset\sim\bm{X}, \overset\sim\bm{X}) + \delta_\epsilon^2\mathcal{\bm{I}}).
$$
Also, we know that $f_*$ is distributed as a normal variable. So we know the joint distribution of $\bm{y}$ and $f$ is also a normal distribution which can be written as:
$$
\begin{bmatrix} 
f_* \\ 
\bm{y}
\end{bmatrix} \sim N(\begin{bmatrix}0 \\ \bm{0}\end{bmatrix}, \begin{bmatrix} k(\overset\sim\bm{x}_*, \overset\sim\bm{x}_*) &  K(\overset\sim\bm{x}_*, \overset\sim\bm{X}) \\ K(\overset\sim\bm{X}, \overset\sim\bm{x}_*)   & K(\overset\sim\bm{X}, \overset\sim\bm{X}) + \delta_\epsilon^2\mathcal{\bm{I}}\end{bmatrix})\\$$
where $K(\overset\sim\bm{X}, \overset\sim\bm{x}_*)$ is a $n \times 1$ vector computed by evaluating the covariance function.

Having a this joint prior distribution, we can get posterior distribution using math! Instead of using Bayesian Rules brute-forcely, I will show a smarter method to do this! We first define $z = f_* + \bm{A}\bm{y}$, where $\bm{A} = -K(\overset\sim{\bm{x}_*}, \overset\sim{\bm{X}})(K(\overset\sim\bm{X}, \overset\sim\bm{X}) + \delta_\epsilon^2\mathcal{\bm{I}}))^{-1}$. Now, we have 
$$
cov(z, \bm{y}) = cov(f_*, \bm{y}) + cov(\bm{A}\bm{y}, \bm{y}) = K(\overset\sim{\bm{x}}_*, \overset\sim\bm{X}) + \bm{A}var(\bm{y}) 
= K(\overset\sim\bm{x}_*, \overset\sim\bm{X}) - K(\overset\sim\bm{x}_*, \overset\sim\bm{X})(K(\overset\sim\bm{X}, \overset\sim\bm{X}) + \delta_\epsilon^2\mathcal{\bm{I}}))^{-1}(K(\overset\sim\bm{X}, \overset\sim\bm{X}) + \delta_\epsilon^2\mathcal{\bm{I}}))
=\bm{0}. 
$$
Therefore $z$ and $\bm{y}$ are uncorrelated and since they are jointly normal, they are independent. To compute conditional mean we have
$$
\mathbb{E}(f_* \mid \bm{y}) = \mathbb{E}(z - \bm{A}\bm{y} \mid \bm{y}) = \mathbb{E}(z \mid \bm{y}) - \mathbb{E}(\bm{A}\bm{y} \mid \bm{y}) = \mathbb{E}(z) - \mathbb{E}(\bm{A}\bm{y}) = \mathbb{E}(z) - \bm{A}\bm{y} = K(\overset\sim\bm{x}_*, \overset\sim\bm{X})(K(\overset\sim\bm{X}, \overset\sim\bm{X}) + \delta_\epsilon^2\mathcal{\bm{I}}))^{-1}\bm{y}.
$$
To compute variance, 
$$
var(f_* \mid \bm{y}) = var(z - \bm{A}\bm{y} \mid \bm{y}) = var(z \mid \bm{y}) = var(z) 
$$
$$
var(f_* \mid \bm{y}) = var(z) = var(f_* + \bm{A}\bm{y}) = var(f_*) + var(\bm{A}\bm{y}) + cov(f_*, \bm{A}\bm{y}) + cov(\bm{A}\bm{y}, f_*) = var(f_*) + \bm{A}var(\bm{y})\bm{A}^T + cov(\bm{y}, f_*)\bm{A}^T + \bm{A}cov(f_*, \bm{y})
$$
$$var(f_* \mid \bm{y}) = k(\overset\sim\bm{x}_*, \overset\sim\bm{x}_*) - K(\overset\sim\bm{x}_*, \overset\sim\bm{X})K(\overset\sim\bm{X}, \overset\sim\bm{X} + \delta_{\epsilon}^2\mathcal{\bm{I}})^{-1}K(\overset\sim\bm{X}, \overset\sim\bm{x}_*).
$$

So now we already compute the predictive distribution of $f$ at a deterministic input $\bm{x}_*$. As I said before we learn conditionally indenpendent GPs for each target dimension which means these GPs are independent at a deterministic input. So we can easily extend the result we got for single dimension target to multi-dimensional situation.

We will also use above results in following derivation. In the fact, you will know this is actually a GP regression if you are familiar with this. However if you are not familiar with this, please take in mind that this is a very general result that you can use it in many other contexts.

In the next section I will show how to compute the predictive distribution of $f$ at an uncertain input $\overset\sim\bm{x}_*$.

### Approximate Inference for Long-Term Predictions
The beautiful part of model-based RL is that it can infer next state given current state and a control signal. In PILCO, we need to learn an optimal policy $\pi^*$ which minimizes the expected cost $J^\pi(\theta)=\sum_{t=0}^T\mathbb{E}_{\bm{x}_t}[c(\bm{x}_t)], \bm{x}_0 \sim \mathcal{N}(\bm\mu_0,\bm\Sigma_0)$ of following $\pi$ for $T$ steps, where $c(\bm{x}_t)$ is the cost (negative reward) of being in state $\bm{x}$ at time $t$. We assume that $\pi$ is a function parameterized by $\theta$. To write expected cost in another form, we have $J^\pi(\theta)=\int_0^Tc(\bm{x}_t)p(\bm{x}_t)d\bm{x}_t$ which means we need to compute $p(\bm{x}_t), t=1, \cdots, T$ first. To compute $p(\bm{x}_t)$, we need this
$$
p(\bm{x}_t) = \iint p(\bm{x}_t \mid \bm{x}_{t-1}, \bm{u}_{t-1})p(\bm{u}_{t-1}, \bm{x}_{t-1})d\bm{x}_{t-1}d\bm{u}_{t-1}=\iint p(\bm{x}_t \mid \bm{x}_{t-1}, \bm{u}_{t-1})p(\bm{u}_{t-1} \mid \bm{x}_{t-1})p(\bm{x}_{t-1})d\bm{x}_{t-1}d\bm{u}_{t-1}, t=1, \cdots, T.
$$
Here, comparing with the derivation we did in last section (Dynamics Model Learning), we actually need to predict distribution of $\bm{x}_t$ at an uncertain input $(\bm{x}_{t-1}, \bm{u}_{t-1})$ as when we propagate state through GP models, we propagate distribution of state. As stated in original paper, this predictive distribution is not Gaussian (check the figure in the papaer). However, we can approximate the predictive distribution by a Gaussian that possesses the same mean and variance (moment matching, this trick will be used in other parts too). From now on, I will show how to compute the mean and variance. However to make it as clear as possible I want to start with a simple case where we can go through the math easily. As we can see in the last section, the input to our GP dynamics model is a distribution over $(\bm{x}_{t-1}, \bm{u}_{t-1})$. However in a simpler case (it's easy to extend) I want to show how we can predict $p(\bm{x}_t)$ given $p(\bm{x}_{t-1})$ first. Then it would be easy to extend this case to having $\bm{u}$ which is the control signal. Moreover, let's start with univariate predictions which is to predict one dimension of $\bm{x}_t \in \mathbb{R}^D$ given an uncertain $\bm{x}_{t-1} \in \mathbb{R}^D$. We can define this problem as following, predicting a function value $f(\bm{x}_{t-1}), f: \mathbb{R}^D \rightarrow \mathbb{R}$, at an **uncertain** previous state input $\bm{x}_{t-1} \sim \mathcal{N}(\bm{\mu}, \bm{\Sigma})$, where $f \sim \mathcal{GP}$ with an $SE$ covariance function plus a noise covariance function. Let's write this predictive distribution as:
$$
p(\bm{x}_t | \mathcal{N}(\bm{x}_{t-1} \mid \bm\mu, \bm\Sigma)) = p(f(\bm{x}_{t-1})|\bm\mu, \bm\Sigma) = \int p(f(\bm{x}_{t-1}) \mid \bm{x}_{t-1})p(\bm{x}_{t-1} \mid \bm\mu, \bm\Sigma)d\bm{x}_{t-1}.
$$
Genrally, the predictive distribution in equation above cannot be computed analytically while we can compute the mean and variance of this distribution and use a Gaussian distribution possess the same mean and variance to approximate it. To compute the mean $\bm\mu_t$, we have:
$$
\bm\mu_t = \iint f(\bm{x}_{t-1})p(f, \bm{x}_{t-1})d(f, \bm{x}_{t-1}) = \iint f(\bm{x}_{t-1})p(f,\bm{x}_{t-1})dfd\bm{x}_{t-1} = \int \mathbb{E}[f(\bm{x}_{t-1}) \mid \bm{x}_{t-1}]p(\bm{x}_{t-1})d\bm{x}_{t-1} \\
= \mathbb{E}_{\bm{x}_{t-1}}[\mathbb{E}_{f}[f(\bm{x}_{t-1}) \mid \bm{x}_{t-1}] \mid \bm\mu, \bm\Sigma].
$$
Here, $\mathbb{E}_{f}[f(\bm{x}_{t-1}) \mid \bm{x}_{t-1}]$ is actually the expectation of a predictive distribution at a deterministic input which is already computed in last section (even though the dimension is not the same, we can still use the same formula). Plugging the result from the last section, we have:
$$
\mu_{t} = \mathbb{E}_{\bm{x}_{t-1}}[m_f(\bm{x}_{t-1}) \mid \bm\mu, \bm\Sigma] = \int m_f(\bm{x}_{t-1})\mathcal{N}(\bm{x}_{t-1} \mid \bm\mu, \bm\Sigma)d\bm{x}_{t-1} = \int K(\bm{x}_{t-1}, \bm{X})(K(\bm{X}, \bm{X}) + \delta_{\epsilon}^2\mathcal{\bm{I}}))^{-1}\bm{y}\mathcal{N}(\bm{x}_{t-1} \mid \bm\mu, \bm\Sigma)d\bm{x}_{t-1} \\ 
= \int K(\bm{x}_{t-1}, \bm{X})\bm{\beta}\mathcal{N}(\bm{x}_{t-1} \mid \bm\mu, \bm\Sigma)d\bm{x}_{t-1} \\
= \bm{\beta}^T \int K(\bm{x}_{t-1}, \bm{X})^T \mathcal{N}(\bm{x}_{t-1} \mid \bm\mu, \bm\Sigma)d\bm{x}_{t-1} \\
= \bm{\beta}^T\bm{q}
$$
where $\bm{\beta} := (K(\bm{X}, \bm{X}) + \delta_\epsilon^2\mathcal{\bm{I}})^{-1}\bm{y}$, $\bm{q} = [q_1, \cdots, q_n]^T \in \mathbb{R}^n$ with 
$$
q_i := \int k(\bm{x}_{t-1}, \bm{x}_i)\mathcal{N}(\bm{x}_{t-1}|\bm\mu, \bm\Sigma)d\bm{x}_{t-1} \\
= \alpha^2|\bm\Sigma\bm\Lambda^{-1} + \bm{I}|^{-\frac{1}{2}}exp(-\frac{1}{2}(\bm{x}_{i} - \bm\mu)^T(\bm\Sigma + \bm\Lambda)^{-1}(\bm{x}_i - \bm\mu)).
$$
To compute variance we need law of total variance and we have
$$
\delta_{t}^2 = var_{\bm{x}_{t-1}, f}[f(\bm{x}_{t-1}) \mid \bm\mu, \bm\Sigma] = \mathbb{E}_{\bm{x}_{t-1}}[var_f[f(\bm{x}_{t-1}) \mid \bm{x}_{t-1}] \mid \bm\mu, \bm\Sigma] + var_{\bm{x}_{t-1}}[\mathbb{E}_f[f(\bm{x}_{t-1})\mid \bm{x}_{t-1}] \mid \bm\mu, \bm\Sigma] \\
= \mathbb{E}_{\bm{x}_{t-1}}[\delta_f^2(\bm{x}_{t-1}) \mid \bm\mu, \bm\Sigma] + (\mathbb{E}_{\bm{x}_{t-1}}[m_f(\bm{x}_{t-1})^2 \mid \bm\mu, \bm\Sigma] - \mathbb{E}_{\bm{x}_{t-1}}[m_f(\bm{x}_{t-1}) \mid \bm\mu, \bm\Sigma]^2).
$$
Using the result from last section for $\delta_f$ and $m_{f}$, we have
$$
\delta_t^2 = \int k(\bm{x}_{t-1}, \bm{x}_{t-1}) - K(\bm{x}_{t-1}, \bm{X})(K(\bm{X}, \bm{X}) + \delta_\epsilon^2\mathcal{\bm{I}})^{-1}K(\bm{X}, \bm{x}_{t-1})p(\bm{x}_{t-1})d\bm{x}_{t-1} + \int K(\bm{x}_{t-1}, \bm{X})\bm\beta\bm\beta^TK(\bm{X}, \bm{x}_{t-1})p(\bm{x}_{t-1})d\bm{x}_{t-1} - (\bm{\beta}^T\bm{q})^2.
$$
Equation above can be computed analytically. 

Awesome! Now we know how to compute univariate predictive distribution on uncertain input. To compute multivariate predictive distribution, we assume the target dimension is $E$. The predictive mean vector $\bm{\mu}_{t}$ is the collection of all independently predicted means computed in the univariate predictions case,
$$
\bm\mu_{t} \mid \bm{\mu}, \bm{\Sigma} = [\bm\beta_1^T\bm{q}_1, \cdots, \bm\beta_E^T\bm{q}_E]^T.
$$
However unlike predicting at deterministic inputs where all the target dimensions are independent, the target dimensions now covary, so the corresponding predictive covariance matrix
$$
\bm{\Sigma}_t \mid \bm{\mu}, \bm{\Sigma} = \begin{bmatrix}var_{f, \bm{x}_{t-1}}[f_1(\bm{x}_{t-1}) \mid \bm{\mu}, \bm{\Sigma}] & \cdots & cov_{f, \bm{x}_{t-1}}[f_1(\bm{x}_{t-1}), f_E(\bm{x}_{t-1}) \mid \bm\mu, \bm\Sigma] \\
\vdots & \ddots & \vdots \\
cov_{f, \bm{x}_{t-1}}[f_E(\bm{x}_{t-1}), f_1(\bm{x}_{t-1}) \mid \bm\mu, \bm\Sigma] & \cdots &  var_{f, \bm{x}_{t-1}}[f_E(\bm{x}_{t-1}) \mid \bm{\mu}, \bm{\Sigma}] \end{bmatrix}
$$
is no longer diagonal. The cross-covariances are given by
$$
cov_{f, \bm{x}_{t-1}}[f_a(\bm{x}_{t-1}), f_b(\bm{x}_{t-1}) \mid \bm\mu, \bm\Sigma] = \mathbb{E}_{f, \bm{x}_{t-1}}[f_a(\bm{x}_{t-1})f_b({\bm{x}_{t-1}}) \mid \bm{\mu}, \bm{\Sigma}] - \bm\mu_{a}(\bm{x}_{t-1})\bm\mu_{b}(\bm{x}_{t-1}). 
$$
The equation above can also be analytically computed. So that we can get the exact mean $\bm\mu_{t}$ and the exact covariance $\bm\Sigma_{t}$ of the generally non-Gaussian predictive distribution $p(f(\bm{x}_{t-1}) \mid \bm\mu, \bm\Sigma)$.

To extend this result to the input including control signal, in other words, predicting a function value $f(\bm{x}_{t-1}, \bm\mu_{t-1}), f: \mathbb{R}^{D+F} \rightarrow \mathbb{R}$, at an **uncertain** previous state input $\bm{x}_{t-1} \sim \mathcal{N}(\bm{\mu}_{\bm{x}}, \bm{\Sigma}_{\bm{x}})$, $\bm{u}_{t-1} \sim \mathcal{N}(\bm{\mu}_{\bm\mu}, \bm\Sigma_{\bm\mu})$ where $f \sim \mathcal{GP}$ with an $SE$ covariance function plus a noise covariance function, we need to follow following 3 substeps:
- A Gaussian distribution $p(\bm{u}_{t-1}) = p(\pi(\bm{x}_{t-1}))$ over actions is computed when mapping $p(\bm{x}_{t-1})$ through the policy $\pi$.
- A joint Gaussian distribution $p(\bm{x}_{t-1}, \bm{u}_{t-1}) = p(\bm{x}_{t-1}, \pi(\bm{x}_{t-1}))$ is computed.
- The distribution $p(\bm{x}_t)$ is computed by applying the results we just got in the simple case.

Now let's consider about computation of a distribution over actions $p(\bm\mu_{t-1})$. First of all, the policy itself should fullfill following two properties:
- For a state distribution $p(\bm{x})$ we need to be able to compute a corresponding distribution over actions $p(\bm{u}) = p(\pi(\bm{x}))$.
- In a realistic application, the policy must be able to deal with constrained control signals.

During the forward simulation, the states are given by probability distribution $p(\bm{x}_t), t = 0, \cdots, T$. The probability distribution of the state $\bm{x}_t$ induces a predictive distribution over actions, even if the policy is deterministic. To get constrained control signals, we assume the control limits are such that $\bm{u} \in [-\bm{u}_{max}, \bm{u}_{max}]$. Let us consider a preliminary policy $\overset\sim{\pi}$ with an unconstrained amplitude. To model the control limits coherently during simulation, we squash that preliminary policy $\overset\sim{\pi}$ through a bounded and differentiable squashing function, e.g. sine, logistic, cumulative Gaussian. However there are sevel advantages for using sine function. One advantageous property of the sine is that it allows for an analytic computation of the mean and the covariance of $p({\pi}(\bm{x}))$ where $\pi(\bm{x}) = \mu_{max}sin(\overset\sim\pi(\bm{x}))$ if $\overset\sim{\pi}(\bm{x})$ is Gaussian distributed so that we can use this mean and covariance to approximate the predictive distribution of $\pi(\bm{x})$ with a Gaussian distribution.

Following, I will take radial basis function (RBF) network as one possible representation of the preliminary policy $\overset\sim\pi$ that allow for a closed-form computation of mean and the covariance of $p(\overset\sim{\pi}(\bm{x}))$ when the state $\bm{x}$ is Gaussian distributed.

The preliminary RBF policy is given by
$$
\overset\sim{\pi}(\bm{x}_{*}) = \Sigma_{s=1}^N\beta_sk_\pi(\bm{x}_s, \bm{x}_*)=\bm\beta_\pi^Tk_\pi(\bm{X}_\pi,\bm{x}_*),
$$
where $\bm{x}_*$ is a test input, $k_\pi$ is a kernel function, and $\bm\beta_\pi:=(K_\pi+\delta_\pi^2\mathcal{\bm{I}})^{-1}\bm{y}_\pi$, $\bm{X}_\pi$ and $\bm{y}_\pi$ are training data. We can see this RBF network as a deterministic GP, which means $var_{\pi}=0$, and it will always output the mean value. Then we can use the results we got before to compute the mean and variance of this predictive distribution and use a Gaussian distribution possess the same mean and variance to approximate this predictive distribution. Then we need to squash the preliminary policy through the sine and compute an approximate Gaussian distribution of $p(\bm{u}_{max}sin(\overset\sim{\pi}(\bm{x}_{t-1})))$ which can be also computed analytically. Then we will need to compute the joint distribution $p(\bm{x}_{t-1}, \bm{u}_{t-1})=p(\bm{x}_{t-1}, \pi(\bm{x}_{t-1}))$. It is also computed in two steps. First, we compute the distribution $p(\bm{x}_{t-1}, \overset\sim{\pi}(\bm{x}_{t-1}))$. Then we compute an approximate fully joint Gaussian distribution $p(\bm{x}_{t-1}, \overset\sim\pi(\bm{x}_{t-1}), \bm{u}_{max}sin(\overset\sim\pi(\bm{x}_{t-1})))$ and marginalize $\overset\sim{\pi}(\bm{x}_{t-1})$ out to obtain the desired joint distribution $p(\bm{x}_{t-1}, \bm{u}_{t-1})$.

Use above results we can compute the predictive state distribution $p(\bm{x}_{t}), t=1, \cdots, T$ iteratively. Then we can also compute the expective cumulative cost of following these $\bm{T}$ steps.

### Policy Learning
Here we need to use a gradient-based policy search method to optimize the policy parameters. This means, we aim to find a parameterized policy $\pi^*$ from a class of policies $\Pi$ with
$$
\pi^* \in arg min_{\pi \in \Pi}J^{\pi_\psi}(x_0).
$$

**(continuing...)**

## REFERENCES
- [PILCO: A Model-Based and Data-Efficient Approach to Policy Search](https://spiral.imperial.ac.uk/bitstream/10044/1/11585/4/icml2011_final.pdf)
- [Efficient Reinforcement Learning using Gaussian Processes](https://pdfs.semanticscholar.org/c9f2/1b84149991f4d547b3f0f625f710750ad8d9.pdf)

