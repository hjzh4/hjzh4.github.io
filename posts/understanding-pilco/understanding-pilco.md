# Understanding PILCO
[Email](mailto:hjzh578@gmail.com) me if you have any question.
## Introduction
PILCO is a famous model-based reinforcement learning algorithm, which leverages gaussian process to model robotic system. Model-based reinforcement learning algorithms are considered to be data-efficient as they optimize the controller policy via the prediction of consquential states following current policy, instead of exploiting data which only comes from interacting with environment. However, a big issue in model-based reinforcement learning is that humans actually can't get a "right" model of a dynamic system (think about Laplace's Daemon), owing to uncertainty introduced by noisy measurements,  unobservable internal properties of the dynamic system,  etc. in which case the model bias is inevitably introduced when people assume that the learned dynamics model sufficiently accurately resembles the real environment. To capture the uncertainty of models, PILCO uses Gaussian Process to model the dynamic system. Moreover, Gaussian Process can also take as input a distribution and output another distribution, so that we can keep uncertainty of models, prediction through the whole algorithm pipeline.

Gaussian Process is a very powerful machine learning tool, while it's also way less popular than neural networks in current robotics community, because it's way more analytically harder than neural networks while mathmatically easier. Owing to this, PILCO is also very hard to understand for a beginner like me. In this blog, I will derivate the whole framework of PILCO for my own learning purpose. I also hope that it can help other beginners like me to understand PILCO easier.

In this blog, I just assume you have a basic linear algebra and calculus and reinforcement learning background.

## Gaussian Process
Understanding Gaussian Process is the basis for understanding PILCO. Here are some resources that introduce me to this field.
- [Gaussian Process for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
- [Gaussian Process Summer School 2018](http://gpss.cc/gpss18/day-1.html)
- [Machine Learning Summer School 2012](https://www.youtube.com/watch?v=BS4Wd5rwNwE)

## PILCO
Let's start from pseudo code.

**init**: Sample controller parameters $\theta \sim \mathcal{N}(\bm{0}, \mathcal{I})$. \
**repeat** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Apply random control signals and record data. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Learn Gaussian Process dynamics model using all data. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**repeat** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Simulate system with current controller $\pi(\theta)$ and get expected cost $J^\pi(\theta)$. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Improve controller paramters to $\theta^*$. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**until** convergence \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**return** $\theta^*$ \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set controller parameters to $\theta^*$. \
**until** task learned

From the pseudo code, we can see there are three key components of PILCO:
- Dynamics Model Learning
- Policy Evaluation
- Policy Improvement

### Dynamics Model Learning


