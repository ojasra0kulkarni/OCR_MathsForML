# Transcribed Math Notes
Source: Complete_Notes_Maths_For_ML.pdf

---

## Page 1

## Page 2

Machine Learning
probabilistic view point.

$X$: domain set (Input)
$Y$: Range set (output)

Given pairs $D = \{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}$
where $x_i, y_i$ is a pair of observation.

Assume that $\exists$ a function $f: X \rightarrow Y$
$f(x_i) = y_i$

The underlying function "$f$" is unknown.
Given $D$, find $f$

Start with some initial guess on $f$.
"refine" the guess using $D$ (data).

---

Consider $X_i$ to be x-ray images.
0-255
q
p
1st col
2nd col
...
p$^{th}$ col
pq-sized vector.
$\mathbb{R}^{PQ}$ - a point in $\mathbb{R}^{PQ}$ dim.

---

## Page 3

$x_i \in X \subset \mathbb{R}^d$
$y_i \in \{0, 1\}$, $0 \to$ diseased
$1 \to$ benign /non-diseased
$f: X \to Y$

"f" is complex to be estimated from
the physics of the problem.

Resort to statistical methods:
make repeated observations & estimate f.

In a probabilistic frame work
$X, Y$ define Random Variables.

---

$\Omega$: sample space [set of all possible
outcomes of a random
Experiment]

Consider the subsets of sample space.
Let $\mathbb{F}$ denote the collection of all possible subsets.
Objective: Assign a "measure" on $\mathbb{F}$.

---

## Page 4

Probability measure is one such measure.

$P : \mathcal{F} \to [0, 1]$

properties of $P$
if $A, B \in \mathcal{F}$.
i) $P(A) \ge 0 \quad \forall A \in \mathcal{F}$.
ii) $P(\Omega) = 1, P(\phi) = 0$
iii) $A, B \text{ s.t. } A \cap B = \phi, P(A \cup B) = P(A) + P(B)$
$(\Omega, \mathcal{F}, P) : \text{Prob. Triplet.}$
---
Define a function $X$ (random variable)
from $\Omega$ to $\mathbb{R}$.

$X: \Omega \to \mathbb{R}$

$\Omega \to \mathbb{R}$
$\mathcal{F} \to \mathcal{B}\text{-sigma algebra}, (-\infty, x]$
$P \to P_x : \text{distribution function}$

$P_x(x) \triangleq P \left[ A : X^{-1}(-\infty, x] \right]$

$(\Omega, \mathcal{F}, P) \xrightarrow{\text{R.v. } X} (\mathbb{R}, \mathcal{B}, P_x) : \text{work with this.}$

---

## Page 5

Random variables with vector-valued range spaces.

In general, R.V. have $ \mathbb{R}^d $ as their range spaces.

$ X: \Omega \longrightarrow \mathbb{R}^d $ (vector-valued R.V.).

$ P_{\underline{X}} (X \in \mathbb{R}^d) = $ probability of the inv. image of cartisian product under X.

**Vector valued RVs.**

Here the range set of the function (RV), is $ \mathbb{R}^d $ where $ d $ is a scalar.

$ X: \Omega \longrightarrow \mathbb{R}^d $

**Joint distributions :**

Let $ \Omega $ be a sample space.

Define two functions, $ x_1 $ & $ x_2 $

$ x_1 : \Omega \longrightarrow \mathbb{R} $, $ P_{x_1} $
$ x_2 : \Omega \longrightarrow \mathbb{R} $, $ P_{x_2} $

---

## Page 6

Define Joint probability distribution as
$P_{x_1 x_2} (a, b) = P[E : \text{intersection of inverse images of } (-\infty, a] \text{ & } (-\infty, b] \text{ under } x_1 \text{ & } x_2, \text{ respectively}]$

The above idea can be extended to
d scalar Random variables.

Define conditional probabilities (distributions)
If A & B $\subset F$,
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

Suppose X & Y are two RVs defined on $\Omega$,
$$P_{x|y} (x | y=y) = \frac{P_{xy}}{P_y}$$

---

## Page 7

Marginal distribution : If $x$ & $y$ are
two RVs, marginal of $x$ is defined as
$$P_x = \int_{\text{range}(y)} P_{xy} dy$$
$$P_y = \int_{\text{range}(x)} P_{xy} dx$$

---

Example:

Consider an image of dimensions $p \times q$.
(Diagram: A rectangle with width $p$ and height $q$, with a small square in the top-left corner.)
$q$ Can be viewed as a $R^{\text{Pq}}$-dim vector.
$p$ vector.

In general, any datapoint is a d-dim vector

Every datapoint is an element in the
range space of a random variable. $X$

The distribution function indicates / quatifies
the likelihood of observing a datapoint under $X$.

---

## Page 8

Distribution function completely specifies the sample space.

Typically, a “label” is taken to be another additional random variable defined on the same sample space.

In this scenario, we have two RVs,
$X$ & $Y$, $X \in \mathbb{R}^d$ & $Y \in \mathbb{R}^k$
$X$: dataspace $Y$: label space $\{1,2,3,4,5\}$

The convention is as follows:
“Learning” starts with a dataset

$$D = \left\{ (x_1, y_1), (x_2, y_2), (x_3, y_3), \dots, (x_N, y_N) \right\}$$
$\sim i.i.d P_{xy}$
Indepent
Identically distributed.

Independent accross data points & not data-dimensions.

---

## Page 9

All problems in Machine learning
Can be seen as
Given $D$ an unknown $P_x$, Estimate $P_x$
& sample from it or conditionals on $P_x$.

Recall.

Given a dataset
$$D = \{ (x_i, y_i) \}_{i=1}^N \sim \text{iid } P_{xy} (\text{unknown})$$
$x_i \in \mathbb{R}^d$, $y_i \in \mathbb{R}^k$ or $\{1, 2, ..., K\}$

Typically,
X: input /features /data
Y: label /output.

X & Y are random variables defined on a sample space.

Fundamental problem of ML:
Given $D$ an unknown distribution,
a) Estimate the distribution
b) Sample from the distribution

---

## Page 10

## Distribution Estimation
Given $D = \{(x_i, y_i)\} \sim \text{iid } P_{xy}$

Examples :
i) Estimate $P_{y|x}$ : Classification, $y \in \{1,2,...,K\}$
   $P_{y|x}$ : Regression, $y \in \mathbb{R}^K$
ii) Estimate $P_x, P_y \text{ or } P_{xy}, P_{x|y}$

In all conditionals, $P_{x|y}(x|y=y)$

## Probability density function:
Give a CRV with a dist function $P_x$,
density function $p_x : X \rightarrow \mathbb{R}^+$, s.t
$$P_x(x) = \int_{-\infty}^{x} p_x(x) dx$$

## The challenge with ML
> Given $D \sim \text{iid}$ from unknown $P_x$, Estimate $P_x$.

Challenge : $P_x$ is completely unknown.

---

## Page 11

Question : How to estimate a density function
given its samples?

Consider we have a dataset $\mathcal{D}$.
$\mathcal{D} = \{x_i\}_{i=1}^N \sim \text{iid } P_x$

i) Assume a parametric functional form on $P_x$,
denoted by $P_\theta$. $x \in \mathbb{R}^d$ "Model Choice"

Eg: $P_\theta^a(x) = W_1^T x + W_2$, $W_1, W_2 \in \mathbb{R}^d$
$\theta = \{W_1, W_2\}$

$$
P_\theta^b(x) = \mathcal{N}(x; \mu, \Sigma)
$$
$\theta = \{\mu, \Sigma\} \quad , \mu \in \mathbb{R}^d, \Sigma \in \mathbb{R}^{d \times d}$

$\Theta$: parameters.

ii) Define or $\underline{\text{compute}}$ a distance metric
between $P_x$ and $P_\theta$.

Let $d$ denote the "distance metric"
blw $P_x$ and $P_\theta$.

$d(P_x, P_\theta) : P_x \times P_\theta \rightarrow \mathbb{R}^+$

---

## Page 12

iii) Find | Estimate the parameters $\theta$
by solving the following optimization problem

$$\theta^* = \underset{\theta}{\operatorname{argmin}} d(P_x, P_\theta)$$
Training the model

---

## Page 13

<u>Decision Trees</u>

Non-parametric Methods
The core-idea is to split the data space
into regions via binary questions on data
dimensions.

Suppose $D = \{(x_i, y_i)\} \sim \text{iid } P_{xy}$
$x \in \mathbb{R}^d \quad y \in \{1, 2, ..., k\}$

At any given node $m$, which represents
a Region $R_m \subset \mathbb{R}^d$, containing $N_m$ samples,
the tree-algorithm asks a binary question
as follows:

Suppose $x^j$ denote the $j^{th}$ dimension of $X$.
& $t$ be a scalar threshold.

Split:
If $x^j \le t$ , go left
$x^j > t$ , go right

The optimization objective for splitting
At every node, search over every data dim.
& threshold $(j, t)$, that minimizes the node
"impurity".

---

## Page 14

Let $I(R)$ be the impurity function.

$$L(j,t) = \frac{N_L}{N_m} I(R_L) + \frac{N_r}{N_m} I(R_r)$$

where $N_L, N_r$ are number of samples that traverse the left & Right regions $R_L \& R_r$, respectively.
---
Example Impurity functions.

i) Entropy or Information gain.

Suppose $P_k$ is the proportion of training samples from Class - K in region $R_m$.

Entropy of a Node
$$H(R_m) = - \sum_{k=1}^{K} P_k \log_2 (P_k)$$

$$\hat{I}(j,t) = H(R_{parent}) - \left[ \frac{N_L}{N_m} H(R_L) + H(R_r) \frac{N_r}{N_m} \right]$$

Choose a split that minimize $-\hat{I}(j,t)$.

---

## Page 15

b) GINI Impurity:

measures the probability of incorrectly
classifying a randomly chosen element
if it were randomly labelled.

prob. of Choosing a datapoint from
Class K is $P_K$.
prob. of incorrectly labelling is $(1-P_K)$

$$Gini(R_m) = \sum_{k=1}^{K} P_k (1-P_k)$$

Choose the split (j,t) that minimizes
the Gini impurity.
---
<u>Regression Trees.</u>

If $Y \in \mathbb{R}$ (Continuous), discrete
Class prob. based metrics Can't be used.

If a sample is falling in region
$R_m$, define the prediction within $R_m$
as
$$\hat{y}_m = \frac{1}{N_m} \sum_{i \in R_m} y_i$$

---

## Page 16

Impurity metric for the Regression Case

$$I(R_m) = \frac{1}{N_m} \sum_{i \in R_m} (y_i - \hat{y}_m)^2$$

i) How to choose candidate thresholds for a split ?

For a node $m$ containing $N_m$ samples, extract all the $N_m$ values for the feature $x^j$,
sort the values, remove duplicates,
rank-order them $u_1 < u_2 < u_3 ... u_p$
$$p \le N_m$$
Define $t_i$ for $i=1...p-1$
$$t_i = \frac{u_i + u_{i+1}}{2}$$

ii) Stopping Criteria | pruning.
a) maximum Depth ,
Define $D_{max}$ as the maximum possible depth (h param).

---

## Page 17

if $D$ is the depth of a given node,
algo stops if $D = D_{max}$.

b) minimum samples for a split
let $N_m$ be the no of samples in
the current node, the algo. will
split further only if
$N_m \ge N_{min}$.

c) minimum Impurity Decrease
split only if $I(j, t) > \epsilon$.

---

## Page 18

## Ensemble Methods

$$
\bar{h}(x) = \underset{\mathbb{D}}{\mathbb{E}}[h(x)]
$$

In Ensemble methods, $ \underset{\mathbb{D}}{\mathbb{E}} $ is approximated via creating datasets by sampling with replacement.

Bagging: Sample with replacement to create an Ensemble.

Suppose the dataset $D$ contain $N$ samples. Create $B$ training sets by sampling so that each new dataset $D_b$ has $N$ samples.

prob. of a sample being picked $1/N$
prob. of $x_i \notin D_b = (1 - 1/N)^N$
$$
\lim_{N \to \infty} (1 - 1/N)^N = \frac{1}{e} = 0.36
$$
we train $B$ i/d classifiers
$h_1(x), h_2(x)... h_B(x)$

---

## Page 19

The final prediction:
$$
\hat{y}_{Bag} = \frac{1}{B} \sum_{b=1}^{B} h_b(x)
$$

---

The Variance
-------------

Suppose $Z_i$ represent a Rv, corresponding
to the prediction of the $i^{th}$ hypothesis on
a given datapoint.

Since all the hypo. are trained on
the data sampled from a given distribution
we assume $Z_1 \ldots Z_B$ to be identically
distributed.

$\therefore Var(Z_i) = \sigma^2$ for all $i$.

since $Z_i \text{ & } Z_j$ are not stat. independent
$corr(Z_i, Z_j) = \rho \quad \forall i \ne j$

$corr() = \frac{Covariance}{\sigma_i \sigma_j}$

---

## Page 26

Given a data point
$x = \{x_1, x_2, ..., x_T\}$, $x_i \in \mathbb{R}^d$ (token)

<u>Objective</u> : Learn a projection / representation
that encodes the interaction b/w the
tokens.

Let $X$ be a matrix, having the data
$X \in \mathbb{R}^{T \times d}$

Define linear projections of $X$ using three
matrices.
$Q = X W_q$ ($X$ dimensions: $T \times d$, $W_q$ dimensions: $d \times d_q$) where $W_q, W_v$ & $W_k$
$K = X W_k$ ($X$ dimensions: $T \times d$, $W_k$ dimensions: $d \times d_k$)
$V = X W_v$ ($X$ dimensions: $T \times d$, $W_v$ dimensions: $d \times d_v$)
are learnable.
Assume $d_k = d_v = d_q$

<u>Defining the Attention vector.</u>
Given a row of $Q$, denoted by $q \in \mathbb{R}^{d_v}$
compute the following to get the "attention".

---

## Page 20

$$\Rightarrow \text{cov}(z_i, z_j) = \rho \cdot \sigma^2$$

To calculate the var. of Ensemble

$$\bar{z} = \frac{1}{B} \sum_{i=1}^{B} z_i$$

$$\text{var}(\bar{z}) = \text{var}\left(\frac{1}{B} \sum_{i=1}^{B} z_i\right)$$

$$= \frac{1}{B^2} \text{var}\left(\sum_{i=1}^{B} z_i\right)$$

$$\text{var}\left(\sum_{i=1}^{B} z_i\right) = \sum_{i=1}^{B} \text{var}(z_i) + \sum_{i \neq j} \text{cov}(z_i, z_j)$$

$$\Rightarrow \text{var}(\bar{z}) = \rho \sigma^2 + \left(\frac{1-\rho}{B}\right) \sigma^2$$

To reduce var($\bar{z}$)
i) Increase B
ii) reduce $\rho$ : ensure that individual
hypo. are uncorrelated

---

## Page 21

Eg: Random Forests:

Ensemble of Decision trees
a randomly sampled subset of data dimensions are chosen at each node of every individual tree to reduce "p".

---

<u>Boosting algorithms.</u>

The Ensemble is built, iteratively in an incremental manner.

Let $h_m(x)$ denote the Classifier learned at the $m^{th}$ iteration.
for $m=1$ to $B$
$$H_m(x) = H_{m-1}(x) + \alpha_m h_m(x).$$
where $H_m(x)$ is the incremental Ensemble obtained at the $m^{th}$ iteration.

<u>Question</u>: Given $H_{m-1}(x)$, how to train $h_m(x)$ to that loss is reduced?

---

## Page 22

Gradient Boosting mechanism
___

Recall,

$$H_m(x) = H_{m-1}(x) + \alpha_m h_m(x)$$

Goal: Find a new learner $h_m(x)$ to minimize the risk.

$$\underset{h, \alpha}{\operatorname{argmin}} \sum_{i=1}^{N} L(y_i, H_m(x_i))$$

Consider

$$L(y_i, H_m(x_i)) = L(y_i, H_{m-1}(x_i) + \alpha_m h_m)$$

$$L(y_i, H_{m-1}(x_i) + \alpha_m h_m) \approx L(y_i, H_{m-1}(x_i)) + \left[ \frac{\partial L(y_i, H(x_i))}{\partial H(x_i)} \right]_{H=H_{m-1}} \cdot \alpha h(x_i)$$
$\uparrow$ Taylor's series around $H_{m-1}$. $g_i$

---

## Page 23

$$ \text{LHS} = L (y_i, H_{m-1}(x_i)) + \alpha g_i h(x_i) $$

$$ \underset{\alpha,h}{\operatorname{argmin}} \sum_{i=1}^{N} \left[ L (y_i, H_{m-1}(x_i)) + \alpha g_i h(x_i) \right] $$
Ild of h&α

$$ = \underset{\alpha,h}{\operatorname{argmin}} \sum_{i=1}^{N} \alpha \cdot g_i h(x_i) $$

$g_i \vert_{H=H_{m-1}}$ can be computed.

question: How to minimize $\sum g_i h(x_i)$

note that $\sum g_i h(x_i) = \langle g, h \rangle$

$\Rightarrow \underset{h}{\operatorname{argmin}} \langle g, h \rangle = h^* \mid h^* \alpha - g_i$

To find $h^*$ that is proportional to
$-g_i$: solve a regression problem with
$-g_i$ as labels.
$h_m = \underset{h}{\operatorname{argmin}} \sum_{i=1}^{N} [-g_i - h(x_i)]^2$

---

## Page 24

The Gradient Boosting Algorithm

Given a dataset D,

start by creating B datasets (sampling with replacement)

for m=1 to B-1 do:
$$H_m(x) = H_{m-1}(x) + \alpha h_m(x)$$
compute
$$-g_i^m = -\left[ \frac{\partial L(y_i, H(x_i))}{\partial H(x_i)} \right]_{H=H_{m-1}}$$
$\rightarrow$ residual

Train $h_m(x)$ by regressing over $g^m$
get
$$H_m(x) = H_{m-1}(x) + \alpha h_m(x)$$
End For.

---

## Page 25

# Attention Mechanism & Transformers

$X \xrightarrow{\phi} Z$
Representations | Embeddings.

$h(x) = \sigma(W_2 \sigma(W_1 x))$
$\underbrace{\qquad}_{\phi(x)}$

Attention :

*(Diagram depicting an encoder-decoder attention mechanism)*
Inputs: $x_1, x_2, \ldots, x_T$
Encoder states: $h_1, h_2, \ldots, h_T$
Attention weights: $\alpha_1, \alpha_2, \ldots$ leading to $\bigoplus$ (summation/concatenation of weighted contexts)
Decoder outputs: $y_1, y_2, \ldots, y_{T'}$

$$y_t = \mathcal{F}\left(y_{t-1}, \sum_{t=1}^{\tau} \alpha_t h_t\right)$$

---

## Page 27

(i) $S_i = q K_i^T \quad \forall K_i \in \mathbb{K}, i=1 \dots T$

Let $S = q K^T$
  $\text{1xdv dvxT}$
a T-length vector,
quantifying the similarities b/w $q$ & all rows of K.

$$ \hat{S} = \frac{S}{\sqrt{d_v}} $$

$$ \alpha = \text{softmax}(\hat{S}) $$

$\alpha$ represents the scaled similarities b/w $q$ & all the elements of K.

Define attention vector $A_q$ for $q$ as
$$ A_q = \sum_{i=1}^{T} \alpha_i V_i $$

Q: query K: key V: value

$$ \text{Attention}(Q, K, V) = \left[ \text{softmax} \left( \frac{q K^T}{\sqrt{d_v}} \right) \right] V $$

$X_{\text{Txd}}$ $\xrightarrow{\text{Attention}}$ $Z_{T \times d_v}$
$\phi(x)$

---

## Page 28

Multi-head Attention
--------------------

Learn multiple attention projections
on $X$. Call them $Z_1, Z_2... Z_M$

where $Z_j : Attn(X)$

Define

$$ MHA(x) = [Z_1 Z_2 ... Z_M] W_z $$
Tx dv
Tx (m x dv)

---

$h_o(x): X \rightarrow Y$
regularize $(\theta)$.

$h_o(x) = W_2 \sigma (W_1 x) \quad \theta=\{W_1, W_2\}$
$= W_2 z. \quad z=\sigma(W_1 x)$

$h_o(x)$ can be viewed as a fn.
of $\theta$ & $z$

---

## Page 29

Normalizations are used to regulate
z.

i) Batch normalization
ii) layer normilization.

$$
\hat{z} \leftarrow \left( \frac{z-\mu_z}{\sigma_z} \right) v + \beta
$$

$$
\mu_z = \frac{1}{B} \sum z_i
$$

$$
\sigma_z = \text{var} (z_i)
$$

(Diagrammatic representation of flow)
x
|
(norm)
|
$z_1$
|
(norm)
|
$z_2$
|
$z_3$

---

## Page 30

Transformer Architecture.

$\uparrow$
linear + softmax
$\uparrow$
Fully Connect layer
$\uparrow$
Residual + norm
$\uparrow$
Multi head attn.
$\uparrow$
x

[The structure from "Multi head attn." up to "linear + softmax" is labeled as] Transformer

---

## Page 31

Consider a sequence
$$
x = \{x_1, x_2 \dots x_T\}
$$

To incorporate the ordinality of the input
sequence, another fixed "temporal variable"
needs to be added.

$$
t = \{0, 1, 2, \dots 3\}
$$

$$
\hat{x}_i = x_i + t_i
$$
scalar $t$ gets ignored & a fixed
vector of dim $d$ needs to be consider.

$$
f : t \rightarrow \hat{t} , \mathbb{R} \rightarrow \mathbb{R}^d
$$
$\hookrightarrow$ positional Embedding.

Eg: sinusoidal Embedding
```
|
d|~~~~~~~
 |  ~~~~~~~
 | ~~~~~~~
 | ~~~~~~~~~~
0|____________t=0____t=1________________t=T.
```

---

## Page 32

$ \hat{t}\left(i, 2t\right) = \sin \left( \frac{i}{N^{2t/d}} \right) $

$ \hat{t}\left(i, 2t+1\right) = \cos \left( \right) $

$ \hat{x}_i = x_i + \hat{t}_i $

---

Transfer Learning

Given $x \in \mathbb{R}^d$, $y \in \mathbb{R}$,

suppose $g_\phi (x): X \to Y$

$g_\phi(\cdot)$ is a NN, learned via ERM
$\hookrightarrow$ "pre-trained NN"

since $g_\phi(\cdot)$ is a Composite function,

the outputs Can be tapped from any layer

$Z = g_\phi(x)|_{\text{layer}=l}$, Embedding of $X$
from $g_\phi$

use Z for any further down-stream tasks.

---

## Page 33

The following diagram illustrates a multi-task learning setup:

$x \rightarrow g_\phi \rightarrow z$
$z \rightarrow f_\theta(z) \rightarrow \text{Class}$
Below $f_\theta(z)$ is ERM.
$z \rightarrow h_u(z) \rightarrow \text{regn}$
A curly brace under $g_\phi$ is labeled $L$.

multi-task learning

---

Knowledge distillation

Suppose $x, y \sim P_{xy}$.
$g_\phi()$ is trained on a task.
$\hookrightarrow$ teacher Network
Let $f_0()$ be another NN (typically
$\hookrightarrow$ student network
with lesser parameters than $g_\phi$)

Question: How to get the performance
of the student network to match that
of the teacher?

---

## Page 34

$g\phi(\cdot)$
$z_t = g\phi | L$

$x \rightarrow f_\theta(\cdot) \rightarrow y$
$z_s = f_\theta(\cdot) | L$

Ensure that $\dim(z_t) = \dim(z_s)$

Regularize the student network s.t.
$D_V (P_{z_t} || P_{z_s})$ is minimized.

$$
\hat{R}_{\text{student}}(\theta) = \underset{\theta}{\operatorname{argmin}} \left[D(P_{z_t} || P_{z_s}) + \frac{1}{N} \sum \ell(f_\theta(\cdot), y)\right]
$$

---

Training NNs with Adaptive learning rate

NNs are trained with ERM, via
gradient descent.

Suppose $\hat{R}(\theta)$ is the ER, grad descent
$\theta^{t+1} \leftarrow \theta^t - \alpha \nabla_\theta \hat{R}(\theta)$
$\hookrightarrow$ fixed learning rate

---

## Page 35

<u>Stochastic gradient descent</u>
$$
\hat{R}(\theta) = \frac{1}{B} \sum_B l(h_\theta(x), y)
$$
where $B \subset D$, randomly sampled.

<u>Adaptive learning rate for grad-des</u>

i) SGD with Momentum
Define $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
where $g_t = \nabla_\theta \hat{R}(\theta)$
$$
\theta^{t+1} \leftarrow \theta_t - \alpha m_t
$$

2) <u>RMS prop</u> :
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$
$$
\theta^{t+1} \leftarrow \theta_t - \frac{\alpha}{\sqrt{v_t}} \odot g_t
$$

---

## Page 36

3) Adaptive momentum Estimation (Adam)

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\theta^{t+1} \leftarrow \theta_t - \frac{\alpha}{\sqrt{v_t}} \cdot m_t
$$

---

## Page 37

<u>Max- Margin classifiers or Support vector</u>
<u>Machines</u>.

Consider a Binary classification problem.
$x \in \mathbb{R}^d$ & $y \in \{-1, 1\}$

<u>Discriminant functions</u>:

Recall that, a binary classifier can
be represented as,
$$ h(x) = \begin{cases} 1 & g(x)>0 \\ -1 & \text{otherwise.} \end{cases} $$
where $g(x) = \text{function of } x.$
$g(x): \mathbb{R}^d \rightarrow \mathbb{R}$

$g(x)$: discriminant function.

For some specific cases of class-cond.
densities, $g(x)$ will be linear.

---

## Page 38

Example:
$x \in \mathbb{R}^2$, $y \in \{-1, 1\}$
$x_i : \{x_i^1, x_i^2\}$
$W^T x + b = 1$
$W^T x + b = -1$
$x^2$
$x : y=1$
$g(x) = W^T x + b$
$2\epsilon$
$X : y=-1$

Linear Separability: A Dataset D is
linearly separable if $\exists$ W & b
s.t
$W^T x_i + b > 0 \quad \forall x_i : y_i = 1$
$W^T x_i + b < 0 \quad \forall x_i : y_i = -1$

Linearly separable data will have
infinitely many discriminant functions.

Question: Is there an "optimal"
discriminant fn. among all these?

---

## Page 39

Optimality via maximizing the margin.

Consider lin. separable data.

since it is lin. separable, $\exists \epsilon > 0$ s.t
$W^T x_i + b \geq \epsilon \quad \forall x_i : y_i = 1$
$W^T x_i + b < -\epsilon \quad \forall x_i : y_i = -1$

Now, W & b can be rescaled, such that
$\hat{W}^T x_i + \hat{b} \geq 1 \quad \forall x_i : y_i = 1$
$\hat{W}^T x_i + \hat{b} \leq -1 \quad \forall x_i : y_i = -1$

The above can be combined as
$y_i (W^T x_i + b) \geq 1 \quad \forall i.$

For lin. sep. Case, $\exists$ W & b s.t
$y_i (W^T x_i + b) \geq 1 \quad \forall i$

also, there exists no data point blw the hyperplanes, $W^T X + b = 1$ & $W^T x + b = -1$
(margins)

---

## Page 40

& they are parallel to $W^Tx+b=0$
Consider the distance (perp) b/w the
margins: $\frac{2}{\|W\|}$
one critera for optimality of $g(x)$
is that $\frac{2}{\|W\|}$ has to be maximized.
Mathematically ,
$$
\min_{W} \frac{\|W\|^2}{2} \\
\text{s.t.} \quad (W^Tx_i+b) y_i: \ge 1 \quad \forall i \\
\quad \quad [1-(W^Tx_i+b) y_i] \le 0
$$
The above problem is a constrained
optimization problem, with Convex cost
& linear constraints.
Solve the above via KKT optimizations.
Define Lagrangian for the above problem:

---

## Page 41


$$
L ( W_{3}+u_{3} b )=\frac{W^{T} W} {2}+\sum_{i=1}^{n} \left[ i-y_{i} ( w^{T} x_{i}+b ) \right] H i
$$
$$
\begin{matrix} {{K k T}} & {{p v}} & {{t h i}} & {{a b o v e}} & {{p w b l e m}} & {{:}} \\ \end{matrix}
$$
$$
a ) \cdot\nabla_{\! \! W} L ( W^{*} , \mu^{*} )=0
$$
 $\hookrightarrow$ 中
$$
1 4 j^{\ast} ( 1-y_{j} ( w^{\tau} x_{j}+b ) )=0
$$
 $\nabla_{w}$ ：
$$
W^{*}=\sum_{i=1}^{n} \mu_{i} y_{i} x_{i}
$$
海9 T
n
$$
S=\left\{\begin{matrix} x_{i} : \mu_{i} > 0 \right\}
$$
egr $\omega\e$ nuata he ormat -agmukphe xi


---

## Page 42


$$
\frac{T_{0} \vert n d \vert\vert^{*}} {\vert^{*}} , w e f o l v e+\sharp e d u a l \vert\ p w b l e m
$$
$$
\frac{D u a l i t y o f} {L ( W_{i} H )}=\frac{W^{T} w} {2}+\sum_{i=1}^{n} \left[ 1-y_{i} ; \left( w^{T} x_{i}+b \right) \right] H i
$$
 $\triangleright\omicron$ 
$$
q \left( 1 \right)=\lim_{w} \frac{1} {0} L \left( w , \mu\right)
$$
 $\mathrm{S o u n d}$ 
 $\alpha$  $\mathbb{D} u a l$ RA
guaurin
$$
\cos\sin\tan\theta+\hbar e D u a l \cos\theta+\pi a t i o n p o b
$$
 $q ( \mu)$ 
max
sxoadleikwa riltoali
 $w^{*} ,_{I}$ r aut ohmget $\wp^{\textnormal{x}}$ pm\ $^e$ onueputstsyt


---

## Page 43


$$
\begin{matrix} {i )} & {{W^{*}}} & {{u^{*}}} & {{a r c}} \\ {{}} & {{E_{a} d u a l}} & {{r o p e i b v e l y}} & {{f^{n i m a l}}} \\ \end{matrix}
$$
$$
i i ) \rightarrow\left( w_{1}^{*} H^{*} \right)=\min_{N} ( w_{1} H^{*} )
$$


---

## Page 44


$$
\left\{\begin{matrix} {{i \displaystyle\int}} & {{w^{*} \in H^{*} \arg_{a n c}}} & {{\log_{a n b l e}}} & {{p_{a r} \lim_{i}}} \\ {{}} & {{\displaystyle\leq d u a l}} & {{r \in p e \mathop{c h v e l y}}} & {{\ldots}} \\ \end{matrix} \right.
$$
中
$$
L ( w^{*} / \mu^{*} )=\min_{N} L ( w / \mu^{*} )
$$
$$
\pi_{c} D u a l \rho/ u n c f i o n :
$$
$$
q ( \mu)=\lim_{w_{1} b} \left\{\frac{1} {2} w^{\dagger} w+\sum_{i=1}^{n} H_{i}^{\dagger} ( 1-y_{i} ( w^{\dagger} x_{i}+b ) ) \right\}
$$
ohnct.temE $\Lambda^{i} \Im$ b $\omega h : c h$ 
 $b_{y}$ 
 $\mathrm{i n} \zeta$ a%
 $\sec e$ 
X
$$
\Sigma_{i} \mu_{i} y_{i}=0
$$
$$
T_{0} \sin( q_{1} \ln( q_{2} )_{2} \ln( q_{2} )_{2} \ln( q_{3} )_{2} \ln( q_{4} )_{2} \sin( q_{4} )_{2} \sin( q_{4} )_{2} \sin( q_{4} )_{2} \sin( q_{4} )_{2} \sin( q_{4} )_{2} \sin( q_{4} )_{2} \sin( q_{4} )_{2} \sin( q_{4} )_{2} \sin( q_{4} )_{2} \sin( q_{4} )_{2} \sin( q_{4} )_{2} )_{2}=2 .
$$
$$
w^{\prime\prime}=\sum_{i \in S} H i y_{i} X i \qquad b^{\prime}=2 \qquad( 1+w ) .
$$
a
$$
q ( \mu)=\sum_{i=1}^{n} \mu_{i}=\frac{1} {2} \sum_{i} \sum_{j} \mu_{i} y_{i} y_{i} x_{i} x_{j}
$$
 $p w b l e m$ ,thedaleX
on\y axpea<s ne" preduct


---

## Page 45


$$
D a a l \circ p \tan2 a \tan p \circ b l e m
$$
下 s+ $\mu\mathrm{i} \geq0$ c'+hsa $m a \times$ 
$$
\Sigma_{i} y_{i} \mu_{i}=0
$$
Theaboe palum Ie (omadahe iK wsithineaxs Constraipes
CAASR losintance Seqyuchal $\mu i n-o p t$ 
(SMo)
 $l i b s \lor M$ 
anteyareOkuanea,
小心
$$
w^{\prime}=\sum_{s} y_{i} \mu_{i} x_{i}
$$
SA. $\mu^{\mu} > 0$ dand
 $1 5 o \verb1 e$ 
 $\flat^{\infty}$ Chct^


---

## Page 46


$$
f o m \cdots o n e \cdots o f f h_{e} K K T \cdots G n d .
$$
$$
\mu_{i}^{*} [ 1-y_{i} ( w^{\dagger} x_{i}+b ) ]=0
$$
 $\Uparrow^{\infty}$ antbdVetkos,
$$
y_{i} \left( w^{\tau} x :+b \right)=1
$$
J
$$
S_{V M}=p r o t \lim_{n o t} \lim_{n o t} \log_{a v}-s e p e r a b l e C a i c
$$
$$
\Rightarrow\pi W I t y_{i} ( w^{\tau} x_{i}+b ) > 1
$$
 $\biguplus$ i
 $\operatorname{s n t r o d u c e}$ edrherwealeinhe
ptoingahan hat ononwgie te eatae rphatesnjudsn
$$
\pi e o r i g i n a l g i o b l e m :
$$
士 $\omicron^{T} \omega$ a slack Vavable
 $\sim^{\rho} n$ 
大
n{aut) E1-,


---

## Page 47

ATAT $\operatorname{l e a d}$ to $\alpha$ 
-souhon 0 ： $s \omicron\sp{\backslash} \omicron$ aCant'vained $\lor\rho\circ n$ 
$$
\min_{w_{1}} \frac{1} {2} w^{\max} w+C \sum_{i=1}^{n} E_{i}
$$
Y
$$
3 . t y_{i} ( w^{\prime} x_{i}+b ) > 1-z_{i}
$$
$$
\mathcal{E} \geq0
$$
sT unta ane otiem
$$
v L=0 \Rightarrow w^{*}=\sum_{i} H_{i}^{*} y_{i} \times i
$$
+
$$
V_{b} L=0 \Rightarrow\sum_{i} \mu_{i} : y_{i}=0
$$
$$
\begin{array} {c} {{i i i \displaystyle\left( \begin{array} {c} {{V_{\xi} L=0 \Longrightarrow\mu_{i}+\alpha_{i}^{*}=C .}} \\ {{i v}} \\ \end{array} \right) \displaystyle\left. \right.}} \\ {{i i \displaystyle\left( \begin{array} {c} {{i-E_{i}-y_{i} \left( w^{T} x_{i}+b \right)}} \\ {{c \end{array} \right) \displaystyle\left( \begin{array} {c} {{0 . \in\Omega_{i}}} \\ \end{array} \right)_{i}}} \\ \end{array}}} \\ \end{array}
$$


---

## Page 48


$$
\begin{array} {c} {{F_{o r m u l a k n g}+h e}} & {{D u a l .}} \\ \hline{{\Downarrow}} \\ {{\chi( H , d )=\lim_{w , b , \, \in{\cal E}}}} & {{L \left( w , b , \, \varepsilon, h , d \right)}} \\ \end{array}
$$
$$
\Sigma y_{i} b=0
$$
$$
\begin{matrix} \rho_{i} \rho_{i}=\rho_{i}+\rho_{i}=\rho_{i}+\rho_{i}=\rho_{i}+\rho_{i}=\rho_{i}+\rho_{i}=\rho_{i}+\rho_{i}=\rho_{i}+\rho_{i}=\rho_{i}+\rho_{i}+\rho_{i}=\rho_{i}+\rho_{i}+\rho_{i}=\rho_{i}+\rho_{i}=\rho_{i}+\rho_{i}=\rho_{i}+\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}=\rho_{i}} \\ \end{matrix}
$$
 $\uplus$  $\tan_{n=1} \ln_{n} \log_{n}$ c→t1
 $D u a l$ plen
$$
\max\sum_{i} \mu_{i}-\frac{1} {2} \sum_{i} \sum_{j} y_{i} y_{j} H_{i} x_{i} x_{j}
$$
3.、 $\mu: \geq0$ 
$$
d_{i} : \geq0 \quad a n d \qquad C=\mu i+d_{i}
$$
 $\alpha\rightsquigarrow$  $0 \leq M : \leq C$ "ig t-omn
 $\mathrm{d_{i}}$ 
 $\mathrm{m a X}$ &+i"zPSD;,pr'T $\Lambda$ 
$$
\begin{matrix} 8 .+M : \geq0 , \\ 0 \leq M : \leq C . \end{matrix}
$$


---

## Page 49


$$
S y \sim w i t h t h e K e m e l f
$$
mtn
$$
\begin{matrix} \angle{\mathrm{D e g i n c a f-k o n}} \\ \end{matrix} \rightarrow\begin{matrix} {\angle{\mathrm{E i o n}}} \\ \end{matrix} \phi: R^{\alpha} \rightarrow R^{\alpha}
$$
 $\omega\mathrm{i} \, \mathrm{U}$ hntu mtyAmy be
$$
\max\Sigma\mu i-\frac{1} {2} \Sigma\sum_{i} \mu_{i} \mu_{j} y_{i} y_{i} \phi( x_{i} )^{\dagger} \phi( x_{j} )
$$
st
$$
0 \leq\mu; \leq C
$$
KR大区→R
snanA sboae wnd tae n
$$
\Sigma\mu i-\frac{1} {2} \Sigma\Sigma\mu\rightarrow4 y_{i} y_{i} k ( x_{i} x_{j} )
$$


---

## Page 50


1 $\mathrm{\ v a l i d}$ 
$$
\underline{{{H e x c e x^{\prime} s ~ t h e o r e m}}} : \ldots\varphi p o s e \ldots\overline{{{K}}} \ldots i s
$$
a
 $\mathrm{n x n}$ mm.sh E,-5{SNO somwel (oe kaot foune
·P4）
Sumke wut tawhe
 $\ddag^{o} \Downarrow$ WwdAsDdzuxY
 $\mathcal{S}$ CHn4+gj*uo
 $\mathit{S}$ K&y,twk/ain)
$$
\begin{matrix} G a u s t i a n K e r n c l : \\ K a ( x_{1} x_{2} )=e^{-\left\| x_{1}-x_{2} \right\|^{2}} \end{matrix}
$$


---

## Page 51


$$
\frac{S v M a s R e g u l a n g e d E R M} {g}
$$
$$
\begin{array} {c} {{m_{i} b_{i}^{\prime} z_{i}=\frac{1} {2} w^{\dagger} w+C \sum E_{i} ;}} \\ {{s+y_{i} ( w^{\dagger} x_{i}+b ) \geq i-E_{i}}} \\ \end{array}
$$
$$
\mathcal{E}_{\mathrm{i}} \geq o
$$
$$
\begin{matrix} \arctan y=w \in b_{j} \in\left\{\begin{matrix} \\ \end{matrix} \right. \left\{\begin{matrix} \\ \end{matrix} \right. . \left\{\begin{matrix} \\ \end{matrix} \right. \left\{\begin{matrix} \end{matrix} \right. . \left\{\begin{matrix} \end{matrix} \right. .=\left\{\begin{matrix} \end{matrix} \right. a b c l y \\ \left\{\begin{matrix} \end{matrix} \right. \left\{\begin{matrix} \end{matrix} \right. .=\left\{\begin{matrix} \end{matrix} \right. \left\{\begin{matrix} \end{matrix} \right. . \left\{\right. .
$$
?
w,s2 A SALWORC $\hat{i}=\vert$ 
$$
\Omega( w )+C \Sigma( h ( x ) , y )
$$
$$
L ( h ( x ) , y )=m a x ( 0 , 1-y_{i} , h ( x_{i} ) )
$$
$$
\cup\tan g e-\log_{1} \alpha_{2}
$$


---

## Page 52


$$
\underline{{{C_{0 n v o} l u b o n a l}}} N_{t u r a l N e t w o r k s}
$$
能中 $d-l i k c "$ 
 $\mathsf{M L P}$ 
 $\sigma$ loaine
 $\theta$ 
 $\varepsilon_{\omicron}$ Twage $\Im D$ vouma
 $\underline{{C_{\mathrm{N N S}}}}$ ：MPML ho $\mathbf{h a r d^{n}}$ 
rginamaotyni $\sigma p a c_{e}$  $L o c a l$ rainae tata
心
0
入 $\bigcirc$  $\underbar{I n}$ 
 $\copyright$ 
 $\supset$ 
I
$$
\begin{array} {c} {{L_{0} c a l \quad\mathrm{R e c e p t i v e f i e d d : c o n n e r i t} \quad a l y}} \\ {{\quad\mathrm{a s u b s e t} \quad o f \quad d a t a / a c k u a l i v a n s \quad\mathrm{t o e v y}}} \\ {{\quad\mathrm{n u v o n i n c o c h} \quad\mathrm{l a v e r .}}} \\ \end{array}
$$


---

## Page 53


$$
\frac{\log_{\alpha\alpha\min} g} {\log_{\alpha\beta}}
$$
wi
中楼菜 s29
)
W
X $\sigma$  $\upsilon$ on
w%
o $\theta_{2}$ 2 2 ·nNw $\mathrm{e}$  $\varDelta a \aleph$  $6^{e l d}$ 
esey
is Kepe
2 $C_{0 N}$ o（
 $\varkappa$ 
 $\kappa_{1}=$  $q-k_{1}+1$ 
 $P-K_{1}+$ 
 $\frac{1} {2}$  $1 < 1$  $\copyright$  $\sigma\left( \textbf{\Lambda} \right)$  $0-k_{2}+1$ 
 $x_{2}=$ 
 $x-2$ 
P $k_{2}+$ 


---

## Page 54


$$
\subset_{\mathsf{N N}}
$$
￥
 $\mathbf{q}$  $\mathsf{P}$  $\kappa$  $\kappa$ 、 $\left[ \frac{2-\kappa} {s} \right]+$ 
 $\mathsf{k}$  $\iota$ klia -
 $\copyright$ 
 $\frac{P-k} {S} \uparrow+1$ 
A
tro A
味
Neag laish
” $\log U_{n} g$ 
$$
q \textcircled{<} [ \begin{matrix} y_{u} y_{u} \\ y_{u} y_{u} \end{matrix} ]
$$


---

## Page 55


$$
\frac{C_{X a m p} l e 3 0 f} {} \xlongequal{C N N-E_{n} d-t_{0}-E n d}
$$
$$
\log\b> \log\b> \log\b> \log\b> \log\b> \log\b> \log\b> \log\G\G\ni\emptyset
$$
-danres
$$
q^{\prime} \boxed{\cdots}_{P}=\frac{\cos v} {\longrightarrow} p^{\circ\alpha} l \xrightarrow{\cos v} p^{\circ\alpha} l^{\prime} \cdots f^{\log\frac{1} {\alpha} \sqrt{\alpha} \cdot n} q
$$
 $\pm\max^{1} \varPsi e$ E $L e N e t$ . hand eporSruspiumds
→ $F u i l_{y}$ taukwt
Smnby atsgmantam
长
 $q$  $\frac{C_{0} n^{\vee}} {\square}$ 0
1
u-net $\mathrm{P}$ 
 $\mathcal{E}_{\mathrm{n C}}$  $\operatorname{D e c}$ 


---

## Page 56


$$
\frac{R e {\it c u r r e n t} N_{t u r a} l \quad N e t \omega_{o r} k s .} {}
$$
$$
S_{u i t d} \wp\circ\sin\phi\circ\sin\phi\circ\phi\circ\phi
$$
 $h a T_{u v c}$ ngu $\mathrm{t o}$  $\mathrm{b e}$  $^o t$ jMluy $\cot\sin\sin\alpha e x$ 
1 $\omega h e v e$ 人 y
$$
x=\{x^{1} , x^{2} \ldots, x^{7} \}
$$
x`eR
 $\Sigma_{\mathcal{J}}$ 中R $\omicron=\rho$ V
Wni'ced sen
yrdEo
tragprnaskd on wukh $\mathrm{d a t a}$ 
0
ateten yrn tait
 $\omicron: \bigcirc$ MspsM to soyuoic $\tan\theta_{a n} ( 1 ) a \tan$  $\nearrow$ ertun
metlwne
 $\mathcal{I}_{\mathbf{n}}$ asqunal anta, $\begin{matrix} \mathrm{e v e n} \\ \angle{e n g} \end{matrix}$  $d a t a {\mathrm{p o i n t}}$ 
 $\varepsilon$  $\\#_{\boldsymbol{\mu} \boldsymbol{\mu} \boldsymbol{?}}$ P $C a n n o t$ handle them.


---

## Page 57


$$
\underline{{\textrm{R N N s}}}
$$
欢中
$$
x=\{x_{1} x_{2} \ldots x_{r} \}
$$
$$
z_{t} \;=\; h_{t} \, h_{t-1} \,+\, W_{z} x_{t}+b_{1}
$$
大 $s_{i n q}$  $\mathbb{R}_{\mathsf{N N}}$  $c$ a
$$
h_{t}=\sigma( z_{t} )
$$
A
$$
\widehat{y}_{t}=H_{3} h_{t}+h_{2}
$$
2
:
 $\grave{\imath}_{\mathrm{t}}$ 车 $\iota$ 
W,
Xt Xw $\times_{\top}$ 
Y
XT $\Delta$  $\preceq$ 
$$
\frac{B a c k-p v o p a g a k o n i n R N N s} {C}
$$
3PTT $l_{e a d s}$  $t_{\infty}$ 
negpcncts


---

## Page 58


$$
\frac{\sinh_{i n} g} {g} g^{r a d i e n t s} i n R N N s
$$
$$
A t c a c h \tan e \det t ,
$$
$$
{\bf Z}_{\bf t} \;=\; \; W_{i} \, h_{t^{-1}}+W_{2} \, {\bf x}_{t}
$$
$$
h_{t}=\sigma( z_{t} )
$$
A
 $\partial\textnormal{h t}$ 0k $\partial\textnormal{h t}$ 力
Blrnalrah
T
anse $\partial h_{t}$ 2bxn
宝
ahk
K=t
o<3;de<
hk,
abK
$$
h_{k+1}=\sigma( W_{1} h_{k}+W_{2} \times_{k+1} )
$$
o hane
$$
\begin{array} {c} {{\partial h k+1}} \\ \end{array}=\begin{array} {c} {{d i a g}} \\ {{\partial h k}} \\ \end{array} \Bigg( \sigma^{\prime} \bigg( Z_{k+1} \bigg) \Bigg) W_{1} .
$$


---

## Page 59

lomaie ksmon oe tadoscukn. $\grave{\omicron} \textsc{h} \tau$ '
 $\partial\textrm{h}_{\L}$ 
$$
\prod_{k=1}^{T-1} \left\{\begin{matrix} \end{matrix} \right. \left\{\begin{matrix} \end{matrix} \right. . \left\{\begin{matrix} \end{matrix} \right. \left\{\right. . \left\{\begin{matrix} \end{matrix} \right. \left\{\right. . \left\{\right. .
$$
$$
\left| \frac{d i a g} {g} \right( \sigma^{\prime} ( z_{\alpha1} ) ) \left| \right|=
$$
力
s $\vartheta$ A
 $\partial\operatorname{h} \tau$ 
 $\supset$ 
 $\alpha$ tagy $c x p o n e n t i a l l y$ 
协 $\mathsf{T-t}$ 
rie unatMteakial ecjctke $\mathcal{I_{n}}$ a rsanect $\delta y \dagger\tan,$ 
$$
h_{t}=\mathcal{F} \left( h_{t 1} , x_{t} \right)
$$


---

## Page 60


3
$$
\frac{\partial h_{t}} {\partial h_{t}}=\prod_{k=t}^{\tan1} \frac{\partial h_{k+1}} {\partial h_{k}}
$$
$$
\begin{array} {c} {{I n \quad a \quad\textrm{N a n i l l} a \quad\textrm{R N N} , \quad\textrm{N a n i l h : n g} \quad\textrm{g r a d}}} \\ {{\textrm{h a p p e n I I-J i n i c e-} \displaystyle\frac{\partial h \kappa\kappa} {\partial h \kappa}=\textrm{d i a g} \left( \sigma^{\prime} i \right) . \begin{array} {c} {{g r a d}} \\ {{W}} \\ \end{array}}} \\ \end{array}
$$
aAIA IwAa
2<
nethieiupde geae rtn.
E he
$$
h_{t}=\alpha_{t} \theta h_{t-1}+\beta_{t} \theta h_{t}
$$
 $\alpha_{\mathrm{t}}$ 1s the vewor iae
LRNL
Bastmaialion'
$$
\frac{T e g r o d e n t s} {} f o r \tan\angle C a b o v e n e h j o r k s .
$$
o
$$
\frac{\partial h_{t}} {\partial h_{t-1}}=\frac{\alpha_{i} a g ( \alpha_{t} )+\frac{\partial\alpha_{t}} {\partial h t-1} d i a g ( h_{t-1} )} {+\frac{\partial( \beta_{t} \partial\tilde{h}_{t} )} {\partial h t-1}} \Bigg\}
$$


---

## Page 61


$$
= \lim_{t} ( \alpha_{t} )+\varSigma_{t} ( w_{3} w_{2} )
$$
$$
\frac{\partial h_{T}} {\partial h_{t}}=\prod_{k=t}^{T-1} \left( \alpha_{k} a g \left( \alpha_{k+1} \right)+E_{k+1} \right)
$$
发 T-\ A
H: $R e \sin a 1 / \sin\beta$ 
 $C_{0 \cap n e \sim1} \operatorname{c f i o n}$ 
 $\sin\angle D-\cos\angle C k : 0$ 
 $\oplus$ 
 $\times$ 


---

## Page 62


$$
\frac{R e g u l a r i g a t i o n} {g}
$$
$$
\begin{array} {c} {R e g u l o n i g a l i o n : T e c h a g u e s+0 : i n c r i a l c} \\ {b+h e m o d e l b i a s .} \\ \end{array}
$$
 $\grave{\omega}$ mntireproatly
) Archstacival -cesbickoof
eMprjpnu paTn
$$
R e g u l a n g a b i o n v i a p a r a m e t_{e v} p a n a l t
$$
Rua\,FR&^:
$$
\widehat{h}_{\theta}^{\bullet} ( x )=\arccos\frac{1} {N} \sum_{i=1}^{N} L ( h_{\theta} ( x ) , y )
$$
 $\log u \log_{a v i j} e d$  $R \sim\circ$ EU $D e f i n e$ pOR
$$
\mathcal{E} \det( n \geq a n \in w-o b j c d i v e \ldots a s \smallint_{0} \|_{\partial W I} )
$$


---

## Page 63


$$
\begin{array} {l} {{\mathrm{R e g-E R M :}}} \\ {{\displaystyle\widehat{h}_{\theta}^{\ast} ( x ) \ =\ \mathrm{a r g m i n ~} \frac{1} {N} \sum_{i=1}^{N} \ L \left( h_{\theta} ( x ) , y_{i} \right) .}} \\ \end{array}
$$
$$
\sin_{j} \alpha t+o \Omega( \theta) < k .
$$
ushoLKeR
1
$$
\frac{1 0} {3 0 N e} r e g-E R M^{\circ}
$$
n
r"f omb.
$$
\widehat{R}_{r} ( 0 )=\frac{1} {N} \sum_{i=1}^{N} \tan( h_{0} ( x ) , y_{i} )+\Lambda( \Omega( 0 )-k )
$$
$$
h g^{*}=\frac{a v g n i n} {0} \widehat{R}_{r} ( \theta)
$$
muastadea
$$
\cot\angle C S f O_{r} \Omega( \theta)
$$
$$
\varOmega~ ( \theta) ~=~ \left\| ~ \theta\right\|_{P}
$$
$$
\begin{matrix} \parallel0 \parallel_{2} :=L_{2}-r e g \mid R i d g e} \\ \parallel0 \parallel_{1} :=L_{1}-r e g \mid L a s s o} \\ \end{matrix}
$$
 $\therefore\bigcirc$ 
$$
\Omega( \theta)=\exp(-\theta)
$$


---

## Page 64


$$
\frac{E_{Q u i v a} I_{e n c e \ldots} b I \omega\cdot R e g-E R M \cdot E_{i} \cdot M A P \cdot E_{J} h i m a t_{e J}} {g}
$$
 $R e c a W$ 
$$
I_{n} \ldots\mathrm{N a p-e s f i m a^{\dagger} e} \ldots\omega e \ldots\mathrm{s e c k} :
$$
$$
\phi\left( \begin{matrix} {\theta} \\ \end{matrix} \right| x , \Bigr) \propto\textrm{P} \left( \begin{matrix} {y \mid x , \theta} \\ \end{matrix} \right) . \ P \left( \begin{matrix} {\theta} \\ \end{matrix} \right)
$$
出心
$$
\phi( y+0 ) \sim\sqrt{( y ; h_{0} ( x ) , I )}
$$
$$
\sin\theta=P ( \theta)=N \left( \theta; 0 , I \right)
$$
$$
\chi( \theta)=\frac{1} {N} \sum_{i=1}^{N} \log P ( y_{i} | \times1 \theta)
$$
$$
\alpha\frac{-1} {N} \sum_{i=1}^{N} \parallel y_{i}-h_{0} ( x_{i} ) \parallel_{2}^{2}
$$
$$
\mathrm{P} ( \theta\left| y , x \right) \propto\underbrace{\mathrm{P} ( y \left| x , \theta\right) \mathrm{P} ( 0 )}
$$
$$
\lambda( \theta) \propto\log P ( \theta| y , x )
$$


---

## Page 65


$$
\alpha\log P ( y \mid x_{1} \theta)+\log P ( 0 )
$$
$$
\alpha\frac{-1} {N} \left\| y_{i}-h_{\theta} ( x_{i} ) \right\|_{2}^{2}-\left\| \theta\right\|_{2}^{2}
$$
一 1 6"+w%g+ &:D)
$$
= \arcsin^{i} \theta-\log_{\theta}^{i}-\log_{\theta}^{i} ( \theta)
$$
$$
= \arg\min\cap\frac{1} {\sqrt{\sum_{i=1}}} \left\| y_{i}-h s ( x ) \right\|_{i}^{2}+\left\| \theta\right\|_{i}^{2}
$$
$$
= \widehat{R}_{r} ( 0 ) , \omega i t h \Omega( 0 )=1 1 0 1 2
$$
京
$$
\frac{\lambda} {i s} \underbrace{\lim}_{j=1} \cup_{j=1} \cup_{j=1} \frac{1} {\lambda} \frac{1} {\lambda} \frac{1} {\lambda} \frac{1} {\lambda} \frac{1} {\lambda} \frac{1} {\lambda}
$$
 $\mathrm{p} ( \diamond)$  $/$ heLeot can-per
'aT $L_{1}$ wl-1dypro

---

## Page 66


$$
\mathrm{\boldmath~ \begin{matrix} {{\sin}} \\ {{\scriptstyle\mathrm{\boldmath~ P ~}}} \\ \end{matrix} \left( \begin{matrix} {{\theta}} \\ \end{matrix} \right) \sim\mathrm{\boldmath~ L a p l a c i a n} \left( \begin{matrix} {{\alpha}} \\ \end{matrix} \right) ,}
$$
in
he $\flat^{\omicron} \flat$ siprametae acand mtun sit\ $\b b e$ Asha tuwn tid of-oantkn
$$
\operatorname{e x p} (-\lfloor0 \rfloor)
$$
7 $l o o s s$ 
Lnsn $l r o g$ 
1
$$
A n o l h e r=e x a m p l e p r R e g u l a n g a h o n .
$$
1
$$
\log_{n} e D=\{( x , y_{1} ) \} \sim i i d P_{n y}
$$
$$
\tan\angle a n e w-\dim a s e t \tan a s e t
$$
$$
\forall x \mathrm{i} \in D ,
$$
$$
\widehat{x_{i}}=x_{i}+\epsilon\in\sim N ( 0 ; \alpha I )
$$
$$
\widehat{D}=\left\{\left( \widehat{x}_{i} , y_{i} \right) \right\}
$$


---

## Page 67


$$
\underline{{\mathrm{c l a i m :}}} \quad\mathrm{~ E R M o n ~} \quad\overrightarrow{D} \quad\equiv\mathrm{~ R-E R M o n ~} D .
$$
$$
\frac{R e v i s h i n g} {g} G v a d i e n t D e i c e n b :
$$
$$
\widehat{R} \left( 0 \right)=\frac{1} {N} \sum_{i=1}^{N} L \left( h_{\theta} \left( x_{i} \right) , y_{i} \right)
$$
2
$$
\theta^{t+1} \longleftarrow\theta^{t}-\beta\cdot\nabla_{\! \theta} \widehat{R} \left(_{\! \theta} \right)
$$
 $A \Downarrow e x$ theCrad-descent:
$$
D e f i n e \widehat{R}_{B} ( 0 )=\frac{1} {B} \sum_{i=1}^{B} L ( h_{B} ( x_{i} ) , y )
$$
whekBLN
8xLDgwhexe
$$
D_{B} C B-l e n g t h \rightarrow H_{2}=0 .
$$
3"
$$
\theta^{t}-\beta\cdot\nabla_{\theta} \vec{R}_{B} ( \theta) \, .
$$
$$
\sinh a \log C G a d \log C ( S G D )
$$


---

## Page 68

 $\mathrm{M a x-}$ maypngueso Aunt uain
 $M a c h i c s$ 
$$
\cos\alpha\sim a B i n a y \cos\beta\cos\alpha\cos\beta\cos\beta
$$
x eR $\mathcal{E}$  $\zeta$ 1Eo3
 $D i s c n m i n a n t$ mahuo
 $\mathrm{R e c a l l}$  $\ \#_{\mathrm{a b}}$  $\gamma e p \times e 1 e t e d a s$ auy $C_{\sf c o n}$ 
ON
b。
$$
h \left( x \right)=\left\{\begin{matrix} 1 g \left( x \right) > 0 \\ -1} & {o t h e r w i s e} \\ \end{matrix} \right.
$$
 $\omega h_{c} \nu e$ 
 $g^{( x )}$ :RCsR
$$
g^{( x )} : \quad d i \sin\tt m i n a n t \quad f u o c h o n .
$$
se


---

## Page 69

二 $\times$  $\epsilon$ 
 $W T_{x}+b=-1 \textcircled{2}$ 十 2E
$$
\begin{matrix} y \in\{-1 , 1 \} \\ x_{i} : \{x_{i}^{2} , x_{i}^{2} \} \end{matrix}
$$
 $\sqrt{7} \times+6=2$ x
:
0
002
中
$$
x : y=-1
$$
司 $( x )=W^{T} x+b$ 
0 O
1 $n e a v$  $\frac{S e p e x a b l i t y} {1}$ 5 $\mathcal{A}$ Dakzse $\mathtt{D}$  $\mathbf{i} \le$ 
州
$$
W^{T} x_{i}+b > 0+f x_{i} : y_{i}=1
$$
“
-epewbke $\i+1$ 
 $\ss$ 
$$
W^{T} x_{i}+b \angle0+X_{i} : y_{i}=-i
$$
1
$$
\begin{array} {c c} {{\lim_{i n \in\omega\cup\psi} s e p e \mathrm{~ r a b l e ~} d a t_{\omega} \qquad\omega; i \ne\mathrm{~ h a v e}}} \\ {{i n \prod_{i} t_{i} y^{i} \ne\mathrm{m a n y} d i \sinh_{i} \phi\hspace{-0 . 5 c m} j_{i n} \in\mathrm{f i n ~ c h a n i t .}}} \\ \end{array}
$$
outie $\mathtt{I_{S}}$ therc an"ophkmad`
srsytn

---

## Page 70


$$
\frac{\alpha_{p} \tanh b_{y} \ldots\textrm{v i a l} \ldots m a \tan\chi_{i n} g \tan\chi_{j}} {\cup}
$$
$$
\zeta_{0 n \le i d e r} \qquad l_{i n \le i p \le r a b l e d a t a}
$$
$$
\sin\left( e \ldots i t \ldots i s \ldots l i n \ldots s e p e \, r a b l e \, , \ldots{\cal J} \in> 0 \ldots s . t . \right.
$$
 $W^{T} x_{i}+b \geq\epsilon$ 艺 $y_{i}=-1$ 1
 $y_{1}=1$ 
 $W^{T} x_{i}+b$ Lr&YX
N $\a$  $\Leftarrow$ bcanbeeauald,sush $\ \#_{a} \uplus$ 
 $\stackrel{\sim} {\sim}$ t $y_{i}=-1$ 
xi+SA $y_{i}=1$ 
W'xtDE+
Te abote in be ambaed at
$$
y_{i} ( w^{T} x i+b ) \geq1+i
$$
is
$$
\lim_{n \rightarrow\infty} \tan_{n} \Xi w \Xi\cdot\textrm{b} . \textrm{I . t .}
$$
$$
y_{i} ( w^{T} x_{i}+b ) \geq1+i
$$
$$
\begin{array} {c} {{\mathrm{d i v y b}_{\longrightarrow} \cdots\mathrm{t h e r t} \qquad e x c \qquad e x c \in\mathrm{h a r y} \in\qquad\mathrm{d i v e r t}}} \\ {{\mathrm{b i o n ~ t h e} \qquad\mathrm{h y p e r ~ p l u n c t} \ : \qquad\mathrm{W^{T} x ~+b=1 ~} \qquad\left\{\qquad W^{T} x+b=-1 \right.}} \\ {{\left( \mathrm{m a r y} \right)}} \\ \end{array}
$$


---

## Page 71


$$
\begin{array} {c c c c c} {{\xi}} & {{\mathrm{t h i y}}} & {{\mathrm{a r c}}} & {{p a v a l l e l}} & {{+o}} & {{\mathrm{W^{T} x+b=0}}} \\ \end{array}
$$
 $C_{O n} \sin\sin d e x$ t:a Artat yreyhne Tke warpn $\imath$  $\parallel\omega\parallel$ 
2
 $\log$  $\mathrm{C x}$ o $2$ W $m a l i t y$ R0 one $\alpha$ 
 $\ \#_{a} \omicron$ - $\parallel\omega$ 中 ngjxt Wximustdy
瓜人 $\operatorname* {l i m}$ 
N 2
 $( w^{T} x :+b )$ y: $\geq\ldots$ +i
8.+
 $[ 1-( w^{T} x_{i}+b ) y_{i} ] \leq0$ 
b onstrana opingatson ablem, wsth Canwex lant 久inea<onstxcults.
Snc $\ \#_{\mathrm{h-e}}$ Awe we AKT ohengant
$$
{\cal D} e f^{n e \ldots} L a g r a n g i a n \ldots\gamma^{n} \tan a b o v e \beta^{n o b l e m :}
$$


---

## Page 72


$$
L ( W_{3}+u_{3} b )=\frac{W^{T} W} {2}+\sum_{i=1}^{n} \left[ i-y_{i} ( w^{T} x_{i}+b ) \right] H i
$$
$$
\begin{matrix} {{K k T}} & {{p v}} & {{t h i}} & {{a b o v e}} & {{p w b l e m}} & {{:}} \\ \end{matrix}
$$
$$
a ) \cdot\nabla_{\! \! W} L ( W^{*} , \mu^{*} )=0
$$
 $\hookrightarrow$ 中
$$
1 4 j^{\ast} ( 1-y_{j} ( w^{\tau} x_{j}+b ) )=0
$$
 $\nabla_{w}$ ：
$$
W^{*}=\sum_{i=1}^{n} \mu_{i} y_{i} x_{i}
$$
海9 T
n
$$
S=\left\{\begin{matrix} x_{i} : \mu_{i} > 0 \right\}
$$
egr $\omega\e$ nuata he ormat -agmukphe xi


---

## Page 73


$$
\frac{T_{0} \vert n d \vert\vert^{*}} {\vert^{*}} , w e f o l v e+\sharp e d u a l \vert\ p w b l e m
$$
$$
\frac{D u a l i t y o f} {L ( W_{i} H )}=\frac{W^{T} w} {2}+\sum_{i=1}^{n} \left[ 1-y_{i} ; \left( w^{T} x_{i}+b \right) \right] H i
$$
 $\triangleright\omicron$ 
$$
q \left( 1 \right)=\lim_{w} \frac{1} {0} L \left( w , \mu\right)
$$
 $\mathrm{S o u n d}$ 
 $\alpha$  $\mathbb{D} u a l$ RA
guaurin
$$
\cos\sin\tan\theta+\hbar e D u a l \cos\theta+\pi a t i o n p o b
$$
 $q ( \mu)$ 
max
sxoadleikwa riltoali
 $w^{*} ,_{I}$ r aut ohmget $\wp^{\textnormal{x}}$ pm\ $^e$ onueputstsyt


---

## Page 74


$$
\begin{matrix} {i )} & {{W^{*}}} & {{u^{*}}} & {{a r c}} \\ {{}} & {{E_{a} d u a l}} & {{r o p e i b v e l y}} & {{f^{n i m a l}}} \\ \end{matrix}
$$
$$
i i ) \rightarrow\left( w_{1}^{*} H^{*} \right)=\min_{N} ( w_{1} H^{*} )
$$


---

## Page 75


$$
E q u i v a l a n c e : b l w . E R M : \xi d i s l u b u l i o n , \nonumber_{\mathcal{E}_{\pm} t}
$$
1
$$
\begin{array} {c} {{\mathrm{s u p p o s e}}} \\ {{D=\left\{x_{i} , y_{i} \right\}_{i=1}^{n}}} \\ \end{array} i i d d D_{x y}
$$
D k $+_{\mathrm{O}}$ exthmdteasaebhabarbsor $\oint$ 
 $\operatorname{v i a}$ SRR,B
8usc+tqne<mnmngekBn
 $1_{\infty}$ h.- xt& mnguianD $\alpha p p r o a c h$ 上 the raun Soblemts to $e s t i m a t e$ 
^ $f \sim\det\alpha$ hx→Y ,om D, $\operatorname{v i a}$ fmpniul camninrgabon
 $p \circ b \quad\alpha p p \circ a c h ,$ 
$$
\tilde{\theta}=\arccos_{\theta} D_{k L} ( P_{x} / P_{\theta} )
$$
ERM,Gen D,
$$
h^{r}=\arcsin\frac{1} {N} \sum_{i=1}^{n} L ( h ( x_{i} ) , y )
$$


---

## Page 76


$$
\underbrace{c l_{a i m}} : \mathcal{D}_{i} v \ldots m i n i m \mathcal{J} \vec{d} i_{0} n \ldots\mathcal{E}_{i} E_{R} M \ldots c v e \ \ \mathcal{E}_{q u i l i d}
$$
 $\stackrel{a_{c}} {\jmath}$ 8
$$
\begin{array} {c} {{\mathrm{s u p p o r e} \quad\omega e \omega a n t \quad t o \quad e x \hbar m a l e P_{y l x}}} \\ {{P_{o} \left( y \vert x \right) \sim N \left( y ; h_{b} ( x ) , \pm\right) \quad\mathrm{i f i n a l e} P_{y l x}}} \\ \end{array}
$$
 $\mathrm{M a x .}$ kuioadeohyesd $\lor\mathrm{e}$ 火
R mocentmsrenu eartin
$$
\frac{\mathrm{L i n e a r F a m i l y ~ o p ~ M o d e I s}} {1} .
$$
 $\epsilon$ KAK
 $\mathrm{h_{\theta} ( x )=\Theta^{T} x+\Theta_{o}}$  $\times$ seR^,&.eR
$$
\begin{matrix} {{\mathrm{w i t h}}} & {{x}} & {{=}} & {{\left[ x} & {1 \right]^{T}}} \\ \end{matrix} , \begin{matrix} {{h_{\theta}}} & {{x}} \\ \end{matrix} )=\begin{matrix} {{\theta^{T} x}} \\ {{\theta}} & {{\in R^{d+1}}} \\ \end{matrix}
$$
$$
E R M \quad\mathrm{~ w i t h ~} \quad L_{i \, n \, e \, a x} \quad\mathrm{M o d e l ~ s ~} \quad\mathcal{E} \quad\mathcal{E} \quad\mathcal{q} \cdot\mathcal{E}_{\mathrm{M o n ~ l o i J}} \, .
$$


---

## Page 77


$$
\begin{matrix} \theta^{*}=\arcsin\frac{1} {N} \sum_{i=1}^{N} \left\| \log\left( x_{i} \right)-y_{i} \right\|_{2}^{2}} \\ =\arcsin_{2} \theta^{n i n} \frac{1} {N} \sum_{i=1}^{N} \left\| \theta^{i} x_{i}-y_{i} \right\|_{2}^{2}} \\ \end{matrix}
$$
T
 $\omega h e v e$ ×= ×，
xi
A
$$
y=[ y_{1} y_{2} \cdot y_{n} ]
$$
$$
\frac{G e n e v a l i g e d \tan C a r M o d e l s} {g}
$$
 $h o ( x )=\theta_{1} x^{\prime}+\theta_{2} x^{2}+\cdots\theta_{\alpha} x^{\alpha}$ 
me t hei on daww InsEa. $\omicron f$  1822 cn $c o n \sin d e v$ nce lineae Cambiaubans $\omicron\f$  $d \omega t a$ dimension1:


---

## Page 78


$$
h_{\theta} ( x ) \; \;=\; \; \vartheta_{1} ( x^{\prime} )^{3}+\; \vartheta_{2} ( x^{\prime} )^{\! \! h}+\; \cdots
$$
AA T $\dot{\sim} \dot{a}$ prer Lompaied trat permuhiani
$$
\begin{array} {c} {{G i j e n \quad x , \quad d i f i n c d \quad\widehat{x}=\phi_{i} ( x )}} \\ {{\phi i s \quad a \quad\frac{1} {0} x e d \quad\tan\phi\times m a b i o n .}} \\ \end{array}
$$
wnce
$$
\widehat{x}=[ ( x^{\prime} )^{2} ( x^{2} )^{3} ( x^{\prime} )^{2}+( x^{2} )^{6}
$$
0
$$
\phi( x )=\left[ \phi_{1} ( x^{\prime} ) \phi_{2} ( x^{\prime} ) \cdots\phi_{p} \left( \frac{1} {2} \right) \cdots\phi_{p} \left( \frac{1} {2} \right) \right]
$$
i
中皮
食→P
中
$$
\therefore\lim_{n=1} \det\det\det\cup\smallint\smallint\smallint\log
$$
$$
\begin{matrix} \sin\theta\cos\mu\cos\theta\sin\theta\sin\theta\cos\theta\cos\theta\cos\theta\cos\theta\cos\theta\cos\theta\sin\theta\cos\theta\sin\theta\cos\theta\sin\theta\sin\theta\sin\theta\sin\theta\sin\theta\sin\theta\sin\theta\sin\theta\sin\theta\sin\theta\sin\theta\sin\theta\sin\theta\sin\theta\sin\theta\sin\theta\sin\theta\sin\theta\sin\sin\theta\sin\sin\theta\sin\sin\sin\sin\sin\rightarrow\rightarrow\rightarrow\infty\\ \min\end{matrix}
$$
tria
$$
\Phi( x_{1} ) \in\mathbb{R}^{\mathcal{P}}
$$
i=(


---

## Page 79


$$
\widehat{\theta}^{r}=\arcsin\frac{1} {N} \sum_{i=1}^{r} \left\| \left( \widehat{\theta} \right)^{T} \phi( x )-y_{i} \right\|_{2}^{2}
$$
usheve
$$
= ( \stackrel{\tau} {\Phi} )^{\dagger} \Phi^{\dagger} Y
$$
8 $( \times_{1} )$ 1
 $\Phi( x_{2} )^{7}$ 
 $\phi( x_{n} )$ 
中
$$
\phi( x )=[ 1 \times x^{2} ]
$$
上晨A业中 $\Theta^{\mathrm{T}} \phi\angle x )$  $S$ damc a3
ERM wit、
 $a_{1} x+a_{2} x^{2}+1$ 
$$
\frac{\angle i n e a r \angle M o d e l J} {0} \rho r C l a s \sp{n} f \cdot a b i o n .
$$
D $\sim$ 2
$$
x \in R^{d} , y_{i} \in\{0 , i , 2 \ldots k \}
$$
$$
h o ( x ) : x \rightarrow y
$$


---

## Page 80


$$
\begin{array} {c} {{\displaystyle h o \left( x \right) \;=\; \Theta^{T} x_{\; \;} , \; \; \ldots w i t h \cdots t h o r \mathrm{~ f ~} \chi}} \\ {{\mathrm{s t a n g e} \; \; \mathrm{~ I p u c e ~} o f \; \; h o \left( x \right) \omega o n^{\prime} t b e \gamma.}} \\ \end{array}
$$
$$
h_{\theta} ( x )=\sigma( \theta^{\top} x )
$$
viha $\sigma( t )$ . E
$$
\tan\alpha\tan\alpha\tan
$$
3
$$
\left\{\begin{array} {c c} {{n o l}} & {{c l a \ t s i f i e r}} \\ {{}} & {{}} & {{h \left( x \right)}} \\ \end{array} \right.=\left\{\begin{array} {c c} {{i}} & {{i f}} \\ {{0}} & {{o l \hbar e r \omega_{i} \lambda e}} \\ \end{array} \right.
$$
心
$$
\frac{F_{0} r k-C l a S s C l a S t h f : C a K o n} {x \in R^{\alpha}} y \in\{0 , 1 , 2 \cdots, k-1 \}
$$
$$
h_{\theta} ( x )=\Theta_{\log\cdot} X_{\alpha\times1}
$$
K<a


---

## Page 81


$$
\begin{array} {c} {{\displaystyle\frac{\mathrm{o b j e c b i v e}} {\mathrm{v e c b i v e}} : \tan\frac{1} {\mathrm{v e c b o r}} a k-\dim\epsilon n \gamma i o n a l}} \\ {{\displaystyle\mathrm{r e a l} \tan\theta_{\mathrm{v e c b o r}} i n+0 \log k-\log\theta_{\mathrm{v e c b}}}} \\ \end{array}
$$
 $\mathrm{S u p p o s e}$ vak&`
$$
\det\alpha=a \beta\cot( v )
$$
$$
\log\tan a \left( v \right)_{j}
$$
-/sa(
菜 $\operatorname{e x p} \left( \mathrm{v j} \right)$ 
 $z_{j}=\sin\theta\max( v ) j$ 
$$
\tan z=[ z_{1} z_{2} \ldots z_{k} ]
$$
R台 2=\
$$
\begin{array} {c} {{\det\tan\tan\tan\theta\cos\theta=\tan\cos\theta=\tan\theta\cos\theta}} \\ {{\det\ker\ker\cos\theta\cos\theta\cos\theta}} \\ \end{array}
$$
$$
h_{\theta} ( x )=\operatorname{s i n} f \operatorname{t m a x} ( \theta_{k x d} x_{d x} )
$$
$$
\pi_{h e} \gamma a n g e f \gamma_{h} ( x ) i s \left[ 0 , i \right]^{k}
$$


---

## Page 82

EAN AIAh、gun $\mathrm{T r} \, e$ ohne-ohingabon paslem Cine $\omicron e$ aesoeny
 $\mathrm{I d_{N} e}$  $\dot{\iota} \ddag$ Aie $\upsilon^{<}=a$ yruw gnitdun
Cradeae derCent Netnod
 $\dot{\romannumeral1}$ oietutain $\check{\imath}$  $6 r a d$ aset dacs tb tenbwy Eamltngeo"


---

## Page 83


$$
\tan\tan\log e n c e d o t
$$
$$
\Theta^{t+1} \longleftarrow\Theta^{t}-\alpha\nabla_{\! \theta}^{\widehat{R}} ( \theta) , \vec{\nabla}_{\! \theta}^{\widehat{R}} ( \theta) \epsilon R
$$
$$
\tan5-\tan p y \tan s \tan y \in[ 0 , 1 ]^{k}
$$
{uany) 中车 $\log\operatorname{h o} ( x )$ 

---

## Page 84


$$
\frac{G e n e x a l i z a t i o n i n M L} {O}
$$
$$
\begin{matrix} {\sin\qquad D=\left\{x_{i} , y_{i} \right\}_{i=1}^{\nu} \sim i i d P_{x y}} \\ \end{matrix} .
$$
$$
h^{*} ( x )=\arcsin_{\epsilon z t} \left[ \lim_{R_{y}} L ( h ( x ) , y ) \right]
$$
$$
\widehat{h}^{*} ( x )=\min_{h} \frac{1} {N} \sum_{i=1}^{N} L ( h ( x ) , y )
$$
$$
D_{\mathrm{b r a n}}=\left\{\left( x_{i} , y_{i} \right) \right\}_{i=1}^{i} \sim i i d P_{i y}
$$
u
$$
\{( x_{i} , y_{i} ) \}_{i=1}^{N^{\prime}} \sim i d P_{x_{4}}
$$
t*grm
$$
\mathrm{E R M} \circ n D_{\mathrm{b r a i n}}+\mathrm{o b t a i n} \widehat{h} ( x )
$$
9
$$
\omega_{0} \sin: \nonumber+\omega\omega\omega\omega[ \alpha\widehat{h}^{*} ( x ) \widehat{p e x} / \omega\omega_{0} \omega
$$


---

## Page 85


$$
\cos\sin\sin\alpha=L : h ( x ) \times Y \rightarrow R^{+}
$$
$$
L ( )=\Vert h ( x )-y \Vert_{2}^{2}
$$
$$
x \in{\mathbb{R}}^{\alpha} y \in{\mathbb{R}}
$$
$$
R ( h )=\sum_{k_{4}} \{\vert h \left( x \right)-y \Vert_{2}^{2} \}
$$
$$
h^{\circ} \left( x \right)=\arccos_{h} \left[ R ( h ) \right]
$$
Ft
Piantnin 0 $\circ$ …
$$
h_{\emptyset} ( x ) \cdot t o b e a h y p o t h e t i s
$$
 $l e a r n e d$ 
$$
\begin{matrix} \angle{2} P_{0} b e+\angle{2} d_{2}+\angle{6} a c t a n g \\ \natural\angle{2} a \natural{2} a d a t a n g / d_{m} \end{matrix}
$$


---

## Page 86


n $\mathrm{h y p o}$ hesr-a Dejne
 $\bar{\mathrm{h}} \left( x \right)$  $\mathsf{P_{D}}$ ini
amaiteAaerge A oeNdcant
8
E
 $\mathrm{P_{x}}_{Y}$ 
$$
= \lim_{P_{0}} \lim_{x_{y}}^{E} \vert h_{0} ( x )-h ( x )+h ( x )-y \vert\vert_{2}^{2} \vert
$$
=O


---

## Page 87


$$
R ( h )=\lim_{P_{0}} \lim_{P_{x y}} [ ( h_{0} ( x )-\overline{{h}} ( x ) )^{2} ]+\sum_{P_{x y}} ( \overline{{h}} ( x )-y )^{2}
$$
onsder
 $\mathrm{P x y}$ n5 二
 $\mathrm{P x}_{9}$ S $\mathcal{P}_{\mathcal{A}}$ t和oto S
 $\mathrm{P_{x}}_{7}$ 
ex
 $\mathrm{P}_{x y}$ 。 AT :.c1 CAT $\mathcal{P}_{x_{y}}$ E REE Fioy
$$
+ \lim_{P_{x} y} \{h^{2} ( x )-y \}^{2} \} I n \geq0
$$
huesle
re

---

## Page 88


$$
\frac{A n a v y z : s} {o f} B a s=E \frac{v a n a n c e} {} .
$$
Conaderthe a $\omicron\o$ bavn.
55u-+k
$$
\rho\cup a n \tan b e t
$$
坛
 $e f b e c t o f$ Aarhulahasce $^o t$  $\mathrm{h y p o}$ hans Cus
 $v a n \cdot a n c e$ :
 $\ref{P D_{D}}$  $\omicron,_{7}$ 5 $h_{0} ( x )-h ( x )$ measuverHke 沪E
中 $\imath_{\bigodot}$ wboan ke thn $\mathrm{C h o i C e}$  $d a t a+e t$ 


---

## Page 89


$$
\begin{array} {c c} {{B_{i a 1}}} & {{\cal E}} \\ \end{array} \lor\begin{array} {c} {{\scriptstyle\mathrm{v a n i a n c e ~ i n ~ p r a c h c e}}} \\ \end{array}
$$
Eramplc
$$
{} \times\, \in\, \mathbb{R} \, , \qquad{} \mathcal{y} \, \in\mathbb{R} \, ,
$$
y
$$
y=a x^{2}+b+\epsilon\in N ( 0 , I )
$$
 $a , b \in\mathbb{R}^{+}$ 
5
1 $\boldsymbol{\sigma}$ 
O
O
r
o
）×
 $\mathrm{h , ( x )}$ EAK
 $\mathrm{B i a s}$ 心楼 $g$ ^
 $\lor\alpha\Upsilon$ 0
$$
2 ) : h_{i} \cdot( x_{i} ) :=y_{i} : \times\times_{i} : \mathrm{l o o k u p t a b l e}
$$
湘 $h_{2} ( x )=0$ ok $\omega i J e$ 
人
wn： $\mathrm{h i g h}$ 


---

## Page 90


$$
P_{r a g m a b c} \omega a y q f \tan a d i n g B E i v
$$
三 $\tau_{y p} \cdot\arccos$ 出心
" $8 0 N o w \sin g$  $\alpha$ mnanse tny
re sam rue
 $\xi_{\tau\Upsilon\circ\Upsilon}$  $\mathbf{B : a I}$ 个 n $\xi_{\gamma n o r} o n$ 
sa< L NacT $D_{v a l} \vert D_{t e c o t}$ 
y $\log_{x=6} \tan_{y}$ 
tay
>
=^mdLAOomgany'
 $\mathrm{T_{i}}_{\mathrm{P}}$ ANywt $\omega\triangleleft H$ aigtihan -hat opuradeai the $\omega\omega$ bar & high var $\left( \arccos t \right) r e g o n$ o ontpamategtin

---

## Page 91


$$
\frac{E x p e c t a t i o n \dots M a x i m i g a t i o n \dots A l g o n t h m} {0} .
$$
$$
F_{\theta} ( q )=\sum_{q ( z )} \log\frac{P_{\theta} ( x , z )} {q ( z )}
$$
3 $\mathrm{( q )}$ x
$$
\pi e o p \tan a l q ( z )=P_{\theta} ( z ) x
$$
supreuesaluaps Mone o" $q ( z )$ :R(e be dauaired at 2m
$$
q^{t+1} \left( 2 \right)=P_{\theta^{t}} \left( 2 1 x \right)
$$
$$
\frac{\rho m p u \tan2} {} F_{\theta} ( q^{t+1} )=\lim_{q^{t+1}} \log_{\frac{P_{\theta} ( x_{1} z )} {q^{t+1}}}
$$


---

## Page 92

"+a^3 $F_{\mathrm{i n}} \alpha$ 8"=ngp+ $( q^{t+1} )$ End $\mathbb{F}_{\mathrm{O N}}$ 
Ay
全
S


---

## Page 93

X

---

## Page 94


$$
\frac{C o n v e x g e n i c} {j}+\dagger h e E M a l g o x t h m .
$$
$$
\frac{\cos\mu} {\alpha}=\frac{\mu} {\alpha} \frac{\mu} {\alpha} \frac{\mu} {\alpha} \frac{\mu} {\alpha} \frac{\mu} {\alpha} \frac{\mu} {\alpha} \frac{\mu} {\alpha} \frac{\mu} {\alpha} \frac{\mu} {\alpha} \frac{\mu} {\alpha} \frac{\mu} {\alpha} \frac{\mu} {\alpha} \frac{\mu} {\alpha} \frac{\mu} {\alpha} \frac{\mu} {\alpha} {\alpha} \frac{\mu} {\alpha} \frac{\mu} {\alpha} {\alpha} \frac{\mu} {\alpha} {\alpha} \frac{\mu} {\alpha} {\alpha} \frac{\mu} {\alpha} {\alpha} \frac{\mu} {\alpha} {\alpha} \frac{\mu} {\alpha} {\alpha} \frac{\mu} {\alpha} {\alpha} \frac{\mu} {\alpha} {\alpha} \frac{\mu} {\alpha} {\alpha} \frac{\mu} {\alpha} {\alpha} \frac{\mu} {\alpha} {\alpha} \frac{\mu} {\alpha} {\beta} \frac{\mu} {\beta} {\beta} \frac{\mu} {\beta} {\beta} \frac{\mu} {\beta} .}
$$
1 $\mathsf{M}$  $\triangleleft\ss_{\mathrm{P}}$ lonsde< 公
 $\frac{\mathrm{p_{0 0 1}}} {1 2}$ 
$$
q^{t+1} ( z )=p_{o^{t}} ( z | x )
$$
 $\mp_{\theta} ( q^{t+1} )$ 
火公 A
$$
\lambda( 0^{t} )=\sqrt{F} \left( q^{t+} \right)
$$
日
$$
:=\sqrt{-\theta} ( q^{t-1} ) \vert_{\theta=0^{t-1}} > F_{\theta} ( q^{t-1} ) \vert_{\theta=0^{t}}
$$
 $\Im\circ$ TTa i
$$
b y \mathrm{T e n i e n ' s i n e q u a l i t y}
$$
→ $\lambda( \theta^{t+1} ) \geq\lambda( \theta^{t} )$  $\mathbb{Z}$ 


---

## Page 95


1
$$
E M \alpha_{n} g_{0} n^{H_{n m}} \rho^{r} G M M I \cdots
$$
$$
P_{\theta} ( x )=\sum_{2} P_{\theta} ( x , z )
$$
$$
= \sum_{j} \alpha_{j} N ( x ; \mu_{i} , \Sigma_{j} )
$$
$$
= \sum_{2} P_{0} ( 2 ) \cdot P_{0} ( x / 2 )
$$
V
P·UK $J$  $P_{\theta} ( 2 )=\alpha_{j}$ Bxeng
9=
3
$$
0 \le\alpha\leq1 , \mu\in R^{d} , \xi\in R
$$
Enar $\mathrm{G M M}$ 
nsldge S


---

## Page 96

W
$$
q^{t+1} ( z )=P_{o} ( z | x )
$$
-tiy
$$
= \frac{P_{\theta^{t}} ( x 1 2 ) P_{\theta^{t}} ( 2 )} {\sum P_{\theta^{t}} ( x 1 2 ) P_{\theta^{t}} ( 2 )}
$$
V
banon
nA
S $\upsilon\circ q$ .
$$
\lim_{P_{\theta}^{t} \left( 2 1 x \right)} \log\frac{P_{\theta} \left( x_{1} z \right)} {P_{\theta}^{t \left( 2 1 x \right)}}
$$


---

## Page 97


$$
= \sum_{j=1}^{m} \left[ \log\frac{P_{\theta} ( x+2 ) P_{\theta} ( z )} {P_{\theta} ( z+x )} \right] P_{\theta} ( z-j )
$$
$$
F_{\theta} ( q^{t r} )=\sum_{j=1}^{n} \log\left( \begin{matrix} {N ( x , \mu_{j} , \Sigma_{j} ) . \alpha_{j}} \\ \end{matrix} \right) P_{\theta} ( z-\mu)
$$
Y N
\


---

## Page 98


$$
\frac{B a y e \tan H e t h o d s f o r d e n s} {j} t_{j} E_{s} t_{i m a t i o n}
$$
 $R e c a l l$ ，in $\mathsf{M L}$ Csknateh $\mathtt{e}$ paramter <ct(0) $1 3$ onsidedtierminitic.
中 $\b{b e}$ tmu A
 $\omicron\circ\mathrm{t}$ aMLoare nry $\varepsilon_{g}$ ' $\Theta_{\mathrm{\scriptsize\mu c}}$ -o $\ell^{\prime}$  $\mathbf{I}$  $\mathrm{h a 1}$ rin $\Re v$ 太 $\tan e s$ Bee
元
Te aboeAkmare $\dot{\omicron}$  $\mathrm{n \circ\mathrm{t}}$ Eashmal
 $\grave{\omicron}$ 
$$
\begin{array} {c} {{\mathrm{B a y e ~ I : a n ~ E x h m a l i o n ~ : ~ I n ~ c o r p o r a l i e . . ~ f h e}}} \\ {{\mathrm{p m a r b e l i g ~ I ~ I n f o r m a l i o n ~ a b o u t ~ H e ~ p o r m .}}} \\ \end{array}
$$


---

## Page 99


$$
\begin{array} {c} {{{\cal A}_{I I u m e} \quad\mathrm{H h e} \quad\mathrm{p a r a m e F e r} \quad\theta\quad\mathrm{H o} \quad b e \quad\mathrm{a R v}}} \\ \hline{{\begin{array} {c} {{g_{\mathrm{L M o d e l}} \quad\mathrm{H h e} \quad\mathrm{p a r a m e F e r} \quad\theta\quad\mathrm{i n \quad i f ~ s \quad d_{i} ~ d r i b u l i o n .}}} \\ \end{array}}} \\ \end{array}
$$
$$
\mathrm{I n ~ t h e ~ p r i v i o u ~ I ~ e x a m p l e ~ p e r ~ B e r ~ ,} R v .
$$
 $\mathrm{P}_{\Theta} ( \vartheta)$  $\flat_{\theta} \left( \theta\right)$  $\circ$  $\omicron\cdot5$ 公
 $\imath$ 
A
 $I \sim\mathrm{p p o s e}$ A a
poen pron
CmruR
$$
P_{\theta} ( x ) > \propto P_{\theta} ( 0 ) \cdot P_{\theta} ( x ) \theta
$$
 $\mathrm{H o} \cup$ a.
 $\mathrm{M A p}$ Take the node
EAimude
$$
0 1 P_{\theta1 \times} a s t h e e i k m a t e f o r \theta
$$


---

## Page 100

Cosiogse qwe Te spie n e
$$
\begin{array} {l} {{\mathrm{H a t} \quad\mathrm{e n} \, \mathrm{I u r e s} \quad\mathrm{H a r} \quad\mathrm{f i c} \quad\mathrm{f o r} \, \mathrm{I e} \, \mathrm{i f} \quad\mathrm{h a v e} \quad\mathrm{H} \, \mathrm{e}}} \\ {{\mathrm{f a m e} \quad\mathrm{d i s t r} \, \mathrm{i b u h o n a l} \quad\mathrm{f o r m} \quad\mathrm{a r} \quad\mathrm{i f} \, \mathrm{i f} \, \mathrm{i c} \mathrm{f} .}} \\ \end{array}
$$
SanN- : NAP Stimate e. Bnouhe
 $\Gamma( \alpha)=\Gamma( \beta)$ Y
$$
\times\sim B e r ( \theta)
$$
thestd gamma $\flat^{\frown}$ 
C
S
天R ese- Coscn beama
$$
P_{\theta1 x} \propto P_{\times1 \theta} \cdot P_{\theta}
$$
$$
P_{\theta1 x} \cdots\propto\left[ \theta^{*} . \left( \omega-\theta\right)^{1-x} \right] . \theta^{\alpha-1} . \left( \omega-\theta\right)^{\beta-1}
$$


---

## Page 101

Ana TR
atde in kenida y $\mathrm{P_{\Theta I x}}$ 
in-tne chose Cude
 $\Theta_{\mathrm{N L}}$ . $\wedge$ 
$$
\Theta_{m A P}=\frac{\sum_{i=1}^{n} x_{i}+\alpha-1} {n+\alpha+\beta-2}
$$
Y
2Ki A
i<（
A
udl $D e n-8 i E_{y}$ Ezmaksn 三 $\ddagger_{x}$ 油 $\mathrm{H_{h e}}$ 州 $\alpha_{\mathrm{n y}}$ 三 $\pi_{e-s c}$ Hk $\omega|_{0}$ whv"y $\overleftarrow{\imath}$ S $\\#_{\mathrm{h t}}$ kq $\delta^{\frown}$ n mtay p yocm aAtupbon
 $\begin{matrix} \varsigma u p p o r t \\$ &， $\mathsf{P_{x}}$ ^"raym
Lk R denote a


---

## Page 102

FaAOAL $\aleph^{e q \circ n}$  $\mathrm{\widehat{R}}$ .
 $\mathrm{p o i s t}$ mry业会
$$
P=\int P_{x} \left( x \right) d x
$$
8 $\underleftrightarrow{\omicron\jmath}$ 一 $\ \%^{\omicron}$ 8
pudmtR $\vec{\sigma}$ 8 $f a l u n g$ swppose $\mathrm{h a v e}$ n ve
&suwsn $b^{\infty} \sim$ 2 noaelsa -, theuct ipad an .Bek $\mathrm{R_{\Lambda}}$ shmate
 $\mathrm{M a x}$ 
 $\mathsf{k}$ 
 $\mathsf{n}$ 
 $\omega h e v e$ （t日he华等wr mnite-letgan
 $\mathsf{K}$ 生pan
 $\cap$ 


---

## Page 103

1R：tmuMscangh
- $\mathtt{K}$ -- gEV whwe
 $\sim$ 
√ $\vdots$ t. rwne th eym
 $\Rightarrow$ 
$$
p_{x} \left( x \right)=\frac{k} {n \cdot v}
$$
 $\vdash$ 元
}*人 $\mathcal{E}$ Kn bx $\mathcal{E}$ R $d e n-1 i t_{y}$ 
CyA
PoyeudAdo Estiah:
3 $\mathtt{V}$ 20 $\mathsf{K}$ ,to eoclmale dentiy-m $\omega e$ a $\mathtt{t o}$ enamate the
 $\omega a n t$ 
 $\det\sin\theta_{1} \tan\theta_{1}$  $\times$ 
生R $\log\tanh h .$ 
-ypunhe eunA $\times$ 


---

## Page 104


$$
\mathrm{T o} \quad\mathrm{c o u n t} \quad\quad k \quad, \quad\mathrm{d e f i n e} \quad\mathrm{a} \quad\mathrm{w i n d o w} \quad j n
$$
$$
\Phi\left( \begin{array} {c} {{u}} \\ \end{array} \right)=\left\{\begin{array} {c c} {{i \ \}} & {{i f \ \ \left| u_{j} \right| \leq y_{2} \ \ \ \ j=i \cdots d}} \\ {{0}} & {{\ \mathrm{o t h e v \, \omega i \, I \, C}}} \\ \end{array} \right.
$$
 $\mathrm{P_{x}} ( \times)$ x-X: MananLa
H
 $\not\exists$ nmsaierd
Arond
\
 $\flat_{\mathrm{x}} ( \mathrm{x} )$ Ex
卜
 $n \cdot h^{\alpha}$ 
$$
\begin{array} {c} {{\phi( u ) \cdot\tan b e}} & {{m a d e}} & {{\pm m o o t h e r \cdot b y}} \\ {{c h o o t i n g}} & {{\phi( u )}} & {{=}} & {{\exp\left(-\left\| u-u_{0} \right\|_{i}^{2} \right)}} \\ \end{array}
$$


---

## Page 105

 $N e a r e-s t$ MAeshmueEiamdio
 $\dot{\aleph}^{\infty}$ Ewnmrpnetuletin
 $\mathcal{E}$ y。
$$
P_{x} \left( x \right)=\frac{k} {n v .}
$$
Ssemle us wse : maluag
 $d e n s i t y$ A力 m
sNNeGrsst
Eahmade
anrpm m $C l a_{3} s$  $\mathcal{A}$ eaple
wc lateeNdume $\cup$ aroun $\alpha$  $\times$  $\mathcal{E}$ CAmg=KkBdxpe
 $\mathrm{L e t}$ m me $\mathrm{t o}$  $\omega i l h_{i n}$ te vouwne $\vee$ oripendng
the thCas.
$$
\sum_{i=1}^{m} K_{i}=K
$$


---

## Page 106


$$
N N e \sin\frac{1} {d} e f o r p ( x , y_{1} )=\frac{k_{i}} {n v}
$$
$$
T_{0} \cos\imath b u c t a B a y e^{\prime} I \cos\jmath^{\prime} t v
$$
$$
\begin{matrix} p \left( y_{i} \mid x \right)=\frac{p \left( x , y_{i} \right)} {\sum_{y} p \left( x , y_{i} \right)}
$$
Y
?iH=
$$
= \frac{k_{1} / n v} {\sum_{i=1}^{m} \frac{k_{i}} {N v}}
$$
。
 $\vdash_{\mathsf{B}} ( \times)$ 天
$$
P \left( y_{i} | x \right) > \phi\left( y_{i} | x \right)
$$
$$
\frac{k i} {k} > \frac{k j} {k}
$$
$$
\mathrm{i} \zeta\mathrm{k_{i} > K_{j}}
$$
$$
k-n \mathrm{e a r e} \mathrm{t} \mathrm{n e a g h b o u r} \mathrm{c l a H i t i o n}
$$


---

## Page 107


$$
M_{i > C} t_{u v c} D_{c n h} t_{i e I} E H .
$$
 $A$ maxhue dainy tndl.
M shu &t^
$$
p_{\theta} \left( \mathrm{v} \right)=\sum_{j=1} \alpha_{j} p_{\theta_{j}} \left( \mathrm{v} \right)
$$
dantiy gn
04e<;41&7d1 $G_{a v 1} \sin$ rucrve dunsty 力 3,IH 力
$$
M_{\arcsin u m} h k e / i h o o d E_{1} t
$$
 $\upsilon$ . $M u_{2} t_{u v c}$ 
 $R e c a U$  $\sim$ " $e=\left\{\alpha_{j} , H_{i} , \Sigma_{j} \right\}$  $j=1$ 
densty
$$
\sum_{i=1}^{N} \log p_{\theta} ( v i )=\sum_{i=1}^{N} \log\sum_{j=1}^{\mu} \alpha_{j} . {\cal N} ( v_{i} , \mu_{j} , \Sigma_{j} \Biggr)
$$


---

## Page 108


$$
L a t c n t v a n i a b l e H o d e l s .
$$
 $G i v e n$ 
$$
D=\left\{\begin{matrix} x_{1} , x_{2} \ldots x_{n} \\ x_{i} \in R^{\alpha} \end{matrix} \right\} \sim i i i d P_{x}
$$
龙m $\omega$  $\mathtt{R} \backslash$  $\mathrm{i n t_{6}}$ h。 $^1 \gg$ rim
 $\mathbf{i} \ll$ n-Ob4evued MiaaeA atent RN tuhcemain $\operatorname{U y}$ 冷 2
VxDg
Kiswit o ax$too tti ouon X $\ \%$ ZaepoArmadto $c o r v e l a t e d$ Z ton be goimusus dutcecke
nth sWi OenNket vor- noda i aegrnd n t 3 $p_{\theta} \left( x , z \right)$  $` ` \ddag$ z $\textsf{i} \supset$ "
Mek·c. $\operatorname{c o n t i}$  $\mathrm{M L}$ sswaeon gp ks Nohe Huad. $\mathrm{F_{o x}}$ 。、 $1 a t \cdot v a n a b l e$ nodel bot. the $\mod e \ell$ parometext& the dentity ovev the $\iota_{\alpha} \iota$ var noade to te ertmaled


---

## Page 109


$$
\lambda( \theta)=\log p_{\theta} ( x )
$$
$$
\mathrm{w e} \quad\mathrm{h a v e}+\mathrm{h e} \quad\mathrm{U i k e l i h o o d} \quad\mathrm{f u n c t i o n} \quad\mathrm{U ( e )} .
$$
$$
= \log_{z} P_{\theta} ( x , z )
$$
m $\mathtt{q} ( z )$ 中N9
$$
u ( 0 )=\log\sum_{2} \sum_{\rho} ( x_{1} , 2 ) \cdot\frac{q ( 2 )} {q ( 2 )}
$$
-"5ZMD $9 ( 2 )$ j
5 $\omega q$  $\textit{l}$ AO
 $\sim\! q$  $\hat{\jmath}^{g}$ "A= $q ( z )$ Oas
E $\mathrm{L o � q}$ a0
e) 2活 $( \mathtt{a} )$ 


---

## Page 110

 $\mathrm{F_{\theta} ( q )}$ mrsasarauihe it ko $1 \omega m \sim2 p$  $\cup_{9}$ - $\varlimsup$ oen
 $\widehat{F_{\Theta}} \left( \mathrm{q} \right)$  $\mathbf{i} \triangleleft$  $\alpha$  $l_{0} \omega e r$ bound on $\mathcal{A} ( \theta)$ 
rg udlided sune Easna urEeye 力 $\mathrm{N e} \, \omega$ inghrhag
 $\Theta^{\star}$ ·Cyh
$$
\log\alpha_{i j}=\alpha_{i j}
$$
 $F_{\theta} ( q )$ wiepe 4 $\underbrace{\varphi v e s+\varphi n}$ nMhtatadloue ey s.to ou
mw hc Be gy
onsder
$$
U ( \theta)-F_{\theta} ( q )=\log P_{\theta} ( x )-\sum_{2} q ( z ) \log\frac{P_{\theta} ( x , z )} {q ( z )}
$$


---

## Page 111


$$
= \log P_{\theta} \left( x \right)-\sum_{z} q \left( z \right) \log P_{\theta} \left( x \right) \cdot P_{\theta} \left( z \right)
$$
$$
= \log p_{\theta} ( x )-\log p_{\theta} ( x )-\sum q ( z ) \log\frac{p_{\theta} ( z | x )} {q ( z )}
$$
$$
= \sum_{2} q ( 2 ) \log\frac{q ( 2 )} {P_{\theta} ( 2 ) x}
$$
Y
$$
D_{k L} ( q ( 2 ) \parallel P_{0} ( 2 1 x )
$$
$$
l ( 0 )-F_{0} ( q )
$$
$$
\tan F o ( q )=\ln( 0 ) , 6 y
$$
Contbw thtn
 $\circ$ 
$$
U ( 0 )=F_{0} ( q )=0 H f D \times1 ( q ( 2 ) H P_{0} ( 2 1 x ) )
$$
1
$$
\Rightarrow q^{*} ( z )=P_{\theta} ( z | x )
$$
 $= \omicron$ 
$$
n i t e x a r i v e A i g o n t h m f o r o p t i m \geq g A i n g
$$
f $\omicron$ t= be Mnwucgnce
：


---

## Page 112


$$
\tan\rho u t e q^{t} ( z )=p_{\theta^{t}} ( z 1 x )
$$
Ethndeun m
N $\left( q^{\mathrm{t}} \right)$ 
pmur
女力 $T_{0} \uparrow h_{0} w$ n $g u a \tan e c-$ 
{6


---

## Page 113


$$
\frac{D_{\pm n} \sin t_{1}} {1} 8 \tan a \tan
$$
，
$$
h^{\prime} ( x )=\arg\max_{y t} \left[ \oint_{x y} L ( h ( x ) , y ) \right]
$$
$$
f^{o \star\cdots\cdots} \mathrm{D-i} \quad\mathrm{L o o s s} \quad, \quad h_{\cal B}^{\bullet} ( y )=\left\{\begin{array} {c c} {{i \quad\quad i \quad i \quad p_{y=1 x}}} & {{> P_{y=o l} ,}} \\ {{0 \quad\mathrm{O H h e r w i s e}}} & {{}} \\ \end{array} \right.
$$
 $\varepsilon_{q}$  $G_{N} \sim e$ A
生品 tndegy $C l a+2 y=e_{2}$ 
S CAasandsod
s
$$
M_{a k i m a m} b i k e l u h o o d ( M_{a n i m a m} k L )=s k i m a k o n
$$
$$
v=\{v_{1} , v_{2} , v_{n} \} \sim i d R ,
$$
S4kd
$$
\begin{array} {c} {{\mathrm{s t a r t .} \omega\forall h \beta_{\theta} \in\xi=\arg_{\theta}^{n a n} D k_{i} \left( P_{i} \parallel P_{\theta} \right)}} \\ {{\equiv\arg_{\theta}^{m a x} \left( \frac{1} {N} \sum_{i=1}^{N} \log_{\theta} p_{\theta} \left( v_{i} \right) \right)}} \\ {{v_{i} \sim_{i} i d \beta}} \\ \end{array}
$$


---

## Page 114

Examgles oL ML EAhmakbon i
D:nnN5 $\mathrm{H o d e l}$ 
"
$$
e h a v e v_{1} v_{2} \ldots v_{N} \sim i i d P_{v}
$$
$$
P_{o} ( v )=\frac{1} {( 2 \pi)^{\frac{1} {1 2}} | \pm1 |^{1 2}} \exp\left\{\begin{matrix}-1-( v-\theta)^{\frac{1} {2}} ( v-\theta) \\ 0 \end{matrix} \right.
$$
$$
\cos\alpha_{2} \alpha_{2}=\frac{1} {\alpha_{1}} \sum_{\overline{v}=1}^{N} \log P_{2} ( v_{1} )
$$
$$
\begin{matrix} \alpha=\frac{1} {N} \\ \end{matrix} \begin{matrix} \sum\\ \end{matrix} \vert\vert\begin{matrix} v_{1}-\theta\end{matrix} \vert\begin{matrix} v_{2}^{2}=\frac{\partial\left( \sum( v_{1}-\theta) \right)} {\partial\theta} \\ =\Sigma2 ( v_{1}-\theta) \end{matrix}
$$
空 $\lor\mathfrak{r}$ 
$$
\mu_{a} x \cdot b x e . z_{\tan} b e
$$
 $\mathbf{\dot{\omega}}=\mathbf{1}$ 
 $\Sigma_{\mathrm{x a m p i c}}$ : Daale Randaon Manale
1
$$
D=\{v_{1} , v_{2} , \ldots, v_{N} \} \sim
$$
=
$$
v i \in\{a_{1} , a_{2} \ldots, a_{n} \} w i t h p r o b . \{p_{1} , P_{2} , p_{n} \}
$$


---

## Page 115


$$
p a r a m e t a r \theta=\{p_{1} , P_{2} \ldots, p_{n} \}
$$
 $\mathbf{\bar{o}}$ srbrust He mons tpunton yr h $\mathtt{R} \mathtt{Y} \, ,$ we trd on-hob sepseaenbabin
ons- hob -<Usesedason
 $f o l i o w s^{\circ}=$ 
ianana
 $z_{i}=\left[ \begin{matrix} 3_{i}^{1}-3_{i}^{2} \end{matrix} \right.$ 
$$
3_{1} \in\{0 , 1 \}
$$
$$
R_{i}^{j}=\left\{\begin{matrix} \arctan\angle V_{i} v_{i}=a \\ 0 \cos\angle V_{i} \end{matrix} \right.
$$
\
 $\varepsilon_{c}$ 营
$$
\begin{matrix} D=\{2 , 3 , u , 1 , 6 \cdots\} \\ v_{1} \in\{1 , 2 , 3 , u , 5 , 6 \} \end{matrix}
$$
$$
z_{1}=[ 0 1 0 0 0 0 ]
$$
 $z_{2}=[ 0 0 1 0 0 0 ]$ 
E
$$
\begin{matrix} \tan\alpha t a D=\{z_{1} , z_{2} \ldots, z_{n} \} \\ z_{1} \in\{0 , 1 \}^{m} \end{matrix}
$$


---

## Page 116


$$
P_{0} \left( v_{i}=a_{j} \right)=P_{i}^{3 i} \times P_{2}^{3 i} \cdot\times P_{n}^{3 i}
$$
: $\mathbf{j}=\mathbf{i}$ ;o[u%nE
 $j^{1 1} \sb{\rho\circ3}$ °
-o
 $P_{\theta} ( \vee_{i} )$ :
-oagaga^R
K 二年物中
了
SA alai kiciac mion
;=| M
&高
$$
N e w \ {\mathrm{o p j e c k v e}} .
$$
$$
\begin{matrix} \arctan\left( \begin{matrix} \end{matrix} \right)=\left( \begin{matrix} \begin{matrix} \end{matrix} \right) .=\left( \begin{matrix} \begin{matrix} \end{matrix} \right)=\left( \begin{matrix} \end{matrix} \right)=\left( \begin{matrix} \end{matrix} \right)=\left( \begin{matrix} \end{matrix} \right)=\left( \begin{matrix} \end{matrix} \right)=\left( \begin{matrix} \end{matrix} \right)=\left( \begin{matrix} \end{matrix} \right)=\left( \begin{matrix} \end{matrix} \right)=\left( \begin{matrix} \end{matrix} \right)=\left( \begin{matrix} \end{matrix} \right)=\left( \begin{matrix} \end{matrix} \right)=( \begin{matrix} \end{matrix}
$$


---

## Page 117


$$
\begin{array} {c} {{\omega_{\pi} \hbar n g+h e : \lim_{i} \log\tanh_{i} \hspace{-2 c m} \mathrm{f i n g-a n} ,}} \\ {{\frac{1} {N} \sum_{i=1}^{n} \sum_{j=1}^{n} \tilde{z}_{i}^{\dagger} \ \log_{P_{j}} \hspace{-2 c m}+\ n \left( \sum_{j=1}^{n} p_{j}-1 \right)}} \\ {{\hspace{-2 c m}+\hspace{-2 c m} \left( p_{j}-i \right) \hspace{-2 c m}+\hspace{-2 c m} \left( p_{j}-i \right) \hspace{-2 c m}+\hspace{-2 c m} \left(-p_{j} \right)}} \\ \end{array}
$$
$$
\begin{matrix} b_{f} t e r \sin g+\hbar e a b o v e=b o b l e m_{-} \\
$$
7
停·出香
 $3$ N
MeantEatrokintaer WTne
dutrbubons
 $\mathrm{P_{\star}}$ 
-3×


---

## Page 118

mat-naldentae aenode cuel-eshmalal $\omega\textrm{i t h}$ ud-modal modes.
$$
\begin{array} {c c} {{\mathrm{s o l l} :}} & {{\mathrm{T r} y}} \\ {{}} & {{\displaystyle\vphantom{\mathrm{T r}} \Big> a}} \\ \end{array} \quad\mathrm{m u l t i-m o d a l} \quad\mathrm{m o d e l} \ .
$$
 $M i x t_{u v e}$  $d e n-2 : t_{4}$ --act*un
 $\log{p o s e}$ 1 $\mathcal{J}=\i$ R^， 公力
 $\epsilon$ 
 $D_{e l i n e}$  $\alpha$ 交 $E_{u v c}$ 中
M
 $\mathrm{P_{\theta} \left( v \right)}$ 
 $\alpha_{i} \in[ 0 , i ]$ Neuntyuntn
Cnsnayaea sl (on
T 9
 $G M M_{3}$ ereunveraL deasiy
 $a \o\ddag$ aumir $\S^{\omicron\vee}$ K
 $\mathrm{P_{e}}$ 


---

## Page 119


$$
D_{i s t n} b_{u} b_{i n a l} D_{i v e r g e n c e .}
$$
sksecke sun a poveaeatihunbon $f^{a n c \tan\phi}$ 
dekncd on the dame -sample opace
dte ewhiy 'ctane" $b / \omega$  $\epsilon m$ 6men an ontAeE 全 suprex-s ORh. t $\delta\omega r p h i I a l$ nanuld ydhn s aegtgya yun
$$
E [-\log P ( x_{1} ) ]
$$
| $\frac{E} {\mathsf{P_{\kappa}}}$  $f ( x )=\sum P_{x} . f ( x )$ 
$$
H ( P_{x} )=\sum_{i}-P ( x ) \log P ( x )=\log i n_{1} \max_{2} \max_{2}
$$
$$
\begin{array} {c c} {{\mathrm{s u p p o s e} \quad\quad\displaystyle{P_{x}} \quad\quad\displaystyle{Q_{x}} \quad\quad\displaystyle{Q_{x}} \quad\mathrm{a r e} \quad\mathrm{t w o} \quad d_{i \raise1 p t \mathrm{s i b u l i o n s} \ldots0}}} \\ {{\quad\quad\quad\quad\displaystyle{\mathrm{s u p}} \quad\quad\quad\displaystyle{\mathrm{s u p}} \quad\quad\mathrm{J a u c e} .}} \\ \end{array}
$$


---

## Page 120


0%y;*
$$
\begin{array} {l} {{\displaystyle\frac{E} {P_{x}}-\log\varphi\left( x \right)=H \left( p , q \right)}} \\ {{\displaystyle\rightarrow\cos\varphi-\tan\varphi y .}} \\ \end{array}
$$
(onsde<
$$
- \textsf{H} \left( \overset{\Bigr)} {\mathbb{P}}+\textsf{H} ( \overset{\Bigr} {\mathbb{P}} , \overset{\Bigr} {\mathbb{Q}} \right)
$$
$$
+ \overline{{z}} \natural\left( x \right) \log\left( p \left( x \right) \right)
$$
9
 $D_{k l} \bigg( \mathbb{P} \backslash\mathbb{Q} \bigg)$ 心 $Q_{x} ( x )$ K- &wcsgpmnL
$$
\pi\sin\omega: D_{K L} ( D N Q ) \neq D_{K L} ( Q N P )
$$
Dy20
$$
D_{k l}=0 \quad\mathrm{i f} f \quad D=Q
$$
$$
D_{k L} ( P_{x} \parallel q_{x} )=\int_{x} p_{x} ( x ) \log\frac{p_{x} ( x )} {Q_{x} ( x )} d x
$$


---

## Page 121


$$
S_{u p p o s e \ldots\omega e \ldots\mathrm{h a v e \ldots N} \ldots\mathrm{J_{a m p l e} s \ldots d_{r a \, \omega\, n} \ldots l i d}}
$$
$$
D=\{v_{i} , v_{2} , v_{3} , v_{n} \} \sim\lim_{i i d} P_{i}
$$
 $G \circ a \backslash$ ·édhatke $\mathtt{P}$ ryun D.
Anunk $\mathtt{P}_{\Theta} \left( \mathtt{v} \right)$  $\mathsf{P_{8} ( v )}$  $\mathrm{d v}$ lom pure $g^{m i n}$ 店 $\alpha\vee$ 2
O $\omega$ AR
-
$$
\int p_{v} ( v ) \log P_{\theta} ( v ) d v
$$
B
$$
= \arg_{0} \max-\prod_{v} p_{v} \left( v \right) \log P_{\theta} \left( v \right) d v
$$
"m{s55g%o ton et rtlenge wnwen

---

## Page 122


$$
E \log P_{\theta} ( v ) \approx\frac{1} {N} \sum_{i=1}^{N} \log P_{\theta} ( v_{i} )
$$
$$
\small{\mathrm{i s h e r e} v i \sim\mathrm{i i d} P_{v}}
$$
$$
\frac{1} {N} \sum_{i=1}^{N} \log P_{\theta} \left( v : \right) \longrightarrow\sum_{P_{i} \leftarrow} \log P_{\theta} \left( v \right)
$$
A S
 $\Theta$ 3
R
 $\vee: \sim\det P_{v}$ odashontoe
$$
= \arcsin_{\theta} \left[ \frac{1} {N} \sum_{i=1}^{N} \log P_{\theta} ( v_{i} ) \right]
$$
$$
P_{\theta} ( v : )=l a k e l i h o o d \rho o f v u n d e v \rho_{\theta}
$$
$$
\equiv m o t i m u m b u l a k e l i h o o d e s t m a t o r
$$


---

## Page 123


$$
J_{0 i n t \ldots l i k e l u h o o d \ldots o p H e \ldots d a t a}
$$
N tlid
$$
\log_{\theta} ( D )=\prod_{\theta\in( v_{i} )}
$$
$$
\widehat{l}_{\theta} ( D )=\sum_{i=1}^{N} \log P_{\theta} ( v_{i} )
$$
 $\aleph$ 
$$
\theta^{*}=\arccos_{\theta}^{m a x}
$$
2
Ciae eAAnecrcrn
pytEY
$$
\begin{matrix} \left( \begin{matrix} a \end{matrix} \right)+0 \infty+0 \left( \begin{matrix} \right) 0 \infty0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
$$
$$
\begin{array} {l c r} {P_{\theta}} & {{\max\cup}} & {{\cup}} \\ {P_{\theta}} & {{\cup}} & {{\cup}} \\ {\tan\cup\log n} & {{\arg\log\log\log\tan\tan\tan\log}} & {{\cup}} \\ \end{array}
$$
n wund


---

## Page 124


$$
\begin{array} {c} {{\mathcal{E}_{g} : \quad\mathcal{A}_{I I u m e} P_{\theta} ( ) \sim N \left( \begin{array} {c} {{\cdot} , h_{\theta} ( v ) , I} \\ \end{array} \right)}} \\ \end{array}
$$
$$
h o ( v ) : h y p o t h e n s f u n c t i o n / H o d e l
$$
etrmrAinsnmtyAmrari seyesAtsn $\in$ 
×&P
 $\mathbb{D a t a}$ 宝 $\cup$ n2 $P_{x y}$ 
$$
\mathrm{P_{y}} |_{\times}
$$
$$
P_{\theta} \left( y \vert x \right)=\Re\left( y ; h_{\theta} \left( x \right) , \bot\right) \underline{{{\bf i} H o d e l}}
$$
$$
h_{\theta} ( x ) : : X \rightarrow\mathbb{R}
$$


---

## Page 125


1
$$
\theta^{*}=\arg_{\theta} \frac{-1} {N} \sum_{i=1}^{N} \log P_{\theta} \left( y_{i} / x_{i} \right)
$$
$$
\alpha\arg_{\theta}-\sum_{i=1}^{N} \log_{\theta} [ ( y_{i}-h o ( x ) )^{2} ]
$$
$$
\theta^{\circ} \alpha g_{\theta}^{m i n} \sum_{i=1}^{N} [ y_{i}-h_{0} ( x_{i} ) ]
$$
A S


---

## Page 126


$$
\frac{R_{i s} k \dots M_{i n i m i \ref{d i b o n}} a L_{o n} \dots F_{r a m e} \omega_{o n} K .} {\bf\partial}
$$
an
$$
D a t a D=\left\{\left( x_{i} , y_{i} \right) \right\}_{i=1}^{N} \approx i i d R_{x_{y}}
$$
$$
\times\in\mathbb{R}^{\alpha} y \in\mathbb{R}^{k} \left/ \left\{1 , 2 , 3 , \ldots k \right\} \right.
$$
 $\mathrm{G o a L}$ i3 $+_{\circ}$ 
h:XYendr al
ia $x \rightarrow Y$ Roniaghidan $^\circ$ n $h \left( \mathrm{x} \right)$ PDdlne ng36d of ls-Honthon b L: t
R
$$
L \left( h \left( x \right) , y \right) \rightarrow R^{+}
$$
$$
\begin{array} {c} {{\displaystyle\mathcal{E} g : s \Big) \; \; L \; \left( h \left( x \right) , y \right) \; \;=\; \; \left\vert\Big\vert\; h \left( x \right)-y \Big\vert\Big\vert_{z}^{2} \; \; \mathrm{: ~ a . s . ~ o m o . ~ l a r s} \right.}} \\ {{\displaystyle b \Bigg) \, L \left( h \left( x \right) , y \right) \; \;=\; \left\{\begin{array} {c c} {{0}} & {{i f \; h \left( x \right)=y \; \; \mathrm{: ~} 3 q . \; \mathrm{o n c . ~ l a ~ s o .}}} \\ {{i}} & {{i f \; h \left( x \right) \neq y}} \\ \end{array} \right.}} \\ \end{array}
$$


---

## Page 127

i
"y
$$
R ( h ) \stackrel{\Delta} {=} \bigoplus_{\alpha_{y}} L ( h ( x ) , y ) : R_{i} \circ k f^{u n c h o n}
$$
polien:. Eskwute $\mathrm{h}^{*} \left( \mathrm{x} \right)$ 
$$
h^{\prime} ( x )=\arccos_{\max} \vert\xlongequal{\leftarrow} R_{x} \vert
$$
tnmnayA
 $\mathbb{P}_{x_{y}}$ N
3 vnhnoin
ein
$$
\sin n o g a t e
$$
$$
\widehat{R} \left( h \right) \cong\frac{1} {N} \sum_{i=1}^{N} L \left( h \left( x_{i} \right) , y_{i} \right) : \sum_{R_{i}=k}
$$
山
$$
\widehat{h}^{\star} \left( x \right)=\begin{array} {c} {{\mathrm{a r g_{m i n}}}} \\ {{\mathcal{O}_{h} \in{\cal H}}} \\ \end{array} \widehat{R} \left( h \right) \Bigg| \begin{array} {c} {{x_{i_{i}} y_{i} \cdots n i i d}} \\ {{}} \\ \end{array} P_{x_{i_{i}}}
$$
$$
\Sigma_{\mathrm{m i p i n i c a l} R i j k M i n i m i g a l i o n} ( \xi R \mu) .
$$


---

## Page 128


 $L L N o .$ 
$$
{\cal H}_{\mathrm{a j e v e r} \, , \quad} {\widehat h} \left( x \right) \quad m a y \; \; n o t \quad\; \; \mathrm{c o n v e r g e ~ t o ~ h^{\prime} ~ ( x ) ~} .
$$
$$
C_{e n t \, r a l} \quad q u e \, s \, h o n \quad: \ \, w h e n \quad w o u l d \ \ \stackrel{\wedge} {h}^{\star} ( x ) \to\stackrel{\wedge} {h} ( x ) ? .
$$
$$
\det e x a g a \tan.
$$
O
$$
\Re_{Y}
$$
1
A Eoamwkeo4 EPN
$$
\arctan D=\left\{( x_{i} , y_{i} ) \right\}_{i=1}^{\prime}
$$
1 店 <uppic $h_{\theta} ( x )=e^{\theta^{\dagger} x}$ 
 $6 y^{\theta}$ 
$$
D e f i n e L_{0}=f_{1} , \mathrm{~ m e a n ~ r q ~ e r v o r} .
$$
$$
\overrightarrow{R} ( h )=\frac{1} {N} \sum_{0=1}^{N} \left[ h_{0} ( x )-y_{1} \right]^{2}
$$
$$
\theta^{*}=\arcsin\widehat{R} ( h_{\theta} )
$$


---

## Page 129

We abue 1s eouclast to
esoniny St coy ML slmdw mn Bfn) ^N{.w.2
fgauwn w $\mathcal{E}$ 店公
力
ethrn : hat menmndgrs ho
WGOK
$$
\cos\sin\alpha=D=\{( x_{i} , y_{i} ) \}_{i=1}^{N} \approx i 1 d R_{x_{y}}
$$
$$
\left\{\begin{matrix} ( h \left( x \right) , y )=\left\{\begin{matrix} 0 \\ 0 \\ 1 \end{matrix} \right. \begin{matrix} h h ( x )=y \\ \end{matrix} \right.
$$
$$
R \left( h \right)=\lim_{R x y} L \left( h \left( x \right)_{1} g \right)
$$


---

## Page 130


1
$$
\frac{\rho v e s \hbar o n} {} : \frac{\mathrm{w h a t h_{j} m i n i m i g e s ~ f \hbar_{c} b o v}} {} : \cdots, \rho_{n}=\rho_{n} \in\frac{\rho_{n}} {1 0 0 0 0}
$$
 $A_{\sim5}$ 
$$
\begin{matrix} h_{B} ( x )=\left\{\begin{matrix} 0^{1}+\frac{1} {6} P ( y-0 ) m \\ 1+P ( y-1 ) \leq P ( y-1 ) \\ \end{matrix} \right. \\ ( 1 ) B a y e^{1}=C l a s e x p^{2}=\end{matrix}
$$
 $\underbrace{C l a i m}$ :REE $\lfloor\mathrm{h} \rfloor$ 
 $\neq h \in1 4$ 
$$
P \cot b : L e t=S_{i} ( h )=\left\{\times\in R^{d} : h ( x )=i \right\}
$$
 $S_{\circ} ( h ) \cap S_{\cdot} ( h )=\phi$ -n . $S_{0} ( h ) \cup S_{1} ( h )=\log^{\alpha}$ 
she T Luany s
on inicatov Bexnaolic

---

## Page 131


$$
\left. \begin{matrix} R ( h )=\underset{B_{x_{y}}} {\phantom{+}} \left[ L ( h ( x ) , y ) \right] \right]
$$
工
$$
= P ( \pm_{h ( x ) \pm y} )
$$
$$
= \Re( h ( x )=1 , y=0 )+\Re( h ( x )=g n y=1 )
$$
PEONN
$$
\Re( y=1 ) R \times( x \in S_{0} ( h ) \vert y=1 )
$$
$$
= \Re\left( y^{2} \right)_{0} \int P_{x 1 \cdot y=0} \left( x \left| y=0 \right) d x+\Re\left( y=\right) \right| P_{x 1 \cdot y=0} d x
$$
$$
\tt_{i n c e \quad S_{i} ( h ) \cap S_{o} ( h )=\phi} ,
$$
-+x{, w oaly Oonsde one
ol the too integedr


---

## Page 132

p- to ownaheto
$$
R \left( h \right) \;=\; \int_{S_{i}} \left\vert\right\vert\! \! P_{y=0} \; P_{x \vert y=0} \; d \, x \; \;+\int_{S_{o}} \left\vert\right\vert\! \! P_{y=\vert} \; P_{x \vert y=\vert} \; d \, x
$$
$$
h_{3}=\left\{\begin{matrix} 1 1 . . . P_{y=1 1 x} > P_{y=0 1 x+1}=P_{y=1} P_{y=1} > P_{x 1 y=0} P_{y=0} \\ 0 . . . \begin{matrix} 1 1 . . . . P_{y=1} P_{y=0} \end{matrix} \right.
$$
$$
\Longrightarrow R ( h_{B} )=\int\min( P_{y=0} \cdot P_{x 1 y=0} , P_{y=1} \cdot P_{x 1 y} ) d x
$$
Y $\mathrm{R} \left( \mathrm{h} \right)$ 4.AR
$$
E_{y 1 x}=\arccos_{h} \varprojlim\left[ ( h \left( x \right)-y )^{2} \right]
$$


---

## Page 133



---

## Page 134


$$
\frac{M_{a \rightarrow\min e} L e a \times m i n g} {j}
$$
$$
p o b a b_{i} b i s f i c v i t \omega p o i n t .
$$
 $\mathcal{9}$ : $\mathrm{R}$  $\vartheta$ 
$$
X : d o m a n s e t ( I n p u t )
$$
。 $\left( o u t p u t \right)$  $( x_{\nu} , y_{\nu} ) ($ 
E
$$
\mathrm{G i v e n} \quad p a i v s \qquad D=\left\{\begin{matrix} {( x_{i_{1}} y_{i} ) ,} \\ \end{matrix} \right. \left( x_{i_{1}} y_{z} \right) ,
$$
中 :×-→Y
 $\frac{\rho} {\Gamma}$ 
Twa $y^{i n g}$ ;w`+ $s$ wwsn $G_{i v e n}$ `.+|^-
 $3 \tan t$ swiNase inda put on f rpe $\\#_{\mathrm{h e}}$ ;^wy{* $\log\sin\sin\alpha_{\log V}$  $\omicron$  $i m a g e s$ 
:
 $\mathtt{q}$ 
 $\mathtt{P}^{\mathrm{H}}$ ol


---

## Page 135


 $\times; \in\times c \mathbb{R}^{\sigma}$ 
 $y_{i} \in\{0 , 1 \}$ ·
$$
\begin{array} {c} {{0 \rightarrow d i \tan\tt< d}} \\ {{i \rightarrow\ldots b e m i g n / \mathrm{n o n-d i \tan\tt< d .}}} \\ \end{array}
$$
$$
f : X \to Y
$$
 $R e s o v t$ to stahikcal methods:
rasepedid obeuobont &Actkmate f mnepailite yoe uwm
JA : somple teuLe. (it ep al eanthe
otcones ef o. rndan
Ezxpnwent ]
 $C_{O n} \sin\sin\alpha e v$ tsubetsafsamaie $3 p a c e$ 
Lek $\mathbb{F}$ dente the Coleckoa $\mathbf{o} f \ldots\mathbf{a l l} \cdot p_{0} \gamma\gamma/ l e \textrm{i u b t e t r} .$ ohtie Aniyn a eoane e $\mathbb{F}$ 


---

## Page 136


$$
P_{\gamma\circ} b a b i l i t_{4}
$$
McAu3 $\mathbf{i < 5}$ ose Mah manr.
 $\mathbb{P}$ F[0,]
cnae.
 $\mathbb{P} \left( \mathrm{\tt~ A} \right) ~ \geq~ \mathrm{o}$ 
m
$$
\mathcal{P} ( \Omega)=i \beta\left( \begin{matrix} {\phi} \\ \end{matrix} \right)=0
$$
$$
\mathrm{~ \Lambda. i ~ \Big) ~ , ~ \mathrm{\Large~ A ~ , ~ B ~} ~} \quad\mathrm{\Large~ s t ~} \quad\mathrm{\Large~ A ~ \cap~ B ~=\phi~ , ~ \} \quad\mathrm{\Large~ P ~ \Big( A ~ v ~ B \Big)=\Re~ \Big( A \Big)+\Re~ \big( B \Big)} .
$$
(A,RFP)T $\mathsf{P}_{\infty} \mathsf{b}$ 下kL
T
$$
x : \Omega\rightarrow R
$$
Ptra A paitnc KXtrodn ocuat
 $\varUpsilon^{\infty m}$ 8
 $\varOmega\rightarrow\mathbb{R}$ 
F →Bayma ahyolna . Cw. a7
 $\P$ -ifiercritham pmtn
Ra)ApAXni
$$
( \Omega, \mathbb{F} , \mathbb{P} ) \xrightarrow{R v . X} ( \mathbb{R} , \mathbb{B} , \mathbb{P}_{x} ) : \frac{\omega\sigma\kappa\cdot\omega\cdot H} {t h i s .}
$$


---

## Page 137


$$
R a n d o m \ldots v a n i b l e s \ldots w i t h \nu e c b o r \cdot v a l u e d \ldots v a n g c
$$
 $\mathrm{s p a c e s}$ .un.m : rn，
 $I p a c e s$ -
$$
x : \Omega\rightarrow R^{d} ( \mathrm{v e c t o r-v a l u e d R v .} )
$$
$$
\begin{array} {r c l} {{P_{\underline{{{x}}}} \left( x \in\mathbb{R}^{d} \right)}} & {{=}} & {{{\mathrm{p o b a b i l i t y ~ q u ~ t h e ~ \quad i n v . ~ \quad~ i m o g e}}}} \\ {{}} & {{}} & {{\mathrm{o p ~ c a r ~ b i s a n d ~ q u n d u c l ~ \quad u n d e r ~ X ~ .}}} \\ \end{array}
$$
Vetoy vawsaRVe
$$
\begin{array} {c} {{H_{\infty} e+h e \quad r a n g c \quad s e t \quad o f \quad H_{\infty} \cup\mu_{\infty} \in H_{\infty} ( R_{V} ) ,}} \\ {{i s \quad R^{d} \quad\mathrm{w h e v e} \quad d \quad i s \quad a \quad s c a l a x}} \\ \end{array}
$$
×： $\bigcap$ → $\mathbb{R}^{\alpha}$ 
$$
J_{0 i n t} d_{i} s t n i b u t i o n s :
$$
$$
\angle e t \ldots\varOmega b e a \Bigr. \noindent\mathrm{~ I a m p l e ~ I p a c e}
$$
D}
$$
n e-\tan\theta\mu\in\tan\theta, x_{1} E x_{2}
$$
 $\chi_{1}$ 1金上发馆
 $x_{2}$ n-R.


---

## Page 138

thnunand ohany Autadr. $a_{\varDelta}$ 元 iaea.o)- PiE: woeuton $\propto$  $\mathcal{S}$ 区, exesesoh] $\circ+$ 
EA
0wod] （esy wde - $\mathbb{T}_{\mathrm{h e}}$ akaue iac. $\mathrm{C a n}$ bse csxiandta $\mathrm{t o}$ 
 $\mathrm{d}$ Kcala Randm Noxabsle 
$$
{\frac{D e f i n e \ldots c o n d i \, b i o n a l \ldots p r o b a b i l i e J} {\mid}} \, d_{i} s t_{n} \, b u b i o n I .
$$
 $\mathrm{A}$ 
1
$$
\begin{matrix} {\mathbb{P} \left( {\tt A} \mid{\tt B} \right)} & {=} & {\mathbb{P} \left( {\tt A} \cap{\tt B} \right)} \\ {} & {} & {\mathbb{P} \left( {\tt B} \right)} \\ \end{matrix}
$$
 $\circ\sim$ -.
$$
\begin{array} {c c c c c c c c c c c c c} {{\mathrm{\tiny~ d u p p o s e}}} & {{X}} & {{E_{i}}} & {{Y}} & {{\mathrm{\tiny~ a r e}}} & {{\mathrm{\tiny~ t w o}}} & {{R v s}} & {{\mathrm{\tiny~ d e f \! \! \! ~ t i n e}}} \\ \end{array}
$$
$$
\begin{matrix} \textcircled{P}_{x} \textcircled{y}=\textcircled{y}=\frac{\sqrt{P}_{x} \textcircled{y}} {\sqrt{y}}} \\ \end{matrix}
$$


---

## Page 139

hagaedatatantyx  $\gamma$ oe bwo $\mathbb{R} \mathtt{J}_{\varDelta}$ mes gnd $\circ f$ ×is djnea a3 R= Re 
vsqd)
$$
P_{y}=1 P_{x y} d x
$$
$$
\xi x a m p l e
$$
Emae anyce eonewn $P \times q$ 
 $\mathtt{q}$ CasNhe yuwsd a4 a $\left\{\begin{matrix} P q \\ -d i m \end{matrix} \right.$ 
 $\mathsf{P}$  $\lor e c+1 0 r$ 
$$
\mathrm{I n : g e n e c a l , \quad a n y \quad d a t a p o i n t \quad i s \quad a \quad d-d i m v e c b w}
$$
esry damadius on slmnat n ho rengeraes onen centex
$$
\begin{array} {c} {{T_{\cal C} \mathrm{d i s t n b u h o n} \downarrow\mathrm{u n c h o n} \qquad\mathrm{i n d i c a t e} \downarrow\mathrm{q u a h i f ~ e r}}} \\ {{+h_{\cal C} \qquad\mathrm{U e l i n b o u d} \qquad\mathrm{o f ~ o b s e r v i n g ~ a ~ d a t u p o i n t} \qquad\mathrm{u n d e r} \times.}} \\ \end{array}
$$


---

## Page 140

 $\operatorname{t h c}$ A线A Dsnaonepachao
 $T y p i c a N y$ AAAR m $\mathrm{t h e r}$ adehonal random venable $\alpha_{e} b$  $\omicron\cap$ + $\epsilon$  $\tt1 a m e$ ranple apoce
 $\times$ 业 $\mathrm{h a v e}$ 
 $\min\tan\theta$  $y \in\mathbb{R}$ 
 $\sharp$  $\mathcal{\gamma}$ me-tna wa $\{1 , 2 , 3 , 4 , 5 \}$ X-:dataspatc
Tie autaetead lher
atarts woth $\alpha$ dataset
$$
D=\left\{\begin{matrix} ( x_{1} , y_{2} ) , ( x_{2} , y_{2} ) , ( x_{3} , y_{3} ) \cdots( x_{4} , y_{3} ) \right\}
$$
 $I_{\mathrm{n d e p e n t}}$ ACoaiaon,
、
datvwbubed
Tadendenb- aiisos ata pantr & no dutu - o me gions


---

## Page 141


$$
\begin{matrix} A U} & {p o b l e m s i n} & {M a c h i n e} & {1 e o m i n g} \\ {c o n} & {b e} & {p o o n} & {0 .} \\ \end{matrix}
$$
 $G_{i} \textrm{J e n}$ 'D o unKnauan Pe Eshoabe $\mathbb{P_{x}}$ 
 $\mathring{\omega}$ oYCond $\mathbb{P_{x}}$ nals
$$
\sin\log l e f^{\infty m i t}
$$
^
 $R e c a l l$ 
$$
D=\left\{\begin{matrix} ( x_{i} , y_{i} ) g_{i=1}^{N} \times\lim_{i=1} \left( \begin{matrix} {u n k n o n} \\ \end{matrix} \right) \right.
$$
 $G_{n v e n}$  $\alpha$  $d a f a s e t$ 
$$
\begin{array} {c c c} {{\mathrm{T y p i c a l l y} ,}} & {{X : \mathrm{i n p u t ~ I p e a t u v e s} / d a t a}} \\ {{}} & {{Y : : \mathrm{i a b e L ~ 1 o u t p a t .}}} \\ \end{array}
$$
$$
\begin{array} {l l} {{X \, {\cal E} \, \gamma\quad\mathrm{a r e} \quad\ \mathrm{s u n d o m} \quad\mathrm{v a n a b l e s} \quad\mathrm{d e f} \quad\mathrm{a n d} \quad\mathrm{a n} \, \mathrm{a n d l e s} \quad\mathrm{d e f} \quad\mathrm{a n d} \quad\mathrm{a n} \, \ \mathrm{a n d l e} \, \quad\mathrm{d e f} \quad\mathrm{a n d} \quad\mathrm{a n d l e} \, \quad\mathrm{a n d} \quad\mathrm{a n d} \quad\mathrm{a n d} \quad\quad\mathrm{a n d} \quad\quad\mathrm{a n d} \quad\quad\mathrm{a n d} \quad\quad\mathrm{a n d} \quad\quad\mathrm{a n d} \quad\quad\mathrm{a n d} \quad\quad\mathrm{a n d} \quad\quad\mathrm{a n} \, \quad\mathrm{a n d} \quad\quad\mathrm{a n} \, \quad\mathrm{a n} \, \quad\mathrm{a n} \, \quad\mathrm{a n} \, \quad\mathrm{a n} \, \quad\mathrm{a n} \, \quad\mathrm{a n} \, \quad\mathrm{a n} \, \quad\mathrm{a n} \, \to\mathrm{a n} \, \to\mathrm{a n} \, \to\mathrm{a n} \, \to\mathrm{a n} \, \to\mathrm{a n} \, \to\mathrm{a n} \, \to\mathrm{a n} \, \to\mathrm{a n} \, \, \mathrm{a n} \, \to\mathrm{a n} \, .}}} \\ \end{array}
$$
Einasectald pudsena of $\dashv L$ ：
 $G n v e n$ D n unhrow $\cap$ dnstebakon,
） $\varepsilon\ddag i m a t e$ .theoisixbahion
） $\rho_{a m p} l e \gamma^{n o m}$ the atobation


---

## Page 142

DtisbiuoSkmcthen
$$
\operatorname{G i v e n} D=\{( x_{i} , y_{i} ) \} \sim i i d P_{x y}
$$
Examples-
Estkmote $\mathbb{P}_{\mathsf{y l x}}$ asay unten, 72Lnyg
 $\mathsf{P_{y | x}}$ R3xces0n,-4K ${\cal I}_{\mathrm{n}}$ )tntk $\mathtt{P_{x}}$ 2
T $\Re\log_{y} ( x \mid y=y )$ 
su aiuln.
D $\mathrm{P_{*}}$ ：x→'， $s \cdot\mathrm{t}$ 1
 $d e n s i f_{y}$ GmweaCRW wsta $d_{i} s t$ Juakin $\mathcal{P}_{x}$ 
puataon
$$
\stackrel{\prime} {P_{x}} \left( x \right)=\int\stackrel{x} {p_{x} \left( x \right)} d x
$$
$$
\frac{\mathrm{T h e} \cdot\mathrm{c h a l l e n g e} \cdot\omega\cdot\mathrm{H} \cdot\mathrm{H L}} {}
$$
bmnD $\triangleleft$ & $\ddag^{\infty m}$ ustmwn $\mathrm{P_{x}}$ .thmt $\mathrm{P_{x}}$ 
$$
\mathrm{c h a l l e n g e :} P_{x} \mathrm{i s} \mathrm{c o m p l e t e l y}
$$
ouon


---

## Page 143

utn the u thndedany nhan
g"^ it: 4mbes?
 $\cot\sin\sin\alpha e x$ w $\mathrm{h a v e}$ adktastbD、
$$
D=\{x_{i} \}_{i=1}^{N} \sim i i d P_{x}
$$
 $\varepsilon_{g}$ A中 $w_{1} \in w_{2} \in R^{\alpha}$ 
 $\mathtt{P_{e}}$ 8 $\mathrm{P_{\kappa}}$ 
Model-Chaile
 $P_{\theta}^{a} ( x )=W_{1}^{T} x+W_{2}$ 
 $\Theta$ 
$$
P_{0}^{b} \left( x \right)=N \left( x ; H , \Sigma\right)
$$
$$
\theta=\{\mu, \Sigma\} , \mu\in R^{\alpha_{1}} \alpha\times\alpha
$$
: puanetes
 $\grave{\varsigma} \check{\varsigma}$ >:;}n^ $\propto$ 山 山 $c$ 
 $\alpha$ 
Kiwsn
$$
P_{x} \approx P_{\theta}
$$
 $\mathrm{L e t}$ diaatie ihe idtstenee meko $\mathrm{c}$ 
 $\flat( \omega$  $\mathrm{p_{x}}$  $\mathcal{E}$ ?s
$$
\alpha\left( P_{x} , P_{\theta} \right) : P_{x} \times P_{\theta} \rightarrow R^{+}
$$


---

## Page 144


$$
\begin{array} {l} {{\mathrm{i i i f} \displaystyle\lim_{i n d} \displaystyle| \mathcal{E}_{s h \, m a^{\dagger} e} \displaystyle\lim_{i n d} \displaystyle\frac{\mathrm{i f}} {e} \displaystyle\lim_{i n d i n g} \displaystyle\lim_{i n d i n g} \displaystyle\lim_{i n d i n g} \displaystyle\lim_{i n d i n g} \displaystyle\lim_{i n d i n g} \displaystyle\lim_{i n d i n} \displaystyle\lim_{i n d i n} .} \\ \end{array}
$$
 $\Theta$ “a" $d \left( p_{x} , p_{\theta} \right)$ 


---

