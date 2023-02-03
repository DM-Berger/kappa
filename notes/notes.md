# Reproducibility Metrics

For a metric to be a useful prediction reproducibility or reliability metric, it
should have the following qualities:

- it should have direct, practical intepretability
- it should not require false assumptions for interpretation or validity
  - e.g. normality, independence of predictions
- it should be simply defined
- it should contain novel information not present in simpler statistics
  - the novel information should not be confounded with something uninteresting
  - it should be *moderately* related to known quantities
    - e.g. a reproducibility metric should be somewhat related to accuracy, accuracy variance
      without being either completely uncorrelated or perfectly correlated


Ideally, a good metric also:

- is relatively insensitive to class distributions and number of classes
  in the classification problem
- is sensitive to a discrepancy between prediction distributions and true distributions
- distinguishes between random vs. non-random (dependent) predictions
- [BONUS]: should be naturally extendable to the regression context



## Pairwise Reproducibility Metrics

Reproducibility metrics need to operate on repeated model runs that share a
validation set of some size $n$. Each repeat evaluation $i$ yields a single
prediction vector of class labels $\symbfit{y}\_i \in [0, 1, \dots, c - 1]^n$
for $c$ classes, and a set of runs yields a set of predictions
$\symbfit{y}\_1, \dots,  \symbfit{y}\_k$. This yields $k(k-1)/2$ distinct
pairings of runs.  Given a function $M$ such
that $M(\symbfit{y}_{i}, \symbfit{y}_{j}) = m_{ij} \in \mathbb{R}$
for $i > j$, then we can define a *(pairwise) metric reproducibility
distribution* $\mathcal{M} = \{ m\_{ij} : i > j \}$
over the repeated model runs. A ***pairwise reproducibility metric*** is any
real-valued summary (statistic) of $\mathcal{M}$.

In most cases, the most natural summary is the mean of all $m_{ij}$.

## Candidate Metrics

### Error-Based

Given two different predictions $\hat{y}_1$ and $\hat{y}_2$ for true label $y$,
we can always define binary residuals

$$
e_i = \begin{cases} 0 & y = \hat{y}_i \\ 1 & y \ne \hat{y}_i  \\ \end{cases}
$$

and then if we have $n$ samples, define agreement based on the length $n$ binary errors or residual vectors:

$$
\symbfit{e}_i = \left(e_i^{(1)}, \dots, e_i^{(n)}\right),
$$

and where $\symbfit{y}_i = \left(y_i^{(1)}, \dots, y_i^{(n)}\right)$. These metrics include:

#### Set-Based

Given binary error vectors $\symbfit{e}_i$, $\symbfit{e}_j$ define:

$$
\symbfit{e}_i \cap \symbfit{e}_j = \symbfit{e}_i \land \symbfit{e}_j \\
\symbfit{e}_i \cup \symbfit{e}_j = \symbfit{e}_i \lor \symbfit{e}_j
$$

where $\land$ and $\lor$ treat the binary vectors as booleans and compute
elementwise logical operations *and* and *or*, respectively. Then we can define
the following metrics:

**Local Error consistency (EC_l)**:

$$\begin{align*}
\text{EC}_{\ell} &= \frac{|\symbfit{e}_i \cap \symbfit{e}_j|}{|\symbfit{e}_i \cup \symbfit{e}_j|} \\
&= \frac{sum(\symbfit{e}_i \land \symbfit{e}_j)}{sum(\symbfit{e}_i \lor\symbfit{e}_j)} \\
\end{align*}$$

**Global Error consistency (EC_g)**:

$$\begin{align*}
\text{EC}_{g} &= \frac{|\symbfit{e}_i \cap \symbfit{e}_j|}{n} \\
&= \frac{\text{sum}(\symbfit{e}_i \land \symbfit{e}_j)}{n} \\
&= \text{mean}(\symbfit{e}_i \land \symbfit{e}_j) \\
\end{align*}$$

### Prediction-Based

Other pairwise reproducibility metrics can be defined by choosing $M$ to be a
distance or similarity metric, or a measure of association (e.g. correlation,
agreement).

**Kappa-Based Error Agreement ($\text{EA}_{\kappa}$)**



