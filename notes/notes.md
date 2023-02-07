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
that $M(\symbfit{y}\_i, \symbfit{y}\_j) = m_{ij} \in \mathbb{R}$
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

**Local Error Consistency (EC_l)**:

$$\begin{align*}
\text{EC}_{\ell} &= \frac{|\symbfit{e}_i \cap \symbfit{e}_j|}{|\symbfit{e}_i \cup \symbfit{e}_j|} \\
&= \frac{sum(\symbfit{e}_i \land \symbfit{e}_j)}{sum(\symbfit{e}_i \lor\symbfit{e}_j)} \\
\end{align*}$$

**Global Error Consistency (EC_g)**:

$$\begin{align*}
\text{EC}_{g} &= \frac{|\symbfit{e}_i \cap \symbfit{e}_j|}{n} \\
&= \frac{\text{sum}(\symbfit{e}_i \land \symbfit{e}_j)}{n} \\
&= \text{mean}(\symbfit{e}_i \land \symbfit{e}_j) \\
\end{align*}$$

**Accuracy-Based Error Consistency (EC_acc)**:

$$\begin{align*}
\text{EC}_{\text{acc}} &= \text{acc}(\symbfit{e}_i, \symbfit{e}_j) \\
 &= \text{mean}(\symbfit{e}_i = \symbfit{e}_j) \\
\end{align*}$$

**Correlation-Based Error Consistency (EC_corr)**:

$$\begin{align*}
\text{EC}_{\text{corr}} &= \text{corr}(\symbfit{e}_i, \symbfit{e}_j) \\
\end{align*}$$

where $\text{corr}(x, y)$ is the Pearson correlation coefficient between $x$
and $y$, making this equivalent to the $\phi$- or Matthews correlation
coefficient between binary errors.

### Prediction-Based

Other pairwise reproducibility metrics can be defined by choosing $M$ to be a
distance, similarity, or association metric operating on the predictions directly.

**Percent Agreement (PA_acc)**

$$\begin{align*}
\text{PA}_{\kappa} &= \text{acc}(\symbfit{y}_i, \symbfit{y}_j) \\
&= \text{mean}(\symbfit{y}_i = \symbfit{y}_j) \\
\end{align*}$$

**Kappa Prediction Agreement (PA_K)**

$$
\text{PA}_{\kappa} = \kappa(\symbfit{y}_i, \symbfit{y}_j)
$$

Where $\kappa$ is the Cohen's Kappa agreement between prediction vectors. While
this naturally handles multi-class problems, interpretation is highly dubious,
as we should in general expect strong dependency among predictions, which
violates a core assumption of Cohen's $\kappa$.

**Cramer's-V Prediction Agreement (PA_V)**

$$
\text{PA}_{V} = V(\symbfit{y}_i, \symbfit{y}_j)
$$

Where $V$ is [Cramer's V](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V),
which is a correlation in $[0, 1]$ based on the $\chi^2$.



**Note**: I also tested **Krippendorf's** $\alpha$, and Gwet's **AC1/2**, but
they show largely identical behaviour to $\kappa$, and so are not discussed
further.


## Metric Evaluation: Simulated

No classifiers need actually be fit to examine the general behaviour of these
metrics. That is, given a classifier $f$, repeated trainings and evaluations on
some test data $\symbfit{x}$ will yield predictions $\symbfit{\hat{y}}_i$  with
correct labels $\symbfit{y}$ shared across evaluations. We can *ignore* $f$ and
$\symbfit{x}$ here and model
$\symbfit{y}$ as being sampled from a discrete random variable $\mathbf{Y}$ with possible values $\{0, 1, ..., c-1\}$ for $c$ classes. We can model the predictions $\symbfit{\hat{y}}_i$ as samples
from a discrete random variable $\hat{\mathbf{Y}}$. The we can investigate the
behaviour of various reproducibility metrics by specifying different distributions
and relationships for $\mathbf{Y}$ and $\hat{\mathbf{Y}}$.


For example, across repeats, there will always be a
**maximum error set size** $s \in [0, 1]$ which is the largest proportion of
test samples for which there is an erroneous prediction (i.e. at least $1 - s$
samples are always classified correctly).  The maximum error set could be:

- **Fixed**: Errors, if they occur, occur always on the same subset of the test
  set, i.,e. always on the same indices of $\symbfit{y}$
- **Variable**: Errors never exceed $s$ proportion of test samples, but are
  not restricted to a particular subset of test samples


In addition, errors can be:

- **Independent**: The predictions $\hat{\symbfit{y}}_i$ and $\hat{\symbfit{y}}_j$ are
  independent for all repeats $i \ne j$
- **Dependent**: All predictions depend on (are partly determined by) some
  source prediction $\symbfit{y}_{\text{base}}$, which may or not be similar
  to the true labels $\symbfit{y}$

Finally, both the true labels and predicted labels can have **different distributions**.
That is, modeling the

### Class Distributions

Without loss of generality, for $c$ classes, where class $i$ occurs with
probability $p_i$, we can sort the **class probabilities** into a class
probability vector:

$$
p = [p_1 \ge p_2 \ge \dots \ge p_c], \quad p_i \in [0, 1]
$$


I.e. because the ordering of classes does not matter, we need only consider


Then *all possible configurations of $c$ classes* are defined by the rate and
regularity of decline of the sorted class probabilities—roughly, how flat vs.
skewed, and how smooth sv. step-like the sorted distribution is. and how smooth
or step-like i. That is, the most extremely-skewed class distribution is:

$$
p = [1, 0,  \dots, 0]
$$

and the flattest distribution is

$$
p_i = \frac{1}{c}
$$

The distribution may decline "smoothly", where e.g.
$\Delta p_i = p_{i}-p_{i+1} > 0$ for $i < c$, or in a stepwise
fashion, where $\Delta p_i = 0$ for some classes.  It is trivial to
implement an algorithm that will *eventually* simulate all possible class
distributions, by simply defining (in pseudo-code):

```python
p_raw = rand_uniform(min=0, max=1, n_samples=n_classes)
p = sort(p_raw) / sum(p_raw)
```

that is, we sample $p_i^{\prime}$ from $c$ i.i.d. $\text{U}(0, 1)$
distributions, and then set $p_i = p_i^{\prime} / \sum p_i^{\prime}$. However,
this will only rarely simulate exponential (skewed) distributions, and so we
can force other distribution types (step-like, multi-modal, exponential) by
simply altering the generation of the $p_i$ values (see Appendix A).

# Appendix A: Generating Discrete Distributions

We can generate variably-exponential distributions via the algorithm:

```python
scale = rand_uniform(1 / 10, 20)
p_raw = linspace(0, 1, n_classes) ** scale   # exponentiate element-wise
p = sort(p_raw) / sum(p_raw)

```

For small values of `scale`, the distribution will be almost uniform, and for
large values, will be highly skewed. With the randomness and sort, this too
will *eventually* cover all possible distributions of classes, the bias is just
for skewed distributions.

For $c > 5$ or so, with enough samples, we will regularly achieve
$\Delta p_i \approx 0$. However, we may also wish to force this situation in
simulating highly multi-modal datasets. This can also be done easily enough:

```python
n_modes = rand_integer(1, n_classes)
extreme = n_classes / n_modes    # will be > 1
p = ones(n_classes)              # e.g. [1, 1, 1, ....]
p[:n_modes] = extreme            # set first n_modes elements to extreme
p = sort(p_raw) / sum(p_raw)     # final sort
```

Alternately, we may wish to force an explicitly random, step-like distribution:

```python
def get_step_ps(n_classes: int) -> ndarray:
    max_width = ceil(n_classes / 5)
    n_steps = rand_integers(2, max(3, max_width + 1))  # n_steps >= 2
    step_heights = rand_uniform(0, 1, n_classes)
    steps, step_widths = [], []
    for i in range(n_steps):
        wmax = min(max_width, n_classes - np.sum(step_widths))
        if wmax > 2:
            width = rand_integer(2, wmax)
        else:
            width = 0
        steps.extend([step_heights[i] for _ in range(width)])
        step_widths.append(width)

    n_remain = n_classes - len(steps)
    p_remain = rand_uniform(0, 1, n_remain)
    p_raw = concatenate(p_remain, steps)
    p = sort(p_raw) / sum(p_raw)
    return ps
```



In the vast majority of competent deployment scenarios (with a competent data
scientist), one would most likely fit a classifier to data with e.g. N classes,
but where $n \ll N$ classes make up the vast majority of the samples. Instead,
one would (hopefully) first try to separate the majority class from the minority
classes, and then apply a separate classification
