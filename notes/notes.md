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
distance or similarity metric, or a measure of association (e.g. correlation,
agreement).

**Kappa-Based Prediction Agreement (PA_K)**

$$
\text{PA}_{\kappa} = \kappa(\symbfit{y}_i, \symbfit{y}_j)
$$

Where $\kappa$ is the Cohen's Kappa agreement between prediction vectors. While
this naturally handles multi-class problems, interpretation is highly dubious,
as we should in general expect strong dependency among predictions, which
violates a core assumption of Cohen's $\kappa$.

**Accuracy-Based Prediction Agreement (PA_acc)**

$$\begin{align*}
\text{PA}_{\kappa} &= \text{acc}(\symbfit{y}_i, \symbfit{y}_j) \\
&= \text{mean}(\symbfit{y}_i = \symbfit{y}_j) \\
\end{align*}$$

**Note**: I also tested **Cramer's V**, **Krippendorf's** $\alpha$, and Gwet's
**AC1/2**, but they show largely identical behaviour to $\kappa$, and so are
not discussed further.


## Metric Evaluation

No classifiers need actually be fit to examine the general behaviour of these
metrics.

Given a classifier $f$, repeated trainings and evaluations will yield a number
of predictions on some test set with correct labels $\symbfit{y}$ shared across
evaluations. Across repeats, there will be a **maximum error set size** $s \in
[0, 1]$ which is the largest proportion of test samples for which there is an
erroneous prediction (i.e. at least $1 - s$ samples are always classified
correctly).

The maximum error set could be:

- **Fixed**: Errors, if they occur, occur always on the same subset of the test
  set, i.,e. always on the same indices of $\symbfit{y}$
- **Variable**: Errors never exceed $s$ proportion of test samples, but are
  not restricted to a particular subset of test samples


In addition, errors can be:

- **Independent**: The predictions $\symbfit{y}_i$ and $\symbfit{y}_j$ are
  independent for all repeats $i \ne j$
- **Dependent**: All predictions depend on (are partly determined by) some
  source prediction $\symbfit{y}_{\text{base}}$, which may or not be similar
  to the true labels $\symbfit{y}$

Finally, both the true labels and predicted labels can have different distributions.
Without loss of generality, for $c$ classes, we can sort the **class probabilities** :

$$
p = [p_1 \ge p_2 \ge \dots \ge p_c]
$$
The label distribution determines the kind of dataset

