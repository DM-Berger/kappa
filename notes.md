
# `scikit-learn` Source Code

The source code for Cohen's Kappa, simplified to the case without sample
weights, is:

```python
confusion = confusion_matrix(y1, y2, labels=labels, sample_weight=sample_weight)
n_classes = confusion.shape[0]
sum0 = np.sum(confusion, axis=0)
sum1 = np.sum(confusion, axis=1)
expected = np.outer(sum0, sum1) / np.sum(sum0)

# For c = n_classes, below is equivalent to:
#     w_mat = np.ones([c, c]) - np.eye(c)
w_mat = np.ones([n_classes, n_classes], dtype=int)
w_mat.flat[:: n_classes + 1] = 0

k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
return 1 - k
```

If we define:
```python
def nondiag(x: ndarray) -> ndarray:
    return x[~np.ones(x.shape[0],  dtype=bool)]
```

then the above code becomes:

```python
confusion = confusion_matrix(y1, y2, labels=labels, sample_weight=sample_weight)
n_classes = confusion.shape[0]
sum0 = np.sum(confusion, axis=0)
sum1 = np.sum(confusion, axis=1)
expected = np.outer(sum0, sum1) / np.sum(sum0)
k = np.sum(nondiag(confusion)) / np.sum(nondiag(expected))
return 1 - k
```



In math: let $\mathbf{C}$ be confusion matrix, let $c$ by number of classes,
let $\mathbf{I}_c$ be $c \times c$ identity matrix, and let $\mathbf{1}^{m \times n}$ be
the matrix with entries all equal to 1.  Also note that for a matrix
$\mathbf{M} \in \mathbb{R}^{c \times c}$ we can write:

$$
\texttt{sum}(\mathbf{M}, \texttt{axis=0}) =
\mathbf{M} \cdot \begin{bmatrix} 1 \\ \vdots \\ 1 \end{bmatrix}
=  \mathbf{M} \cdot \mathbf{1}^{c \times 1} \in \mathbb{R}^{c \times 1}
$$

and

$$
\texttt{sum}(\mathbf{M}, \texttt{axis=1}) =
\begin{bmatrix} 1 \;  \dots \; 1 \end{bmatrix} \cdot \mathbf{M}
=  \mathbf{1}^{1 \times c} \cdot \mathbf{M} \in \mathbb{R}^{1 \times c}
$$

Also, for any square $n \times n$ matrix $\mathbf{M}$ we can write:

$$
\text{nondiag}(\mathbf{M}) = \mathbf{M} - \mathbf{I}_n
$$

Then we have, denoting `expected` above as $\mathbf{E}$:

$$\begin{align*}

\mathbf{E} &= \texttt{sum}(\mathbf{M}, \texttt{axis=0}) \cdot \texttt{sum}(\mathbf{M}, \texttt{axis=1}) \in \mathbb{R}^{c \times c}\\

&= (\mathbf{M} \cdot \mathbf{1}^{c \times 1}) (\mathbf{1}^{1 \times c} \cdot \mathbf{M}) \\

&= \mathbf{M} \cdot \mathbf{1}^{c \times c} \cdot  \mathbf{M} \\



\mathbf{\tilde{E}} &= \text{sum}(\mathbf{M})^{-1} \cdot \mathbf{E}  \\
\kappa(\mathbf{M}) &= \text{sum}\left(\text{nondiag}(\mathbf{M})\right) / \mathbf{\tilde{E}}  \\
\kappa(\mathbf{M}) &=\frac{\text{sum}(\mathbf{M}) \cdot \text{sum}\left(\text{nondiag}(\mathbf{M})\right)}{\text{sum}\left(\text{nondiag}(\mathbf{M} \cdot \mathbf{1}^{c \times c} \cdot  \mathbf{M})\right)}  \\

\end{align*}$$