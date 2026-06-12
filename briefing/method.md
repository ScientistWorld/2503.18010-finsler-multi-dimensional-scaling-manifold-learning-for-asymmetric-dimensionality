# Paper's Method

## Key Contribution

The paper generalizes multidimensional scaling to asymmetric dissimilarities by embedding data in a canonical Finsler, specifically Randers, space instead of a symmetric Riemannian/Euclidean space.

## Approach

The canonical embedding space keeps straight-line geodesics but assigns direction-dependent lengths:
`d(x, y) = ||y - x||_2 + omega^T (y - x)`, with a constant drift vector `omega` of norm below one. The paper minimizes a Finsler stress objective and derives a SMACOF-style majorization update. For larger examples it uses a stabilized linear solve based on multiplying the update equation by `V^T` and solving with GMRES.

The paper also presents a neural representation-learning version for larger directed graphs, using the same canonical distance in a Fermi-Dirac decoder for link prediction.

## Main Claims

Primary claim: low-dimensional embeddings with directed distances preserve asymmetric dissimilarities better than Euclidean embeddings.

Secondary claims: the same framework supports intuitive visualizations of asymmetric manifolds, current maps, and graph hierarchies; it is compatible with robust MDS weighting schemes; and the representation-learning variant improves directed graph reconstruction and link prediction.
