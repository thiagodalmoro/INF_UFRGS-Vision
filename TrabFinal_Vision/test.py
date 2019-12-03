import numpy as np
from scipy import spatial

# test points
pts = np.random.rand(100_000, 2)
print(pts)
# two points which are fruthest apart will occur as vertices of the convex hull
candidates = pts[spatial.ConvexHull(pts).vertices]

# get distances between each pair of candidate points
dist_mat = spatial.distance_matrix(candidates, candidates)

# get indices of candidates that are furthest apart
i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

print(candidates[i], candidates[j])
