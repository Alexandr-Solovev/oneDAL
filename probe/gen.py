import numpy as np, sys
g = np.random.default_rng(0)
groups = [((0, 0, 0), 0.3, 60), ((3, 0, 1), 0.6, 60), ((0, 4, -1), 1.0, 60), ((4, 4, 4), 0.4, 40)]
X = np.concatenate([g.normal(c, s, size=(n, 3)) for c, s, n in groups] + [g.uniform(-3, 7, size=(20, 3))])
X.astype(np.float64).tofile(sys.argv[1])
