import numpy as np

nx = 100 # width
ny = 500 # max height

num_strains = 2

ic = np.random.randint(0, num_strains, size=nx, dtype=np.int32)

ic[:] = 0
ic[40:60] = 1

v = np.array([1.0, 1.01], dtype=np.float)

seed = np.random.randint(2**32 - 1)

import rough_front_expansion.cython as rfe

sim = rfe.Rough_Front(nx=nx, ny=ny, num_strains=2, v=v, ic=ic, seed=seed)

sim.run(sim.max_iterations)