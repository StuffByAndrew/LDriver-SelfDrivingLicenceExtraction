from itertools import starmap
import numpy as np
A = ['A', 'A', 'A', 'A']
B = ['B', 'B','B','B']
abc = ['A', 'B', 'C']
abc2 = ['A', 'B', 'D']
Ac = np.array([1,2,3,4])
Bc = np.array([0,10,0,10])
i = 0
new = list(starmap((lambda l, l_old, c, c_old: l if c > c_old else l_old), zip(B, A, Bc, Ac)))