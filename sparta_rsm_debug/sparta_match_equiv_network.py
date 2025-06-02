# match the transformation of equivariant OH network output with sparta data
# do mock fit

import escnn
import numpy as np

g = escnn.group.octa_group()
unit_vec = np.array([0.7071067811865476, 0, 0.7071067811865475])
print(g.elements)
print(len(g.elements))
