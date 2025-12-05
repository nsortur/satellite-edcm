import escnn
import numpy as np

g = escnn.group.octa_group()
unit_vec = np.array([0.7071067811865476, 0, 0.7071067811865475])
unique_vectors = []
tolerance = 1e-10

for element in g.elements:
    rot = g.standard_representation(element)
    rotated_vec = np.dot(rot, unit_vec)
    
    rounded_vec = np.round(rotated_vec, decimals=10)
    
    is_unique = True
    for existing_vec in unique_vectors:
        if np.allclose(rounded_vec, existing_vec, atol=tolerance):
            is_unique = False
            break
    
    if is_unique:
        unique_vectors.append(rounded_vec)

for vec in unique_vectors:
    print(vec)
