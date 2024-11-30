from alternative_phase_1 import create_bus_admittance_matrix
import pytest
import pickle
import numpy as np

def test_case_example():
    with open('example_y_bus.pkl', 'rb') as file:
        y_busTrue = pickle.load(file)
    y_bus = create_bus_admittance_matrix('example.json')

    tolerance = 1e-8
    difference = y_busTrue.tocsr() - y_bus.tocsr()
    assert np.all(np.abs(difference.data) < tolerance), "Matrices are not equal within the given tolerance."

test_case_example()

