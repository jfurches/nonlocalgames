import pytest
import numpy as np

from qinfo import *

class TestCheckFunctions:
    def test_hermicity(self):
        H = 1 / np.sqrt(2) * np.array([
            [1, 1],
            [1, -1]
        ])

        assert is_hermitian(H)
        assert not is_antihermitian(H)
        assert is_antihermitian(1j * H)
    
    def test_diagonal(self):
        H = 1 / np.sqrt(2) * np.array([
            [1, 1],
            [1, -1]
        ])

        assert is_diagonal(I)
        assert not is_diagonal(H)

class TestRotations:
    def test_ry(self):
        psi = np.array([1, 0])
        R = Ry(np.pi/2)
        psi_p = 1 / np.sqrt(2) * np.array([1 + 0j, 1])

        # Check that Ry(pi/2) = H
        rotated = (R @ psi)
        assert np.allclose(rotated, psi_p)

        # Check unitary for several random parameters
        for phi in np.random.normal(scale=np.pi/2, size=5):
            assert np.allclose(Ry(-phi), Ry(phi).conj().T)
            assert np.allclose(Ry(-phi) @ Ry(phi), np.eye(2, dtype=complex))
    
class TestMeasurement:
    @pytest.mark.parametrize('n', [1,2,3,4,5])
    def test_Miq(self, n, trials=5):
        d = 2 ** n
        total_op = np.zeros((d, d), dtype=complex)
        for _ in range(trials):
            phi = np.random.normal(scale=np.pi/2)
            
            # If this isn't true, everything else fails
            Ai = Z
            assert is_hermitian(Ai)
        
            # Assert that unitary rotation preserves hermicity
            Ai = Ry(-phi) @ Ai @ Ry(phi)
            assert is_hermitian(Ai)

            # Check that embedding the operator in a larger space preserves
            # hermicity
            A = tensor_i(Ai, np.random.choice(n), n)
            assert is_hermitian(A)
    
            total_op += A
        
        # The sum of all these hermitian ops should also be hermitian
        assert is_hermitian(total_op)