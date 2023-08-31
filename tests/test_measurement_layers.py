import itertools

import pytest
import numpy as np

from nonlocalgames.measurement import MeasurementLayer
from nonlocalgames.qinfo import is_unitary

class TestMeasurement:
    @pytest.mark.parametrize('layer', ('ry', 'cnotry', 'u3', 'u10'))
    def test_layers(self, layer):
        ml = MeasurementLayer.get(layer, 2, 14, 2)
        ml.phi[:] = np.random.normal(size=ml.phi.shape)

        for va, vb in itertools.product(range(14), repeat=2):
            Uq = ml.uq((va, vb))
            assert is_unitary(Uq)
