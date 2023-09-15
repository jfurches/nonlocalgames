from nonlocalgames import methods
from nonlocalgames.hamiltonians import Ramsey

if __name__ == '__main__':
    ham = Ramsey(measurement_layer='ry')
    results = methods.dual_phase_optim(
        ham,
        tol=1e-6,
        adapt_thresh=1e-2,
        theta_thresh=1e-6)