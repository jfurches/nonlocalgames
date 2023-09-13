from nonlocalgames import methods, util
from nonlocalgames.hamiltonians import Ramsey

if __name__ == '__main__':
    ham = Ramsey(measurement_layer='ry')
    results = methods.dual_phase_optim(
        ham,
        tol=1e-6,
        adapt_thresh=1e-3,
        theta_thresh=1e-6)