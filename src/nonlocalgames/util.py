from typing import Dict, TypeVar, Tuple
from functools import partial

def load_seeds(path = 'data/seeds.txt'):
    '''Loads some random seeds'''
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    seed_func = lambda l: int(l.strip())
    seeds = list(map(seed_func, lines))
    
    return seeds

T = TypeVar('T')

def from_ket_form(counts: Dict[str, T]) -> Dict[Tuple[int], T]:
    '''Transforms the qiskit counts vector from ket form with string keys
    to tuples of integer keys
    
    Example:
        >>> d = {'01 10': 7}
        >>> from_ket_form(d)
            {(2, 1): 7}
    '''
    postprocessed = {}
    from_bin = partial(int, base=2)
    for bitstring, count in counts.items():
        # Transform the binary strings back into integers per player.
        # Qiskit will output a bit string in the format
        # 'bn bn-1 ... b0', where bi is the bit string for classical register i.
        # Additionally, the bits are reversed, i.e. in little endian order
        # with the MSB on the left.
        answers = tuple(map(from_bin, reversed(bitstring.split(' '))))
        postprocessed[answers] = count
    
    return postprocessed
