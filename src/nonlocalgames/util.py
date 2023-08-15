def load_seeds(path = 'data/seeds.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    seed_func = lambda l: int(l.strip())
    seeds = list(map(seed_func, lines))
    
    return seeds