from   time import time
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(axis = -1, keepdims = True)) # numerical stability
    return e / e.sum(axis = -1, keepdims = True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def convert_time(time):
    m, s = divmod(round(time), 60)
    h, m = divmod(m, 60)
    return f'{h:.0f}h {m:02.0f}m {s:02.0f}s'

def check_verbose(N, verbose):
    if isinstance(verbose, str):
        if verbose == 'auto':
            temp = 10 ** np.floor(np.log10(N) - 1)
            k    = np.array([1, 2, 5])
            arg  = np.fabs(N // temp / k - 10).argmin()
            verbose = int(temp * k[arg])
        else:
            raise Exception()
    elif isinstance(verbose, str):
        verbose = int(N * verbose + 0.5)

    assert isinstance(verbose, int)
    assert 0 < verbose < N

    return verbose

class Verbose():

    def __init__(self, N, verbose):
        self.N       = N
        self.verbose = verbose
        self.num     = max(len(f'{N:,d}'), len('iteration'))
        self.start   = time()

        space        = max(len('iteration') - self.num, 0)
        
        print(' ' * space + 'iteration | log pstar')
        print('-' * space + '----------+----------')

    def print(self, i, val, end = '\n'):
        if i > 0:
            elapsed = time() - self.start
            average = elapsed / (i + 1)
            eta     = (self.N - i) * average

            elapsed = f'    elapsed {convert_time(elapsed)}'
            eta     = f'    eta {convert_time(eta)}'
        else:
            elapsed = ''
            eta     = ''
        print(f'\r{i:>{self.num},d} | {val:+.2e}{elapsed}{eta}', end = end)