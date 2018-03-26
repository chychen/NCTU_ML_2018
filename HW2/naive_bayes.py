from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import argparse
from utils import load_mnist


def main():
    if ARGS.mode == 'discrete':
        pass
    elif ARGS.mode == 'continuous':
        pass
    else:
        raise ValueError('{} is not a valid mode'.format(ARGS.mode))
    return

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("mode", type=str,
                        help="\'discrete\' or \'continuous\'")
    ARGS = PARSER.parse_args()
    
    main()