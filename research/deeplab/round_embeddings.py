"""
Round precomputed pixel instance embeddings and evaluate performance.
"""

import argparse
import numpy as np


def batch_eval(args):
    """
    :param args:
    """



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_dir",
                        help='Path to directory containing instance ID PNG files. Files must match pattern'
                             '{imagename}_instanceIds.png, e.g. frankfurt_000001_080091_instanceIds.png')

    parser.add_argument("log_dir",
                        help='Path to directory containing pixel embedding files. Files must match pattern'
                             '{imagename}.npy')

    args = parser.parse_args()

    batch_eval(args)