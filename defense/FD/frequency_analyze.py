import argparse
from dct_analyze import dct_result
from fft_analyse import fft_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument(
        '--path',
        type=str,
        default='/home/chengyiqiu/code/INBA/results/cifar10/inba/convnext/20241128150334'
    )
    parser.add_argument(
        '--total',
        type=int,
        default=1024
    )
    args = parser.parse_args()
    dct_result(args)
    fft_result(args)