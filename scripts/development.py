"""Supported guarded entry point; no new training is authorised by this command."""
import argparse
import runpy
from src.development_guard import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', choices=['src.training.train_crf', 'src.evaluation.evaluate_crf'], required=True)
    args, rest = parser.parse_known_args()
    install()
    import sys
    sys.argv = [args.module, *rest]
    runpy.run_module(args.module, run_name='__main__')


if __name__ == '__main__':
    main()
