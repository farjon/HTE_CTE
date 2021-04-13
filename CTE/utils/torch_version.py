from __future__ import print_function

import torch
import sys

minimum_torch_version = 1,4


def torch_version():
    return tuple(map(int, torch.__version__.split('.')))


def torch_version_ok():
    return torch_version() >= minimum_torch_version


def assert_torch_version():
    detected = torch.__version__
    required = '.'.join(map(str, minimum_torch_version))
    assert(torch_version() >= minimum_torch_version), 'You are using torch version {}. The minimum required version is {}.'.format(detected, required)


def check_torch_version():
    try:
        assert_torch_version()
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)