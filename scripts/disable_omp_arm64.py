import platform

import toml


if platform.processor() == 'arm':
    with open('pyproject.toml', 'rt') as f:
        cfg = toml.load(f)
    cfg['tool']['scikit-build']['cmake']['args'] = ['-DUSE_OPENMP=OFF']
    with open('pyproject.toml', 'wt') as f:
        toml.dump(cfg, f)
