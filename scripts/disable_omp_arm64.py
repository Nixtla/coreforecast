import os

import tomli


if os.environ["CIBW_BUILD"] == "cp38-macosx_arm64":
    with open("pyproject.toml", "rt") as f:
        cfg = tomli.load(f)
    cfg["tool"]["scikit-build"]["cmake"]["args"] = ["-DUSE_OPENMP=OFF"]
    with open("pyproject.toml", "wt") as f:
        tomli.dump(cfg, f)
