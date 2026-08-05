#!/usr/bin/env python3
"""Check scripts/jaxlie_shim against the real jaxlie.

The container cannot run this -- that is the whole point of the shim, jax has
no cp38 wheels. Run it on any Python >= 3.9 host:

    python3 -m venv /tmp/jaxvenv
    /tmp/jaxvenv/bin/pip install jaxlie scipy numpy
    /tmp/jaxvenv/bin/python scripts/verify_jaxlie_shim.py

x64 has to be on before jax is touched, otherwise jaxlie computes in float32
and the comparison bottoms out at ~1e-5 for reasons that have nothing to do
with the shim.
"""
import importlib.util
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jaxlie as real          # noqa: E402
import numpy as np             # noqa: E402

SHIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jaxlie_shim", "jaxlie", "__init__.py")
spec = importlib.util.spec_from_file_location("jaxlie_shim", SHIM)
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)

rng = np.random.default_rng(0)
max_exp = max_log = max_rt = 0.0
for k in range(500):
    # first few cases pin down the small-angle branch
    if k == 0:
        omega = np.zeros(3)
    elif k < 3:
        omega = rng.normal(size=3) * (1e-9 if k == 1 else 1e-5)
    else:
        omega = rng.normal(size=3) * rng.uniform(0.0, 3.0)
    tangent = np.concatenate([rng.normal(size=3) * 5.0, omega])

    T_real = np.asarray(real.SE3.exp(tangent).as_matrix())
    max_exp = max(max_exp, np.abs(T_real - shim.SE3.exp(tangent).as_matrix()).max())

    l_real = np.asarray(real.SE3.from_matrix(T_real).log())
    l_shim = shim.SE3.from_matrix(T_real).log()
    max_log = max(max_log, np.abs(l_real - l_shim).max())
    max_rt = max(max_rt, np.abs(shim.SE3.exp(l_shim).as_matrix() - T_real).max())

print("max |exp_real - exp_shim| : {:.3e}".format(max_exp))
print("max |log_real - log_shim| : {:.3e}".format(max_log))
print("max round-trip error      : {:.3e}".format(max_rt))
sys.exit(0 if max(max_exp, max_log, max_rt) < 1e-7 else 1)
