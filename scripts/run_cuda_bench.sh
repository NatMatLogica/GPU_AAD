#!/bin/bash
export NUMBA_ENABLE_CUDASIM=1
./venv/bin/python -m model.simm_portfolio_cuda "$@"
