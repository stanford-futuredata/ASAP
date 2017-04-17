#!/usr/bin/env bash

[ -n "$(which pytest)" ] || (set -x; pip install pytest)

set -ex
./ASAP.py -ji Taxi.csv -o test.csv
pytest -v
