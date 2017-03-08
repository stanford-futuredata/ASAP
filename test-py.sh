#!/usr/bin/env bash

set -ex
./ASAP.py -ji Taxi.csv -o test.csv
pytest -v
