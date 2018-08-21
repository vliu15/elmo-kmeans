#!/bin/bash

cat sentences.txt | tr A-Z a-z | sed 's/[.,!?]/ &/' > clean.txt
python clean.py
