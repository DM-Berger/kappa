#!/bin/bash
scp demo.py niagara:/gpfs/fs0/scratch/j/jlevman/dberger/kappa/demo.py &
rsync -chavzp niagara:/gpfs/fs0/scratch/j/jlevman/dberger/kappa/plots .