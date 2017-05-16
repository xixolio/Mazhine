#!/bin/bash
#PBS -l cput=1000:00:00
#PBS -l walltime=1000:00:00

use anaconda2
use gcc48
python /user/i/iaraya/files/Mazhine/proyecto2/cv.py $1 $2