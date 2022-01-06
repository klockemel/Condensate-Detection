#!/bin/bash

python concatData.py

mkdir data
mkdir compImgs
mkdir runvalues

mv *ues.csv runvalues/
mv *pimg.png compImgs/
mv *data.csv data/
