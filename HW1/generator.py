""" script to generate simulated dataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

with open('data.txt', mode='w') as file_:
    for i in range(1, 1000):
        file_.write('{},{}\n'.format(i, i*12))
