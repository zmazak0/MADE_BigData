#!/usr/bin/env python
import sys
from csv import reader

counter = 0
mean = 0
variance = 0

for line in reader(sys.stdin):
    try:
        cur_price = float(line[9])
        variance = counter * variance / (counter + 1) + counter * ((mean - cur_price) / (counter + 1)) ** 2
        mean = (counter * mean + cur_price) / (counter + 1)
        counter = counter + 1
    except:
        continue
print("{}\t{}\t{}".format(counter, mean, variance)