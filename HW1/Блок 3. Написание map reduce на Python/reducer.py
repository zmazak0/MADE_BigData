#!/usr/bin/env python
import sys

counter = 0
mean_itog = 0
variance_itog = 0

for line in sys.stdin:
    cnt, mean, variance = line.strip().split('\t')
    variance_itog = (counter * variance_itog + cnt * variance) / (counter + cnt) + counter * cnt * ((mean_itog - mean) / (counter + cnt)) ** 2
    mean_itog = (counter * mean_itog + cnt * mean) / (counter + cnt)
    counter = counter + cnt
print("{}\t{}".format(mean_itog, variance_itog))