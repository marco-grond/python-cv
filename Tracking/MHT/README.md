# MHT
A Python implementation of a Multiple Hypothesis Tracker following the implementation details and optimizations proposed by Cox and Hingorani [[1]](#1).

## Execution
And example can be run by executing:
```
python MHT.py [detection_file]
```
where `[detection_file]` can be replaced with any text file of the correct format. The format for the detections are given as:
```
t_1
x_1 y_1
...
x_n y_n

t_2
x_1 y_1
...
x_n y_n

...

t_m
x_1 y_1
...
x_n y_n
```
where `t_1, t_2, ..., t_m` are the times at which detections were made and `x_n and y_n` are the object detections for each associated time.

## References
<a id="1">[1]</a> 
Cox, I. J., and Hingorani, S. L. (1996). 
An efficient implementation of Reid's multiple hypothesis tracking algorithm and its evaluation for the purpose of visual tracking.
IEEE Transactions on Pattern Analysis and Machine Intelligence.
