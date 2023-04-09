# CogwheelSearch
A python program that searches for valid cogwheel phase cycling for NMR pulse sequences
Cogwheel phase cycling is described in the following reference.
[Levitt, M. H.; Madhu, P. K.; Hughes, C. E. Cogwheel Phase Cycling. 
Journal of Magnetic Resonance 2002, 155 (2), 300–306. ](
https://doi.org/10.1006/jmre.2002.2520.)

CogwheelSearch does a brute force search of all possible solutions that satisfy
given constraints like :
* required pathways
* allowed coherence levels 
* maximum number of unwanted pathway allowed

The input file and search procedure is mainly inspired by [CCCP++ v1.2](https://github.com/ajerschow/CCCP-pp) from Alexej Jerschow. 
[J. Magn. Reson. 160, 59-64, (2003).](http://dx.doi.org/10.1016/S1090-7807(02)00031-9)
However CogwheelSearch does not analyse existing phase cycling.

Written in python CogwheelSearch is intrinsically slower than CCCP-pp 
however it allows to define several required pathways and implements 
(limited) just in time compilation and parallel programming which makes 
it quite useful.

The program uses an input parameter file and write the results at the end of it.

Provided input file should be self explanatory I hope...

Julien TREBOSC

TODO: 
* implement multi channel phase cycling (hmqc, inept etc...)
* improve code efficiency i.e. njit enable check_windings and search functions...
