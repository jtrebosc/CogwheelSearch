#!/usr/bin/env python3
# -*- coding: utf8 -*- 
# This code is published under 
#                    GNU GENERAL PUBLIC LICENSE
#                       Version 3, 29 June 2007
# Copyright Julien TREBOSC 2021
#



import numpy as np
import sys
import itertools
import multiprocessing as mp
from numba import jit, njit

WILDCARD = '-999'      # integer that represents a wild card for managing forbidden_pathways wildcard

allowed_parameters = { # key is name of parameter, value is dimension : 0 is scalar, 1 is list, 2  is list of list
    'allowed_coherences':2,
    'required_pathways':2,
    'allowed_pathways':2,   
    'forbidden_pathways':2,   
    'cycled_pulses':1,
    'unwanted_path_max':0,
    'COGN_min':0,
    'COGN_max':0,
    'COGN_step':0,
    'max_core':0,
}

def read_input_file(filename):
    """Read the input file to define input parameters
    ----- arguments -----
    filename : the name of the file to be parsed
    ----- result -----
    a dictionnary with the parameters read from the file:
    allowed_coherences (list of list)
    required_pathways (list of list)
    allowed_pathways (list of list)  with wild cards
    forbidden_pathways (list of list)   
    cycled_pulses (list)
    unwanted_path_max (int)
    COGN_min (int)
    COGN_max (int)
    COGN_step (int)
    
    """
    with open(filename, 'r') as f:
        return f.readlines()

def readpar(lines, i):      
    """A parameter is identified on lines[i]. This function reads the value possibly spanning on several lines 
       returns the name and value of the parameter and updated index the line counter in the lines 
    """
    values = []
    name, value = lines[i].split('=') # at that point there must be a = sign in line
    name = name.strip()
    # replace '*' by WILDCARD in value
    value = value.replace('*', WILDCARD)
    if len(value) > 0: # there is a value after =
        if allowed_parameters[name] == 0 :
            return name, int(value)
        elif allowed_parameters[name] == 1:
            return  name, [int(v) for v in value.split()]
        elif allowed_parameters[name] == 2:
            values.append([int(v) for v in value.split()])
    j = i+1
    while  (j < len(lines)) and ('=' not in lines[j]) and (lines[j] != ''):
#        print(i, len(lines)) 
        # replace '*' by WILDCARD in line
        lines[j] = lines[j].replace('*', WILDCARD)
        value = [int(v) for v in lines[j].split()]
        j += 1
        values.append(value)

    return name, values

def remove_comments(lines):
    """Removes comments in an array of strings. 
    Array elements starting with # are removed, strings with # are stripped.
    input : array of strings
    output : new array of strings with comments removed
    """
    new_lines = []
    for line in lines:
        if line.startswith('#') : continue # a hash starting a line : the whole line doesn't count
        #note that is space or tab is before hash then it is en empty line with a comment not a comment line
        if line.startswith('ENDPAR========'): break
        new_lines.append(line.split('#')[0].strip()) # all lines are stripped 
    return new_lines

def parse_input_paramters(lines):
    """
    input : the lines from parameter file
    ----- result -----
    a dictionnary with the parameters read from the file:
    allowed_coherences (list of list) : Allowed coherences for every evolution used to calculate all possible pathways
    required_pathways (list of list) : pathways that must be allowed by cogwheel solution
    allowed_pathways  (list of list) :pathways that are not required but can be allowed because expected efficiency is very weak
    cycled_pulses (list) : The pulses for which windings numbers are searched. if 0, the winding is fixed to 0
    unwanted_path_max (int) : maximum number of unwanted pathway (neither required nor allowed)
    COGN_min (int) : initial number of phases searched
    COGN_max (int) : maximum number of phases searched 
    COGN_step (int): increment of number of phases searched 
   """  
    parameters = {}
    # strip any comment
    uncommented_lines = remove_comments(lines)
    for i in range(len(uncommented_lines)):
        if '=' in uncommented_lines[i]:
            parameter, value = readpar(uncommented_lines, i)
            if parameter in allowed_parameters:
                parameters[parameter] = value
    
    return parameters

sample_input_file = """
# Comment  lines starting with # (# as first character) are fully ignored
# Any character after # on a line is discarded. If spaces are present before 
# then it corresponds to an empty line NOT a comment line
# parameter starts with parameter_name =
# can span several non empty lines with possibly comment lines interspersed
# an empty line or another parameter name (one word followed by '=' sign ends the current 
# parameter definition

allowed_coherences=     # allowed coherences array for pathways filtering
# p1 p2 p3 P4          
 0  0  0  1  0    # 3Q 
 0  0  0  1  0    # 2Q 
 0  1  1  1  1    # 1Q 
 1  1  1  1  1    # 0Q 
 0  1  1  1  1    #-1Q 
 0  0  0  1  0    #-2Q 
 0  0  0  1  0    #-3Q 

required_pathways=      # required pathway that must be allowed : one pathway per line
 0  1 -1  3 -1
 0 -1  1 -3  1
allowed_pathways=       # allowed pathways that are not required
 0 -1  1  3 -1
 0  1 -1 -3  1
forbidden_pathways=       # allowed pathways that are not required
 *  *  *  *  0            # forbidden pathways can use * wild card: here any pathway finishing on 0 is excluded

cycled_pulses =  1 1 0 1 # pulses that are cycled (0 means the pulse winding will be fixed to 1) 
                         #                        (1 means the all winding are searched for the pulse)
 # this is an empty line. NOT a pure comment line
unwanted_path_max = 0   # max allowed additionnal unwanted pathway in valid winding
COGN_min =  10  # starting N for solution search
COGN_max = 24   # maximum N for soluiton search
COGN_step = 2   # step increment for N 
max_core = 4    # optionnal to limit the number of parallel processes launched during calculation

ENDPAR========  ========   ENDPAR string with 8 = sign marks the end of parameters. 
Results will be stored after that mark. Any previously stored results will be deleted
"""

@njit
def path_is_allowed(windings, deltaP, N):
    """ Check if a deltaP pathway is allowed by windings numbers.
windings : a 1D ndarray of size N containing the cogwheel winding numbers of 
the different pulses and receiver
deltaP: a 1D ndarray of size N containing the coherence quantum jumps at each 
pulse and 1 as last element for receiver
N the cogwheel main number (phases are multiple of 2 pi /N)
The path is allowed if receiver winding is such that (Wrec+sum(Wi*DPi))%N == 0
With Wi and DPi the winding number and deltaP of pulse i
"""
    return (windings*deltaP).sum()%N == 0

@njit
def required_deltaP_allowed(npwindings, required_deltaP, N):
    """Check if winding numbers allow required pathway coherence jumps (deltaP) """
    for i in range(required_deltaP.shape[0]):
        if not path_is_allowed(npwindings, required_deltaP[i], N):
            return False
    return True

@njit
def calc_deltaP(path):
    """ Returns the pathway coherence jumps (deltaP) of a given coherence pathway """
    return np.array([path[i+1] - path[i] for i in range(len(path)-1)])

#@njit
def path_is_forbidden(path, forbidden_pathways):
    """ Return True if path match one of the forbidden pathways. """
    (num_path, num_path_elem) = forbidden_pathways.shape 
    for i in range(num_path):
        for j in range(num_path_elem):
            if forbidden_pathways[i,j] == int(WILDCARD): # A wild card means True so skip element
                continue
            if path[j] !=  forbidden_pathways[i,j]:  # This element doesn't match a forbidden path line
                break
        else: # If for loop didn't break then no False is found meaning that a fobidden pathway is found
#            print(f"Path {path} is among forbidden ones\n {forbidden_pathways}\n")
            return True
#    print(f"Path {path} is not among forbidden ones\n {forbidden_pathways}\n")
    return False   # all lines screened triggered a break (path didn't match any forbidden pathways)

def check_windings(windings, required_deltaP, N, convd_allowed_coh, required_pathways, allowed_pathways, forbidden_pathways, unwanted_path_max):
    """A function to calculate if a winding number complies with conditions about required, allowed or forbidden pathways
    and number of maximum unwanted pathway
    """
    npwindings = np.ones(len(windings)+1, dtype=int)
    npdeltaP   = np.ones(len(windings)+1, dtype=int)
#            print(f"Checking if windings {windings} is valid")
    npwindings[:-1] = windings
    # set phase winding according to first required pathway
    npwindings[-1] = -(npwindings[:-1]*required_deltaP[0][:-1]).sum() % N
    #check that other required pathways are allowed
    if not required_deltaP_allowed(npwindings, required_deltaP, N) : 
#                print(f"required_deltaPs {required_deltaP} is not allowed by windings {windings}")
#                print(f"indeed : {(npwindings*required_deltaP[0]).sum()%N } is not 0")
        return None
#            print(f"required_deltaP {required_deltaP} is allowed by windings {windings}")
#            print(f"indeed : {(npwindings*required_deltaP[0]).sum()%N } is 0")

    allowed = [] 
    unwanted = []
    unwanted_path_count = 0

    for path in itertools.product(*convd_allowed_coh):
        #check if this path is among required ones
        path_list = list(path)
        if path_list in required_pathways:
#            print(f"path {path} is among required ones")
            continue
#        print(f"path {path} is not among required ones: is it allowed ?")
        npdeltaP[0:-1] = calc_deltaP(path)
        if path_is_allowed(npwindings, npdeltaP, N):
#            print(f"path {path} is allowed")
            if path_list in allowed_pathways:
#                print(f"{path_list} is in {allowed_pathways}\n")
                allowed.append(path)
            else:
                if path_is_forbidden(np.array(path_list), forbidden_pathways):
#                    print(f"Path {path_list} is forbidden\n")
                    return None
                unwanted.append(path)
                unwanted_path_count += 1
                if unwanted_path_count > unwanted_path_max:
#                    print(f"exceeded allowed path count: {unwanted_path_count}")
                    return None
    # path for loop didn't break : winning winding!!
    return (npwindings.tolist(), unwanted, allowed)
#                print("wining windings !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#                print(npwindings, unwanted)
    
def check_windings_wrap(args_list):
    """Just a wrap function that take a single list argument to allow 
       using Pool.imap for parallelization and keyboard interrupt
    """
    try:
        return check_windings(*args_list )
    except KeyboardInterrupt:
        return

def search(stats):
    """ The search function that loops on different N and on corresponding possible windings in parallel manner 
    Note that it relies on global parameter variables:
    max_core, N, num_of_pulses, ..."""
    # initialise some numpy arrays
    pool = mp.Pool(processes=max_core)
    from itertools import product as prod
    from itertools import repeat as rep
    for N in range(COGN_min, COGN_max+1, COGN_step):
        print(f"Search cogwheel windings for N={N}")
        stats[N] = {'count': 0}
        # for windings known var : num_of_pulse, required_deltaP, convd_allowed_coh, N
        searched_windings = [list(range(N))]*num_of_pulses
        #skip the pulse that are not being windinged
        for pulse_idx in range(num_of_pulses):
            if  cycled_pulses[pulse_idx] == 0:
                searched_windings[pulse_idx] = [0]

        #parallelized for loop 
        for result in pool.imap_unordered(check_windings_wrap, 
                       zip(prod(*searched_windings), rep(required_deltaP), 
                           rep(N), rep(convd_allowed_coh), rep(required_pathways.tolist()), 
                           rep(allowed_pathways.tolist()), rep(forbidden_pathways), 
                           rep(unwanted_path_max)), 500):
#        for windings in itertools.product(*searched_windings):
#            result = check_windings(windings, required_deltaP, N, convd_allowed_coh, required_pathways.tolist())
            if result is not None:
                win_wind, unwanted, allowed = result
                stats[N]['count'] += 1
                stats[N][tuple(win_wind)] = [unwanted, allowed]
                
        print(f"Found {stats[N]['count']} valid windings for COG {N}")

def print_results(stats, filename):
    """A function to print the results stored in dictionnary stats"""
#    print(f"For required pathes \n {'\n'.join([' '.join(l) for l in required_pathways]}")
    lines = read_input_file(filename)
    for i in range(len(lines)):
        if lines[i].startswith('ENDPAR========'): 
            break
    lines = lines[0:i+1]
    lines.append("Solutions found are: \n")
    for N, dic in stats.items():
        if dic['count'] == 0: continue
        for windings, uwp in dic.items():
            if windings == 'count': continue
            lines.append(f"COG {N} -> windings {':'.join([str(i) for i in windings])}\n")
            lines.append(f"         allowed pathways allowed {uwp[1]} \n") 
            lines.append(f"         unwanted pathways allowed {uwp[0]} \n") 
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line)

if __name__ == "__main__" :
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        lines = read_input_file(filename)
        parameters = parse_input_paramters(lines)
        allowed_coherences = np.array(parameters['allowed_coherences'])
        required_pathways = np.array(parameters['required_pathways'])
        if 'forbidden_pathways' in parameters.keys():
            forbidden_pathways = np.array(parameters['forbidden_pathways'])
        else : 
            forbidden_pathways = np.empty(shape=(0,0))
        if 'allowed_pathways' in parameters.keys():
            allowed_pathways = np.array(parameters['allowed_pathways'])
        else : 
            allowed_pathways = np.empty(shape=(0,0))
        cycled_pulses = np.array(parameters['cycled_pulses'])
        unwanted_path_max = parameters['unwanted_path_max']
        COGN_min =  parameters['COGN_min']
        COGN_max = parameters['COGN_max']
        COGN_step = parameters['COGN_step']
        max_core = parameters['max_core']
    else:
        # inputs the expected possible coherences
        allowed_coherences = np.array([
        # p1 p2 p1 pf pf pc            
        [0 ,0, 0, 1, 0,],  # 3Q     0, 
        [0 ,0, 0, 1, 0,],  # 2Q     0, 
        [0 ,1, 1, 1, 1,],  # 1Q     0, 
        [1 ,1, 1, 1, 1,],  # 0Q     1, 
        [0 ,1, 1, 1, 1,],  #-1Q     0, 
        [0 ,0, 0, 1, 0,],  #-2Q     0, 
        [0 ,0, 0, 1, 0,],  #-3Q     0, 
        ], dtype=int)

        # specify the coherence pathway to authorize
        required_pathways = np.array([
        [0, -1, 1,-3, 0,],
        [0,  1,-1, 3, 0,],
        ], dtype=int)

        allowed_pathways = np.array([
        [0, -1, 1,-3, 0,],
        [0,  1,-1, 3, 0,],
        ], dtype=int)
        # ,0, 0,
        # ,0, 0,
        forbidden_pathways = np.array([[]], dtype=int)

        # p1 p2 p1 pf pf pc            
        cycled_pulses = np.array([
          1, 1, 1, 1, 
        ]) # 0 if winding is not searched (winding fixed to 0), 1 if winding searched up to COGN 

        unwanted_path_max = 0
        COGN_min =  10
        COGN_max = 24
        COGN_step = 2
        max_core = 4

    required_deltaP = np.ones_like(required_pathways) 
    required_deltaP[:,0:-1] = required_pathways[:,1:] - required_pathways[:,:-1]
    wdg_len = required_deltaP.shape[-1]
    num_of_pulses = wdg_len - 1

    max_Q= allowed_coherences.shape[0]//2
    # convd_allowed_coh to contain a list of coherence levels allowed between pulses
    convd_allowed_coh = [[max_Q - coh for coh in 
                          np.flatnonzero(allowed_coherences[:,col])] 
                                             for col in range(wdg_len)]

    stats = {}
    try:
        search(stats)
        print_results(stats, filename)
    except KeyboardInterrupt:
        print_results(stats, filename)
        sys.exit()

