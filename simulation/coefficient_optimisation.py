# ===============================================================================
# @file:    coefficient_optimisation.py
# @note:    This script find optimal washout filter coefficients
# @author:  Ziga Miklosic
# @date:    22.01.2021
# @brief:   This script find optimal washout filter coefficients based
#           on genetic algoritm.
# ===============================================================================

# ===============================================================================
#       IMPORTS  
# ===============================================================================
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import copy

from filters.filter_utils import FunctionGenerator
from system_model import SystemModel

# ===============================================================================
#       CONSTANTS
# ===============================================================================

## ****** USER CONFIGURATIONS ******

## Sample frequency
#   Sample frequency of real system   
#
# Unit: Hz
SAMPLE_FREQ = 50.0

# Ideal sample frequency
#   As a reference to sample rate constrained embedded system
#
# Unit: Hz
IDEAL_SAMPLE_FREQ = 500.0

## Time window
#
# Unit: second
TIME_WINDOW = 1

## Input signal shape
INPUT_SIGNAL_FREQ = 0.1
INPUT_SIGNAL_AMPLITUDE = 9.81/4
INPUT_SIGNAL_OFFSET = INPUT_SIGNAL_AMPLITUDE
INPUT_SIGNAL_PHASE = -0.25

## Mux input signal
INPUT_SIGNAL_SELECTION = FunctionGenerator.FG_KIND_RECT

## Number of samples in time window
SAMPLE_NUM = int(( IDEAL_SAMPLE_FREQ * TIME_WINDOW ) + 1.0 )



# ==================================================
#   WASHOUTE FILTER COEFFICIENTS LIMITS
# ==================================================

# Cutoff frequency limits
WASHOUT_FILTER_FC_MIN_VALUE = 0.001
WASHOUT_FILTER_FC_MAX_VALUE = 5.0

# Damping factor limits
WASHOUT_FILTER_Z_MIN_VALUE  = 0.005
WASHOUT_FILTER_Z_MAX_VALUE  = 3.0


# ==================================================
#   GENETIC ALGORITHM SETTINGS
# ==================================================

# Population size
POPULATION_SIZE = 10

# Number of generations
GENERATION_SIZE = 20

# Mutation propability
MUTATION_PROPABILITY = 0.50

# Mutation impact
# NOTE: Percent of mutation impact on gene change 
MUTATION_IMPACT = 0.05

# Number of elite population
ELITISM_NUM = 2

# Size of tournament
# NOTE: Must not be smaller than population size
TURNAMENT_SIZE = 8

# Crossover propability
CROSSOVER_PROPABILITY = 0.50


# ==================================================
#   STIMULI SIGNAL ROUTE OPTIONS
# ==================================================
INPUT_SIGNAL_ROUTE_TO_AX = 0
INPUT_SIGNAL_ROUTE_TO_AY = 1
INPUT_SIGNAL_ROUTE_TO_AZ = 2
INPUT_SIGNAL_ROUTE_TO_ROLL = 3
INPUT_SIGNAL_ROUTE_TO_PITCH = 4
INPUT_SIGNAL_ROUTE_TO_YAW = 5

INPUT_SIGNAL_ROUTE = INPUT_SIGNAL_ROUTE_TO_AX


## ****** END OF USER CONFIGURATIONS ******

# ===============================================================================
#       CLASSES
# ===============================================================================    

# First order filter coefficients
class Filter1ndOrderCoefficient:
    def __init__(self):
        self.fc = 0

# Second order filter coefficients
class Filter2ndOrderCoefficient:
    def __init__(self):
        self.fc = 0
        self.z = 0    

# Wht washout filter coefficients
class WashoutWhtCoefficients:
    def __init__(self):
        self.x = Filter2ndOrderCoefficient()
        self.y = Filter2ndOrderCoefficient()
        self.z = Filter2ndOrderCoefficient()

# Wrtzt washout filter coefficients
class WashoutWrtztCoefficients:
    def __init__(self):
        self.x = Filter1ndOrderCoefficient()
        self.y = Filter1ndOrderCoefficient()
        self.z = Filter1ndOrderCoefficient()

# W12 washout filter coefficients
class WashoutW12Coefficients:
    def __init__(self):
        self.roll   = Filter2ndOrderCoefficient()
        self.pitch  = Filter2ndOrderCoefficient()

# W11 washout filter coefficients
class WashoutW11Coefficients:
    def __init__(self):
        self.roll   = Filter1ndOrderCoefficient()
        self.pitch  = Filter1ndOrderCoefficient()
        self.yaw    = Filter1ndOrderCoefficient()

# Specimen as one of many in population
class Specimen:
    def __init__(self, Wht, Wrtzt, W11, W12):
        self.Wht    = Wht
        self.Wrtzt  = Wrtzt
        self.W11    = W11
        self.W12    = W12

# Stopwatch - time execution measurement tool
class StopWatch:

    def __init__(self):
        self._time = 0
        self._is_running = False

    def start(self):
        if False == self._is_running:
            self._is_running = True
            self._time = time.time()
        else:
            assert False, "Stopwatch already running!"

    def stop(self):
        if True == self._is_running:
            self._is_running = False
            return time.time() - self._time
        else:
            assert False, "Stopwatch has not been started!"

    def restart(self):
        _time_pass = self.stop()
        self.start()
        return _time_pass

    def time(self):
        return time.time() - self._time



# ===============================================================================
#       FUNCTIONS
# ===============================================================================


# ===============================================================================
# @brief: Generate random intiger array in range (low, high) 
#
# @param[in]:    low            - Minimum random value
# @param[in]:    high           - Maximum random value
# @param[in]:    size           - Size of array
# @return:       array/value    - Array or single value if size = 1
# ===============================================================================
def get_random_int(low, high, size):
    if 1 == size:
        return np.random.randint(low, high, size)[0]
    else:
        return np.random.randint(low, high, size)

    

# ===============================================================================
# @brief: Generate random float array in range (low, high) 
#
# @param[in]:    low            - Minimum random value
# @param[in]:    high           - Maximum random value
# @param[in]:    size           - Size of array
# @return:       array/value    - Array or single value if size = 1
# ===============================================================================
def get_random_float(low, high, size):
    if 1 == size:
        return np.random.uniform(low, high, size)[0]
    else:
        return np.random.uniform(low, high, size)

# ===============================================================================
# @brief: Generate random specimen genes
#
# @param[in]:    fc_low                 - Minimum value of cutoff frequency
# @param[in]:    fc_high                - Maximum value of cutoff frequency 
# @param[in]:    z_low                  - Minimum value of damping factor
# @param[in]:    z_high                 - Maximum value of damping factor 
# @return:       Wht, Wrtzt, W11, W12   - Specimen washout filter coefficients
# ===============================================================================
def generate_specimen_random_gene(fc_low, fc_high, z_low, z_high):

    # Create washout coefficients
    Wht     = WashoutWhtCoefficients()
    Wrtzt   = WashoutWrtztCoefficients()
    W12     = WashoutW12Coefficients()
    W11     = WashoutW11Coefficients()

    # Generte Wht values
    Wht.x.fc = get_random_float(fc_low, fc_high, 1)
    Wht.x.z  = get_random_float(z_low, z_high, 1)
    Wht.y.fc = get_random_float(fc_low, fc_high, 1)
    Wht.y.z  = get_random_float(z_low, z_high, 1)
    Wht.z.fc = get_random_float(fc_low, fc_high, 1)
    Wht.z.z  = get_random_float(z_low, z_high, 1)

    # Generate Wrtzt values
    Wrtzt.x.fc = get_random_float(fc_low, fc_high, 1)
    Wrtzt.y.fc = get_random_float(fc_low, fc_high, 1)
    Wrtzt.z.fc = get_random_float(fc_low, fc_high, 1)

    # Generate W12 values
    W12.roll.fc  = get_random_float(fc_low, fc_high, 1)
    W12.roll.z   = get_random_float(z_low, z_high, 1)
    W12.pitch.fc = get_random_float(fc_low, fc_high, 1)
    W12.pitch.z  = get_random_float(z_low, z_high, 1)
    
    # Generate W11 values
    W11.roll.fc  = get_random_float(fc_low, fc_high, 1)
    W11.pitch.fc = get_random_float(fc_low, fc_high, 1)
    W11.yaw.fc   = get_random_float(fc_low, fc_high, 1)

    #return Wht, Wrtzt, W11, W12
    return Specimen( Wht, Wrtzt, W11, W12 )

# ===============================================================================
# @brief: List specimen coefficients (genes)
#
#      Coefficient list order:
#
#           [0] - Wht x-axis fc
#           [1] - Wht x-axis zeta
#           [2] - Wht y-axis fc
#           [3] - Wht y-axis zeta
#           [4] - Wht z-axis fc
#           [5] - Wht z-axis zeta
#
#           [6] - Wrtzt x-axis fc
#           [7] - Wrtzt y-axis fc
#           [8] - Wrtzt z-axis fc
#
#           [9]  - W12 roll fc
#           [10] - W12 roll zeta
#           [11] - W12 pitch fc
#           [12] - W12 pitch zeta    
# 
#           [13] - W11 roll fc
#           [14] - W11 pitch fc
#           [15] - W11 yaw fc                
#
# @param[in]:    specimen   - Speciment of a population
# @return:       coef       - List of coefficients
# ===============================================================================
def list_specimen_coefficient(specimen):

    coef = [specimen.Wht.x.fc, specimen.Wht.x.z,\
            specimen.Wht.y.fc, specimen.Wht.y.z,\
            specimen.Wht.z.fc, specimen.Wht.z.z,\
            specimen.Wrtzt.x.fc, specimen.Wrtzt.y.fc, specimen.Wrtzt.z.fc,\
            specimen.W12.roll.fc, specimen.W12.roll.z,\
            specimen.W12.pitch.fc, specimen.W12.pitch.z,\
            specimen.W11.roll.fc, specimen.W11.pitch.fc, specimen.W11.yaw.fc ] 

    return coef

# ===============================================================================
# @brief:   Print specimen coefficients. Raw option to print in such for
#           that copy/past to sript is easyier.
#
# @param[in]:    specimen   - Speciment of population 
# @param[in]:    raw        - Print option
# @return:       void
# ===============================================================================
def print_specimen_coefficient(specimen, raw=False):
    coef = list_specimen_coefficient(specimen)

    if False == raw:
        print("Wht   =[ x:[fc:%.3f, zeta:%.3f] y:[fc:%.3f, zeta:%.3f] z:[fc:%.3f, zeta:%.3f] ]" % ( coef[0], coef[1], coef[2], coef[3], coef[4], coef[5] ))
        print("Wrtzt =[ x:[fc:%.3f] y:[fc:%.3f] z:[fc:%.3f] ]" % ( coef[6], coef[7], coef[8] ))
        print("W12   =[ roll:[fc:%.3f, zeta:%.3f] pitch:[fc:%.3f, zeta:%.3f] ]" % ( coef[9], coef[10], coef[11], coef[12] ))
        print("W11   =[ roll:[fc:%.3f] pitch:[fc:%.3f] yaw:[fc:%.3f] ]\n" % ( coef[13], coef[14], coef[15] ))
    else:
        print("Wht   =[[%.6f,%.6f],[%.6f,%.6f],[%.6f,%.6f]]"    % ( coef[0], coef[1], coef[2], coef[3], coef[4], coef[5] ))
        print("Wrtzt =[%.6f,%.6f,%.6f]"                         % ( coef[6], coef[7], coef[8] ))
        print("W12   =[[ %.6f, %.6f ],[%.6f,%.6f]]"             % ( coef[9], coef[10], coef[11], coef[12] ))
        print("W11   =[%.6f,%.6f,%.6f]\n"                         % ( coef[13], coef[14], coef[15] ))

# ===============================================================================
# @brief:   Calculate fitness of system module based on sensation error
#
#
# @note:    Fitness of speciment is defined as inversly proportional
#           model RMS error value. RMS values of accelerations and 
#           angluar velocities are sum togheter.
#
#
# @param[in]:    specimen   - Speciment of population 
# @param[in]:    fs         - Sample frequency used in system model
# @param[in]:    stim       - Stimuli signal
# @param[in]:    stim_size  - Stimuli signal size in samples
# @param[in]:    route_opt  - Stimuli signal route to dimestion option
# @return:       fitness    - Fitness value of specimen
# ===============================================================================
def calculate_fitness(specimen, fs, stim, stim_size, route_opt):
    err_a_sum = [0] * 3
    err_a_rms = [0] * 3
    err_w_sum = [0] * 3
    err_w_rms = [0] * 3
    fitness = 0

    # Convert filter coefficient to suit SystemModel input
    WHT= [[ specimen.Wht.x.fc, specimen.Wht.x.z ],
          [ specimen.Wht.y.fc, specimen.Wht.y.z ],
          [ specimen.Wht.z.fc, specimen.Wht.z.z ]]
    
    WRTZT = [ specimen.Wrtzt.x.fc, specimen.Wrtzt.y.fc, specimen.Wrtzt.z.fc ]

    W12 = [[ specimen.W12.roll.fc, specimen.W12.roll.z ],
           [ specimen.W12.pitch.fc, specimen.W12.pitch.z ]]

    W11 = [ specimen.W11.roll.fc, specimen.W11.pitch.fc, specimen.W11.yaw.fc ]

    # Create syste model
    sys_model = SystemModel( Wht=WHT, Wrtzt=WRTZT, W11=W11, W12=W12, fs=fs)

    # Simulate stimulated system model
    for n in range(stim_size):

        # Route stimuli
        a_in, beta_in = system_model_route_input_signal( stim[n], route_opt )

        # Simulate model
        err_a, err_w = sys_model.update( a_in, beta_in )

        # Square & sum for RMS value
        for i in range(3):
            err_a_sum[i] += ( err_a[i]**2 )
            err_w_sum[i] += ( err_w[i]**2 )

    # Calculate RMS
    for i in range(3):
        err_a_rms[i] = np.sqrt( err_a_sum[i] / stim_size )
        err_w_rms[i] = np.sqrt( err_w_sum[i] / stim_size )

        fitness += ( err_a_rms[i] + err_w_rms[i] )

    # Higher system model error lower the fitness
    fitness = ( 1 / fitness )

    del(sys_model)

    return fitness


# ===============================================================================
# @brief:   Calcualte population fitness and overall rank
#
# @param[in]:    pop                - Population 
# @param[in]:    pop_size           - Size of population
# @param[in]:    fs                 - Sample frequency for system model
# @param[in]:    stim_signal        - Signal to stimuli system model
# @param[in]:    stim_size          - Size of stimuli signal
# @param[in]:    stim_route         - Stimuli route options
# @return:       _pop_fitness       - Population fintess, contains individual fitness
# @return:       _pop_fitness_sum   - Population fintess sum (rank)
# ===============================================================================
def calculate_population_fitness(pop, pop_size, fs, stim_signal, stim_size, stim_route):
    _pop_fitness = []
    _pop_fitness_sum = 0

    for n in range(pop_size):
        _pop_fitness.append( calculate_fitness(pop[n], fs, stim_signal, stim_size, stim_route ))
        _pop_fitness_sum += _pop_fitness[-1]

    return _pop_fitness, _pop_fitness_sum



# Based on turnamet selection
def select_parents(pop, pop_fitness):

    # Pick turnament candidates
    turnament_candidates_idx = random.sample(range(0, len(pop)), TURNAMENT_SIZE)

    #print(turnament_candidates_idx)

    # Collect candidate fitness
    candidate_fitness = []
    for n in range(TURNAMENT_SIZE):
        candidate_fitness.append( pop_fitness[ turnament_candidates_idx[n]] )

    # Find two highest value in candidates fitness
    best_candidate_fitness = max( candidate_fitness )
    candidate_fitness.remove(best_candidate_fitness)
    second_best_candidate_fitness = max( candidate_fitness )

    # Two highest candidates becomes parents
    for n in range(len(pop_fitness)):
        if pop_fitness[n] == best_candidate_fitness:
            p1 = pop[n]
        if pop_fitness[n] == second_best_candidate_fitness:
            p2 = pop[n]

    # NOTE: Parent 1 is always better than p2
    return p1, p2


def apply_crossover(gene_1, gene_2, crossover_rate):
    if 1 == np.random.choice([0, 1], p=[1.0 - crossover_rate, crossover_rate], size=1):
        return gene_2
    else:
        return gene_1



def make_love(p1, p2, crossover_rate):

    # Inherit from parent 1
    child = Specimen(Wht=p1.Wht, Wrtzt=p1.Wrtzt, W11=p1.W11, W12=p1.W12)

    """
    child.Wht.x.fc  = apply_crossover( p1.Wht.x.fc, p2.Wht.x.fc, crossover_rate )
    child.Wht.x.z   = apply_crossover( p1.Wht.x.z, p2.Wht.x.z, crossover_rate )
    child.Wht.y.fc  = apply_crossover( p1.Wht.y.fc, p2.Wht.y.fc, crossover_rate )
    child.Wht.y.z   = apply_crossover( p1.Wht.y.z, p2.Wht.y.z, crossover_rate )
    child.Wht.z.fc  = apply_crossover( p1.Wht.z.fc, p2.Wht.z.fc, crossover_rate ) 
    child.Wht.z.z   = apply_crossover( p1.Wht.z.z, p2.Wht.z.z, crossover_rate ) 
    """

    # NOTE: COUPLED GENES
    if 1 == np.random.choice([0, 1], p=[1.0 - crossover_rate, crossover_rate], size=1):
        child.Wht.x.fc = p2.Wht.x.fc
        child.Wht.x.z  = p2.Wht.x.z

    if 1 == np.random.choice([0, 1], p=[1.0 - crossover_rate, crossover_rate], size=1):
        child.Wht.y.fc = p2.Wht.y.fc
        child.Wht.y.z  = p2.Wht.y.z

    if 1 == np.random.choice([0, 1], p=[1.0 - crossover_rate, crossover_rate], size=1):
        child.Wht.z.fc = p2.Wht.z.fc
        child.Wht.z.z  = p2.Wht.z.z
    


    child.Wrtzt.x.fc = apply_crossover( p1.Wrtzt.x.fc, p2.Wrtzt.x.fc, crossover_rate )
    child.Wrtzt.y.fc = apply_crossover( p1.Wrtzt.y.fc, p2.Wrtzt.y.fc, crossover_rate )
    child.Wrtzt.z.fc = apply_crossover( p1.Wrtzt.z.fc, p2.Wrtzt.z.fc, crossover_rate )

    """
    child.W12.roll.fc  = apply_crossover( p1.W12.roll.fc, p2.W12.roll.fc, crossover_rate )
    child.W12.roll.z   = apply_crossover( p1.W12.roll.z, p2.W12.roll.z, crossover_rate )
    child.W12.pitch.fc = apply_crossover( p1.W12.pitch.fc, p2.W12.pitch.fc, crossover_rate )
    child.W12.pitch.z  = apply_crossover( p1.W12.pitch.z, p2.W12.pitch.z, crossover_rate )
    """

    # COUPLED GENES
    if 1 == np.random.choice([0, 1], p=[1.0 - crossover_rate, crossover_rate], size=1):
        child.W12.roll.fc = p2.W12.roll.fc
        child.W12.roll.z = p2.W12.roll.z

    if 1 == np.random.choice([0, 1], p=[1.0 - crossover_rate, crossover_rate], size=1):
        child.W12.pitch.fc = p2.W12.pitch.fc
        child.W12.pitch.z  = p2.W12.pitch.z


    child.W11.roll.fc  = apply_crossover( p1.W11.roll.fc, p2.W11.roll.fc, crossover_rate )
    child.W11.pitch.fc = apply_crossover( p1.W11.pitch.fc, p2.W11.pitch.fc, crossover_rate )
    child.W11.yaw.fc   = apply_crossover( p1.W11.yaw.fc, p2.W11.yaw.fc, crossover_rate )

    return child


def get_mutation_target(mutation_rate, size):
    return np.random.choice([0, 1], p=[1.0 - mutation_rate, mutation_rate], size=size)
    

def mutate_child_gene(gene, mutation_rate, low, high):

    # Is gene being mutated?
    if 1 == np.random.choice([0, 1], p=[1.0 - mutation_rate, mutation_rate], size=1):

        # 50/50 if positive or negative affect of mutation
        if 1 == np.random.choice([0, 1], p=[0.5, 0.5], size=1):
            mut_gene = (( 1.0 + MUTATION_IMPACT ) * gene )
        else:
            mut_gene = (( 1.0 - MUTATION_IMPACT ) * gene )

        # Limit mutation
        if mut_gene > high:
            mut_gene = high
        elif mut_gene < low:
            mut_gene = low

        return mut_gene 
    else:
        return gene

def mutate_child(child, mutation_rate):

    # Wht
    child.Wht.x.fc  = mutate_child_gene( child.Wht.x.fc, mutation_rate, WASHOUT_FILTER_FC_MIN_VALUE, WASHOUT_FILTER_FC_MAX_VALUE )
    child.Wht.x.z   = mutate_child_gene( child.Wht.x.z, mutation_rate, WASHOUT_FILTER_Z_MIN_VALUE, WASHOUT_FILTER_Z_MAX_VALUE )
    child.Wht.y.fc  = mutate_child_gene( child.Wht.y.fc, mutation_rate, WASHOUT_FILTER_FC_MIN_VALUE, WASHOUT_FILTER_FC_MAX_VALUE )
    child.Wht.y.z   = mutate_child_gene( child.Wht.y.z, mutation_rate, WASHOUT_FILTER_Z_MIN_VALUE, WASHOUT_FILTER_Z_MAX_VALUE )
    child.Wht.z.fc  = mutate_child_gene( child.Wht.z.fc, mutation_rate, WASHOUT_FILTER_FC_MIN_VALUE, WASHOUT_FILTER_FC_MAX_VALUE )
    child.Wht.z.z   = mutate_child_gene( child.Wht.z.z, mutation_rate, WASHOUT_FILTER_Z_MIN_VALUE, WASHOUT_FILTER_Z_MAX_VALUE )

    # Wrtz
    child.Wrtzt.x.fc = mutate_child_gene( child.Wrtzt.x.fc, mutation_rate, WASHOUT_FILTER_FC_MIN_VALUE, WASHOUT_FILTER_FC_MAX_VALUE )
    child.Wrtzt.y.fc = mutate_child_gene( child.Wrtzt.y.fc, mutation_rate, WASHOUT_FILTER_FC_MIN_VALUE, WASHOUT_FILTER_FC_MAX_VALUE )
    child.Wrtzt.z.fc = mutate_child_gene( child.Wrtzt.z.fc, mutation_rate, WASHOUT_FILTER_FC_MIN_VALUE, WASHOUT_FILTER_FC_MAX_VALUE )

    # W12
    child.W12.roll.fc   = mutate_child_gene( child.W12.roll.fc, mutation_rate, WASHOUT_FILTER_FC_MIN_VALUE, WASHOUT_FILTER_FC_MAX_VALUE )
    child.W12.roll.z    = mutate_child_gene( child.W12.roll.z, mutation_rate, WASHOUT_FILTER_Z_MIN_VALUE, WASHOUT_FILTER_Z_MAX_VALUE )
    child.W12.pitch.fc  = mutate_child_gene( child.W12.pitch.fc, mutation_rate, WASHOUT_FILTER_FC_MIN_VALUE, WASHOUT_FILTER_FC_MAX_VALUE )
    child.W12.pitch.z   = mutate_child_gene( child.W12.pitch.z, mutation_rate, WASHOUT_FILTER_Z_MIN_VALUE, WASHOUT_FILTER_Z_MAX_VALUE )

    # W11
    child.W11.roll.fc  = mutate_child_gene( child.W11.roll.fc, mutation_rate, WASHOUT_FILTER_FC_MIN_VALUE, WASHOUT_FILTER_FC_MAX_VALUE )
    child.W11.pitch.fc = mutate_child_gene( child.W11.pitch.fc, mutation_rate, WASHOUT_FILTER_FC_MIN_VALUE, WASHOUT_FILTER_FC_MAX_VALUE )
    child.W11.yaw.fc   = mutate_child_gene( child.W11.yaw.fc, mutation_rate, WASHOUT_FILTER_FC_MIN_VALUE, WASHOUT_FILTER_FC_MAX_VALUE )

    return child


def find_best_specimen_and_fitness(pop, pop_fitness):

    # Find best fintess
    _best_specimen_fitness = max(pop_fitness)
    
    # Find best specimen
    for idx, s in enumerate(pop):
        if _best_specimen_fitness == pop_fitness[idx]:
            _best_specimen = s
    
    return _best_specimen, _best_specimen_fitness



def select_elite(pop, pop_fitness, elite_num):
    _elite_pop = []
    pop_temp = []
    pop_fit_temp = []

    # Make a working copy
    pop_temp = copy.deepcopy( pop )
    pop_fit_temp = copy.deepcopy( pop_fitness )
    
    # Select elite numer of best specimen
    for n in range(elite_num):
        
        # Find best specimen
        _best_specimen, _best_specimen_fitness = find_best_specimen_and_fitness(pop_temp, pop_fit_temp)

        # Add to elite
        _elite_pop.append( _best_specimen )

        print("Elite fit: %.3f" %   _best_specimen_fitness)

        # Remove from search list
        pop_temp.remove( _best_specimen )
        pop_fit_temp.remove( _best_specimen_fitness )


    return _elite_pop


def make_new_generation(pop, pop_fitness, mutation_rate, crossover_rate, elite_num):
    _new_pop = [] 

    # Apply elitsm
    _elite_pop = select_elite(pop, pop_fitness, elite_num)

    for e in _elite_pop:
        _new_pop.append( e )

    # Generate new POPULATION SIZE number of childs
    for s in range( POPULATION_SIZE - elite_num ):

        # Select parents & remove then from next selection cycle
        #p1, p2 = select_parents(pop, pop_fitness)
        p1, p2 = select_parents( copy.deepcopy(pop), copy.deepcopy(pop_fitness) ) 
        
        # Make a child
        child = make_love(p1, p2, crossover_rate)

        # Mutate child
        child = mutate_child(child, mutation_rate)

        # Add child to new generation of population
        _new_pop.append(child)
        
    return _new_pop
    


def print_pop_fitness_ratio(pop_fitness, pop_fitness_sum):
    fitness_ratio = []
    for p_fit in pop_fitness:
        fitness_ratio.append( p_fit / ( pop_fitness_sum / POPULATION_SIZE ))
    print("f/f_avg: %s" % ["%.3f" % r for r in fitness_ratio])




# ===============================================================================
# @brief: Route signals to stimuli individual dimensions - MUX
#
# @param[in]:    inp_sig    - Input signal
# @param[in]:    sel        - Multiplexor selection
# @return:       a,w        - Acceleration & angular rates outputs
# ===============================================================================
def system_model_route_input_signal(inp_sig, sel):
    a = [0] * 3
    w = [0] * 3

    if sel == INPUT_SIGNAL_ROUTE_TO_AX:
        a[0] = inp_sig
    elif sel == INPUT_SIGNAL_ROUTE_TO_AY:
        a[1] = inp_sig
    elif sel == INPUT_SIGNAL_ROUTE_TO_AZ:
        a[2] = inp_sig
    elif sel == INPUT_SIGNAL_ROUTE_TO_WX:
        w[0] = inp_sig
    elif sel == INPUT_SIGNAL_ROUTE_TO_WY:
        w[1] = inp_sig
    elif sel == INPUT_SIGNAL_ROUTE_TO_WZ:
        w[2] = inp_sig
    else:
        raise AssertionError

    return a, w



# ===============================================================================
# @brief: Generate stimuli signal, based on FunctionGenerator settings...
#
# @return: Stimuli signal
# ===============================================================================
def generate_stimuli_signal():

    # Time array
    _time, _dt = np.linspace( 0.0, TIME_WINDOW, num=SAMPLE_NUM, retstep=True )

    # Filter input/output
    _x = [ 0 ] * SAMPLE_NUM
    _x_d = [0]

    # Generate inputs
    _fg = FunctionGenerator( INPUT_SIGNAL_FREQ, INPUT_SIGNAL_AMPLITUDE, INPUT_SIGNAL_OFFSET, INPUT_SIGNAL_PHASE, INPUT_SIGNAL_SELECTION )
    
    # Down sample
    _downsamp_cnt = 0
    _downsamp_samp = [0]
    _d_time = [0]
    
    # Generate stimuli signals
    for n in range(SAMPLE_NUM):
        #_x[n] = ( _fg.generate( _time[n] ))

        
        # Some custom signal
        CUSTOM_SIG_MAX = 1

        DELAY_TIME = 0.1
        RISE_TIME = 1
        FALL_TIME = 1
        DURATION_OF_MAX = 0.5

        if _time[n] < DELAY_TIME:
            _x[n] = 0.0
        
        elif _time[n] < ( RISE_TIME + DELAY_TIME ):
            _x[n] = _x[n-1] + CUSTOM_SIG_MAX / IDEAL_SAMPLE_FREQ
        
        elif _time[n] < ( DURATION_OF_MAX + RISE_TIME + DELAY_TIME ):
            _x[n] = CUSTOM_SIG_MAX
        
        elif _time[n] < ( DURATION_OF_MAX + RISE_TIME + FALL_TIME + DELAY_TIME ):
            _x[n] = _x[n-1] - CUSTOM_SIG_MAX / IDEAL_SAMPLE_FREQ
        
        elif _time[n] < TIME_WINDOW:
            _x[n] = 0
        
        else:
            _x[n] = 0
        

    return _x, len(_x)


def generate_specimen(Wht_in, Wrtzt_in, W12_in, W11_in):

    # Create washout coefficients
    Wht     = WashoutWhtCoefficients()
    Wrtzt   = WashoutWrtztCoefficients()
    W12     = WashoutW12Coefficients()
    W11     = WashoutW11Coefficients()

    # Generte Wht values
    Wht.x.fc = Wht_in[0][0]
    Wht.x.z  = Wht_in[0][1]
    Wht.y.fc = Wht_in[1][0]
    Wht.y.z  = Wht_in[1][1]
    Wht.z.fc = Wht_in[2][0]
    Wht.z.z  = Wht_in[2][1]

    # Generate Wrtzt values
    Wrtzt.x.fc = Wrtzt_in[0]
    Wrtzt.y.fc = Wrtzt_in[1]
    Wrtzt.z.fc = Wrtzt_in[2]

    # Generate W12 values
    W12.roll.fc  = W12_in[0][0]
    W12.roll.z   = W12_in[0][1]
    W12.pitch.fc = W12_in[1][0]
    W12.pitch.z  = W12_in[1][1]
    
    # Generate W11 values
    W11.roll.fc  = W11_in[0]
    W11.pitch.fc = W11_in[1]
    W11.yaw.fc   = W11_in[2]

    return Specimen( Wht, Wrtzt, W11, W12 )




# ===============================================================================
#       MAIN ENTRY
# ===============================================================================

if __name__ == "__main__":

    
    # Clear console
    os.system("cls")

    # Generation & evolution timer
    gen_timer = StopWatch()
    evo_timer = StopWatch()

    # Generate stimuli signal
    stim_signal, stim_size = generate_stimuli_signal()

    # Population
    pop = []


    # ===============================================================================
    #   START GENERATION OF POPULATION ZERO WITH GOOD SPECIMENS
    # ===============================================================================
    POPULATION_ZERO_INJECTION_NUM = 2

    # Initial good example
    Wht   =[[0.016496,0.042655],[0.013504,0.063655],[0.019051,0.063655]]
    Wrtzt =[0.005404,0.008660,0.135590]
    W12   =[[ 0.027772, 0.198342 ],[0.030232,0.311606]]
    W11   =[0.014108,0.040818,0.023847]
    
    # Add speciment to popolation
    pop.append( generate_specimen(Wht, Wrtzt, W12, W11 ))

    # Initial good example
    Wht   =[[0.015000,0.037500],[0.015000,0.084375],[0.010824,0.037500]]
    Wrtzt =[0.010000,0.007500,0.324367]
    W12   =[[ 0.032473, 0.189845 ],[0.113906,0.427149]]
    W11   =[0.022500,0.008438,0.024027]

    # Add speciment to population
    pop.append( generate_specimen(Wht, Wrtzt, W12, W11 ))

    # ===============================================================================
    #   RANDOM GENERATION OF POPULATION ZERO
    # ===============================================================================
    for n in range(POPULATION_SIZE-POPULATION_ZERO_INJECTION_NUM):
        pop.append( generate_specimen_random_gene(WASHOUT_FILTER_FC_MIN_VALUE, WASHOUT_FILTER_FC_MAX_VALUE, WASHOUT_FILTER_Z_MIN_VALUE, WASHOUT_FILTER_Z_MAX_VALUE) )
        
    # Variables for statistics
    best_specimen = Specimen(0,0,0,0)
    best_speciment_fit = 0
    first_fit = 0
    pop_fitness_sum_prev = 0
    best_specimen_fitness_prev = 0

    # Start timer
    evo_timer.start()
    gen_timer.start()

    pop_fitness = []

    # Loop thru generations
    for g in range(GENERATION_SIZE):

        print("===============================================================================================")
        print("     GENERATION #%s" % g);
        print("===============================================================================================")

        # ===============================================================================
        #   1. CALCULATE POPULATION FITNESS
        # ===============================================================================
        pop_fitness, pop_fitness_sum = calculate_population_fitness( pop, POPULATION_SIZE, SAMPLE_FREQ, stim_signal, stim_size, INPUT_SIGNAL_ROUTE )

        # Find best speciment and its fitness
        best_specimen, best_specimen_fitness = find_best_specimen_and_fitness(copy.deepcopy(pop), pop_fitness)

        print("Pop fitness: %s" % ["%.3f" % p for p in pop_fitness])

        # ===============================================================================
        #   2. REPRODUCTION
        # ===============================================================================

        # Make a new (BETTER) generation
        pop = make_new_generation(pop, pop_fitness, MUTATION_PROPABILITY, CROSSOVER_PROPABILITY, ELITISM_NUM)



        # ===============================================================================
        #   Intermediate reports of evolution
        # ===============================================================================

        # Restart timer
        exe_time = gen_timer.restart()


        # Store first population fitness sum
        if g == 0:
            first_best_specimen = best_specimen 
            first_best_specimen_fitness = best_specimen_fitness
            first_pop_fitness_sum = pop_fitness_sum

        # Report progress
        else:
            print("Overall evolution progress: %.2f %%" % (( pop_fitness_sum - first_pop_fitness_sum ) / 100.0 ))
            print("Generation progress: %.2f %%\n" % (( pop_fitness_sum - pop_fitness_sum_prev ) / 100.0 ))
        pop_fitness_sum_prev = pop_fitness_sum

        
        print("Best specimen progress: %.2f" % ( best_specimen_fitness - best_specimen_fitness_prev ))
        print("Best specimen progress: %.3f %%" % (( best_specimen_fitness - best_specimen_fitness_prev ) / 100.0 ))
        best_specimen_fitness_prev = best_specimen_fitness
        print("f_best/f_avg: %.3f" % ( best_specimen_fitness / ( pop_fitness_sum / POPULATION_SIZE )))

        print_specimen_coefficient( best_specimen, raw=True )

        print("")
        print_pop_fitness_ratio(pop_fitness, pop_fitness_sum)
        print("Fitness: %s" % ["%.3f" % p for p in pop_fitness])

        print("Execution time: %.0f sec" % exe_time )
        print("Evolution duration: %.2f min\n" % ( evo_timer.time()/60.0 ))
       
    # Stop evolution timer
    evo_duration = evo_timer.stop()



    # End report
    print("===============================================================================================")
    print("     EVOLUTION FINISHED")
    print("===============================================================================================")
    #print("Best speciment fit: \n %s" % print_specimen_coefficient( best_speciment_fit ))
    print("First population fit: %.2f" % first_pop_fitness_sum)
    print("End population fit: %.2f" % pop_fitness_sum)
    print("Overall evolution progress: %.2f %%" % (( pop_fitness_sum - first_pop_fitness_sum ) / 100.0 ))
    
    print("First best specimen fit: %.2f" % first_best_specimen_fitness)
    print("End population best specimen fit: %.2f" % best_specimen_fitness)
    print("Best specimen progress: %.3f %%" % (( best_specimen_fitness - first_best_specimen_fitness ) / 100.0 ))

    print("End score: %.2f\n" % ( pop_fitness_sum / POPULATION_SIZE ))
    print("Evolution total duration: %.2f sec\n" % evo_duration )    

    
    print("First population best specimen coefficients: ")
    print_specimen_coefficient( first_best_specimen, raw=True )

    print("End population best specimen coefficients: ")
    print_specimen_coefficient( best_specimen, raw=True )

    print("\nGA SETTINGS:")
    print("Number of generations: %s" % GENERATION_SIZE)
    print("Number of populations: %s" % POPULATION_SIZE)
    print("Mutation propability: %s" % MUTATION_PROPABILITY)
    print("Mutation impact: %s" % MUTATION_IMPACT)
    print("Elithism number: %s" % ELITISM_NUM)
    print("Crossover propability: %s" % CROSSOVER_PROPABILITY)

    print("\nSTIMULI SETTINGS:")
    print("Time window: %.1f sec" % TIME_WINDOW)
    print("Route to: %s" % INPUT_SIGNAL_ROUTE)
    

# ===============================================================================
#       END OF FILE
# ===============================================================================
