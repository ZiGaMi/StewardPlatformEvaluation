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
import matplotlib.pyplot as plt
import numpy as np

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
SAMPLE_FREQ = 100.0

# Ideal sample frequency
#   As a reference to sample rate constrained embedded system
#
# Unit: Hz
IDEAL_SAMPLE_FREQ = 1000.0

## Time window
#
# Unit: second
TIME_WINDOW = 5

## Input signal shape
INPUT_SIGNAL_FREQ = 0.1
INPUT_SIGNAL_AMPLITUDE = 9.81/4
INPUT_SIGNAL_OFFSET = INPUT_SIGNAL_AMPLITUDE
INPUT_SIGNAL_PHASE = -0.25

## Mux input signal
INPUT_SIGNAL_SELECTION = FunctionGenerator.FG_KIND_RECT

## Number of samples in time window
SAMPLE_NUM = int(( IDEAL_SAMPLE_FREQ * TIME_WINDOW ) + 1.0 )


# =====================================================
## TRANSLATION CHANNEL SETTINGS

# HPF Wht 2nd order filter
WASHOUT_HPF_WHT_FC_X  = 1.0
WASHOUT_HPF_WHT_Z_X   = .7071
WASHOUT_HPF_WHT_FC_Y  = 1.0
WASHOUT_HPF_WHT_Z_Y   = .7071
WASHOUT_HPF_WHT_FC_Z  = 1.0
WASHOUT_HPF_WHT_Z_Z   = .7071

WASHOUT_HPF_WHT_COEFFICIENT = [[ WASHOUT_HPF_WHT_FC_X, WASHOUT_HPF_WHT_Z_X ],
                               [ WASHOUT_HPF_WHT_FC_Y, WASHOUT_HPF_WHT_Z_Y ],
                               [ WASHOUT_HPF_WHT_FC_Z, WASHOUT_HPF_WHT_Z_Z ]]


# HPF Wrtzt 1st order filter
WASHOUT_HPF_WRTZT_FC_X  = 1.0
WASHOUT_HPF_WRTZT_FC_Y  = 1.0
WASHOUT_HPF_WRTZT_FC_Z  = 1.0

WASHOUT_HPF_WRTZT_COEFFICIENT = [ WASHOUT_HPF_WRTZT_FC_X, WASHOUT_HPF_WRTZT_FC_Y, WASHOUT_HPF_WRTZT_FC_Z ]

# =====================================================
## COORDINATION CHANNEL SETTINGS

# LPF W12 2nd order filter
WASHOUT_LPF_W12_FC_ROLL     = 1.0
WASHOUT_LPF_W12_Z_ROLL      = 1.0
WASHOUT_LPF_W12_FC_PITCH    = 1.0
WASHOUT_LPF_W12_Z_PITCH     = 1.0

WASHOUT_LPF_W12_COEFFICIENT = [[ WASHOUT_LPF_W12_FC_ROLL, WASHOUT_LPF_W12_Z_ROLL ],
                               [ WASHOUT_LPF_W12_FC_PITCH, WASHOUT_LPF_W12_Z_PITCH ]]

# =====================================================
## ROTATION CHANNEL SETTINGS

# HPF W11 1st order filter
WASHOUT_HPF_W11_FC_ROLL     = 1.0
WASHOUT_HPF_W11_FC_PITCH    = 1.0
WASHOUT_HPF_W11_FC_YAW      = 1.0

WASHOUT_HPF_W11_COEFFICIENT = [ WASHOUT_HPF_W11_FC_ROLL, WASHOUT_HPF_W11_FC_PITCH, WASHOUT_HPF_W11_FC_YAW ]


## ****** END OF USER CONFIGURATIONS ******

# ===============================================================================
#       FUNCTIONS
# ===============================================================================


def get_random_int(low, high, size):
    return np.random.randint(low, high, size)


def get_random_float(low, high, size):
    if 1 == size:
        return np.random.uniform(low, high, size)[0]
    else:
        return np.random.uniform(low, high, size)

def get_mutation_target(mutation_rate, size):
    return np.random.choice([0, 1], p=[1.0 - mutation_rate, mutation_rate], size=size)


# Higher fitness parent has higher propability of selection
# One parent thus can be selected multiple times
# pop_fitness in range from 0-1
def select_two_parants(pop_fitness, size):
    pop_idx = [ n for n in range(size) ]

    # Select parent 1
    p1_idx = int( np.random.choice(pop_idx, p=pop_fitness, size=(1, 1)) )

    # Select parent 2
    while True:
        p2_idx = int( np.random.choice(pop_idx, p=pop_fitness, size=(1, 1)) )

        if p1_idx != p2_idx:
            break
    
    return p1_idx, p2_idx


def make_new_childs(p1, p2, num_of_coef, mutation_rate, low, high):

    c1 = [0] * num_of_coef
    c2 = [0] * num_of_coef

    # Mutate
    mutation_target = get_mutation_target( mutation_rate, num_of_coef )

    # Mix gene
    for n in range(num_of_coef):
        
        # Crossover only odd gene   
        if 1 == ( n % 2 ):
            c = p1[n]
            p1[n] = p2[n]
            p2[n] = c

        c1[n] = p1[n]
        c2[n] = p2[n]

        # Mutate
        mutation_target = get_mutation_target( mutation_rate, num_of_coef )

        for n in range(num_of_coef):
            if 1 == mutation_target[n]:
                c1[n] = get_random_float(low, high, 1)
                c2[n] = get_random_float(low, high, 1)

    return c1, c2
    
# Specimen as one of many in population
class Specimen:

    def __init__(self, Wht, Wrtzt, W11, W12):
        self.Wht    = Wht
        self.Wrtzt  = Wrtzt
        self.W11    = W11
        self.W12    = W12


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


# Fitness of speciment is defined as inversly proportional
# model RMS error value. RMS values of accelerations and 
# angluar velocities are sum togheter at that point 
def calculate_fitness(specimen, fs, stim, stim_size, route_opt):
    err_a_sum = [0] * 3
    err_a_rms = [0] * 3
    err_w_sum = [0] * 3
    err_w_rms = [0] * 3
    fitness = 0

    # Create system model
    sys_model = SystemModel( Wht=specimen.Wht, Wrtzt=specimen.Wrtzt, W11=specimen.W11, W12=specimen.W12, fs=fs)

    # Simulate stimulated system model
    for n in range(stim_size):

        # Route stimuli
        a_in, beta_in = system_model_route_input_signal( stim[n], route_opt )

        # Simulate model
        err_a, err_w = sys_model.update( a_in, beta_in )

        # Square & sum for RMS value
        for i in range(3):
            err_a_sum[i] += err_a[i]**2
            err_w_sum[i] += err_w[i]**2

    # Calculate RMS
    for i in range(3):
        err_a_rms[i] = np.sqrt( err_a_sum[i] / stim_size )
        err_w_rms[i] = np.sqrt( err_w_sum[i] / stim_size )

        fitness += ( err_a_rms[i] + err_w_rms[i] )

    # Higher system model error lower the fitness
    fitness = 1 / fitness

    return fitness


def generate_specimen_gene(low, high):

    Wht = [ [ get_random_float(low, high, 1), get_random_float(low, high, 1) ],
            [ get_random_float(low, high, 1), get_random_float(low, high, 1) ],
            [ get_random_float(low, high, 1), get_random_float(low, high, 1) ]]

    Wrtzt = [ get_random_float(low, high, 1), get_random_float(low, high, 1), get_random_float(low, high, 1) ]

    W11 = [ get_random_float(low, high, 1), get_random_float(low, high, 1), get_random_float(low, high, 1) ]

    W12 = [ [ get_random_float(low, high, 1), get_random_float(low, high, 1) ],
            [ get_random_float(low, high, 1), get_random_float(low, high, 1) ]]

    return Wht, Wrtzt, W11, W12



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
        if _time[n] < 1.0:
            _x[n] = 0.0
        elif _time[n] < 2.0:
            _x[n] = _x[n-1] + 0.5 / IDEAL_SAMPLE_FREQ
        elif _time[n] < 3.0:
            _x[n] = 0.5
        elif _time[n] < 4.0:
            _x[n] = _x[n-1] - 0.5 / IDEAL_SAMPLE_FREQ
        elif _time[n] < 10.0:
            _x[n] = 0
        else:
            _x[n] = 0
        

    return _x, len(_x)


def calculate_pop_fit_and_error(pop_fit, size):
    pop_fit_nor = [0] * size
    fit_sum = 0

    # Sum fit of whole population fitness
    for n in range(size):
        fit_sum += pop_fit[n]

    # Normalise fitness
    # NOTE: Higher fitness means lower error of system model
    for n in range(size):
        pop_fit_nor[n] = pop_fit[n] / fit_sum

    return pop_fit_nor, fit_sum




# ===============================================================================
#       CLASSES
# ===============================================================================    




POPULATION_SIZE = 5
GENERATION_SIZE = 10

COEFFICIENT_MIN_VALUE = 0.0
COEFFICIENT_MAX_VALUE = 2.0


## Input signal route
INPUT_SIGNAL_ROUTE_TO_AX = 0
INPUT_SIGNAL_ROUTE_TO_AY = 1
INPUT_SIGNAL_ROUTE_TO_AZ = 2
INPUT_SIGNAL_ROUTE_TO_ROLL = 3
INPUT_SIGNAL_ROUTE_TO_PITCH = 4
INPUT_SIGNAL_ROUTE_TO_YAW = 5

INPUT_SIGNAL_ROUTE = INPUT_SIGNAL_ROUTE_TO_AX



# ===============================================================================
#       MAIN ENTRY
# ===============================================================================
if __name__ == "__main__":


    # Generate stimuli signal
    stim_signal, stim_size = generate_stimuli_signal()


    # ===============================================================================
    #   RANDOM GENERATION OF POPULATION ZERO
    # ===============================================================================
    pop = []
    for n in range(POPULATION_SIZE):
        Wht, Wrtzt, W11, W22 = generate_specimen_gene(COEFFICIENT_MIN_VALUE, COEFFICIENT_MAX_VALUE)
        pop.append( Specimen( Wht=Wht, Wrtzt=Wrtzt, W11=W11, W12=W22 ))


    # ===============================================================================
    #   CALCULATE POPULATION FITNESS
    # ===============================================================================
    pop_fitness = []
    for n in range(POPULATION_SIZE):
        print("Calculation of fitness... Specimen #%s" % n)
        pop_fitness.append( calculate_fitness(pop[n], SAMPLE_FREQ, stim_signal, stim_size, INPUT_SIGNAL_ROUTE ))

    # Normalise population fitness
    specimen_fitness, pop_fitness = calculate_pop_fit_and_error( pop_fitness, POPULATION_SIZE )
    print("(Generation #1) Specimen fitness (percent): %s | Overall pop fitness: %.2f " % ( specimen_fitness, pop_fitness ))






# ===============================================================================
#       END OF FILE
# ===============================================================================
