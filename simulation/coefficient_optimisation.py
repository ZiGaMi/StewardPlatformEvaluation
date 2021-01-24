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
TIME_WINDOW = 2

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

    return Wht, Wrtzt, W11, W12

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
        print("W11   =[ roll:[fc:%.3f] pitch:[fc:%.3f] yaw:[fc:%.3f] ]" % ( coef[13], coef[14], coef[15] ))
    else:
        print("Wht   =[[%.6f,%.6f],[%.6f,%.6f],[%.6f,%.6f]]"    % ( coef[0], coef[1], coef[2], coef[3], coef[4], coef[5] ))
        print("Wrtzt =[%.6f,%.6f,%.6f]"                         % ( coef[6], coef[7], coef[8] ))
        print("W12   =[[ %.6f, %.6f ],[%.6f,%.6f]]"             % ( coef[9], coef[10], coef[11], coef[12] ))
        print("W11   =[%.6f,%.6f,%.6f]"                         % ( coef[13], coef[14], coef[15] ))

             






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


def make_new_childs(p1, p2, mutation_rate, low, high):

    # Crossover
    #c1 = Specimen( p2.Wht, p1.Wrtzt, p2.W11, p1.W12 )
    #c2 = Specimen( p1.Wht, p2.Wrtzt, p1.W11, p2.W12 )
    c1 = Specimen( p1.Wht, p1.Wrtzt, p1.W11, p1.W12 )
    c2 = Specimen( p2.Wht, p2.Wrtzt, p2.W11, p2.W12 )

    # Cross over translation channel X fc & titl coordination channel damping factor
    Wht_fc_x = c1.Wht[0][0]
    c1.Wht[0][0] = c2.Wht[0][0] 
    c2.Wht[0][0] = Wht_fc_x

    W12_z_roll = c1.W12[0][1]
    c1.W12[0][1] = c2.W12[0][1]
    c2.W12[0][1] = W12_z_roll



    # Mutate
    mutation_target = get_mutation_target( mutation_rate, 4 )
    mut_Wht, mut_Wrtzt, mut_W11, mut_W12 = generate_specimen_gene(low, high)

    for n in range(4):
        if 1 == mutation_target[n]:
            if 0 == n:  
                c1.Wht[0][0] = mut_Wht[0][0]
            elif 1 == n:
                c1.Wht[0][1] = mut_Wht[0][1]
            elif 2 == n:
                c1.W12[0][0] = mut_W12[0][0]
            elif 3 == n:
                c1.W12[0][1] = mut_W12[0][1]

    # Mutate
    mutation_target = get_mutation_target( mutation_rate, 4 )
    mut_Wht, mut_Wrtzt, mut_W11, mut_W22 = generate_specimen_gene(low, high)

    for n in range(4):
        if 1 == mutation_target[n]:
            if 0 == n:  
                c2.Wht[0][0] = mut_Wht[0][0]
            elif 1 == n:
                c2.Wht[0][1] = mut_Wht[0][1]
            elif 2 == n:
                c2.W12[0][0] = mut_W12[0][0]
            elif 3 == n:
                c2.W12[0][1] = mut_W12[0][1]

    return c1, c2
    



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
            err_a_sum[i] += ( err_a[i]**2 )
            err_w_sum[i] += ( err_w[i]**2 )

    # Calculate RMS
    for i in range(3):
        err_a_rms[i] = np.sqrt( err_a_sum[i] / stim_size )
        err_w_rms[i] = np.sqrt( err_w_sum[i] / stim_size )

        fitness += ( err_a_rms[i] + err_w_rms[i] )

    # Higher system model error lower the fitness
    fitness = 1 / fitness

    return fitness







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


def calculate_pop_fit_and_error(pop_fit, size):
    _pop_fit_nor = [0] * size
    _fit_sum = 0

    # Sum fit of whole population fitness
    for n in range(size):
        _fit_sum += pop_fit[n]

    # Normalise fitness
    # NOTE: Higher fitness means lower error of system model
    for n in range(size):
        _pop_fit_nor[n] = pop_fit[n] / _fit_sum

    return _pop_fit_nor, _fit_sum
















# ==================================================
#   WASHOUTE FILTER COEFFICIENTS LIMITS
# ==================================================
WASHOUT_FILTER_FC_MIN_VALUE = 0.01
WASHOUT_FILTER_FC_MAX_VALUE = 10.0
WASHOUT_FILTER_Z_MIN_VALUE  = 0.1
WASHOUT_FILTER_Z_MAX_VALUE  = 2.0


# ==================================================
#   GENETIC ALGORITHM SETTINGS
# ==================================================

# Population size
# NOTE: Prefered to be even
POPULATION_SIZE = 10

# Number of generations
GENERATION_SIZE = 10

# Mutation rate
MUTATION_RATE = 0.05


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


    # ===============================================================================
    #   RANDOM GENERATION OF POPULATION ZERO
    # ===============================================================================
    pop = []
    for n in range(POPULATION_SIZE):
        #Wht, Wrtzt, W11, W22 = generate_specimen_gene(COEFFICIENT_MIN_VALUE, COEFFICIENT_MAX_VALUE)
        Wht, Wrtzt, W11, W22 = generate_specimen_random_gene(WASHOUT_FILTER_FC_MIN_VALUE, WASHOUT_FILTER_FC_MAX_VALUE,\
                                                             WASHOUT_FILTER_Z_MIN_VALUE, WASHOUT_FILTER_Z_MAX_VALUE)
        pop.append( Specimen( Wht=Wht, Wrtzt=Wrtzt, W11=W11, W12=W22 ))

        
        print_specimen_coefficient( pop[n] )
        print("")



    """
    # Variables for statistics
    best_specimen = Specimen(0,0,0,0)
    best_speciment_fit = 0
    first_fit = 0
    overall_pop_fit_prev = 0

    # Start timer
    evo_timer.start()
    gen_timer.start()








    for g in range(GENERATION_SIZE):

        print("===============================================================================================")
        print("     GENERATION #%s" % g);
        print("===============================================================================================")

        # ===============================================================================
        #   1. CALCULATE POPULATION FITNESS
        # ===============================================================================
        pop_fitness = []
        for n in range(POPULATION_SIZE):
            pop_fitness.append( calculate_fitness(pop[n], SAMPLE_FREQ, stim_signal, stim_size, INPUT_SIGNAL_ROUTE ))

        # Normalise population fitness
        pop_fit_nor, overall_pop_fit = calculate_pop_fit_and_error( pop_fitness, POPULATION_SIZE )
        #print("Population fitness distribution (percent): %s \nOverall population fitness: %.2f " % ( pop_fit_nor, overall_pop_fit ))

        if g == 0:
            first_fit = overall_pop_fit



    
        # ===============================================================================
        #   2. SELECTION & REPRODUCTION
        # ===============================================================================
        elitsm_s_1 = Specimen(0,0,0,0)
        elitsm_s_2 = Specimen(0,0,0,0)

        max = 0
        max_idx = 0
        max_idx_prev = 0
        temp_pop = pop_fit_nor
        for idx, fit in enumerate(temp_pop):
            if fit > max:
                max = fit
                max_idx_prev = max_idx
                max_idx = idx
                


        elitsm_s_1 = pop[max_idx]
        elitsm_s_2 = pop[max_idx_prev]


        for s in range(int(POPULATION_SIZE/2)):

            # Select parents
            p1_idx, p2_idx = select_two_parants( pop_fit_nor, POPULATION_SIZE )

            # Make love
            pop[p1_idx], pop[p2_idx] = make_new_childs( pop[p1_idx], pop[p2_idx], MUTATION_RATE, COEFFICIENT_MIN_VALUE, COEFFICIENT_MAX_VALUE )

        # KUJNEKAJ!!!!!!!!!!
        pop[0] = elitsm_s_1
        #pop[1] = elitsm_s_2

        




        if pop_fitness[max_idx] >  best_speciment_fit:
            best_speciment_fit = pop_fitness[max_idx]
            best_specimen = elitsm_s_1

        # ===============================================================================
        #   Intermediate reports of evolution
        # ===============================================================================

        # Restart timer
        exe_time = gen_timer.restart()

        # Report
        print("Generation progress (in fits): %.2f\n" % ( overall_pop_fit - overall_pop_fit_prev ))
        print("Best specimen:\n -Wht = %s \n -Wrtzt= %s \n -W11 = %s\n -W12 = %s\n" % ( elitsm_s_1.Wht, elitsm_s_1.Wrtzt, elitsm_s_1.W11, elitsm_s_1.W12 ))
        print("Execution time: %.0f ms" % (exe_time * 1e3 ))
        print("Evolution duration: %.2f sec\n" % evo_timer.time() )

        # Store previous overall population fit 
        overall_pop_fit_prev = overall_pop_fit


    # Stop evolution timer
    evo_duration = evo_timer.stop()

    # End report
    print("===============================================================================================")
    print("     EVOLUTION FINISHED");
    print("===============================================================================================")
    print("Best speciment fit: %.2f" % best_speciment_fit)
    print("First population fit: %.2f" % first_fit)
    print("End population fit: %.2f\n" % overall_pop_fit)
    print("Best coefficients:\n -Wht = %s \n -Wrtzt= %s \n -W11 = %s\n -W12 = %s\n" % ( best_specimen.Wht, best_specimen.Wrtzt, best_specimen.W11, best_specimen.W12 ))
    print("Evolution total duration: %.2f sec\n" % evo_duration )    
    """

# ===============================================================================
#       END OF FILE
# ===============================================================================
