# ===============================================================================
# @file:    system_mode.py
# @note:    This script is TOP model of washout filter evaluation
# @author:  Ziga Miklosic
# @date:    16.01.2021
# @brief:   Top model of steward washout filter desing evaluation 
# ===============================================================================

# ===============================================================================
#       IMPORTS  
# ===============================================================================
import sys
import matplotlib.pyplot as plt
import numpy as np

from filters.filter_utils import FunctionGenerator
from human_perception import VestibularSystem
from washout_filter import WashoutFilter

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
IDEAL_SAMPLE_FREQ = 10000.0

## Time window
#
# Unit: second
TIME_WINDOW = 10

## Input signal shape
INPUT_SIGNAL_FREQ = 0.1
INPUT_SIGNAL_AMPLITUDE = 9.81/4
INPUT_SIGNAL_OFFSET = INPUT_SIGNAL_AMPLITUDE
INPUT_SIGNAL_PHASE = -0.25

## Mux input signal
INPUT_SIGNAL_SELECTION = FunctionGenerator.FG_KIND_RECT

## Input signal route
INPUT_SIGNAL_ROUTE_TO_AX = 0
INPUT_SIGNAL_ROUTE_TO_AY = 1
INPUT_SIGNAL_ROUTE_TO_AZ = 2
INPUT_SIGNAL_ROUTE_TO_WX = 3
INPUT_SIGNAL_ROUTE_TO_WY = 4
INPUT_SIGNAL_ROUTE_TO_WZ = 5

INPUT_SIGNAL_ROUTE = INPUT_SIGNAL_ROUTE_TO_AY

# =====================================================
# WASHOUT FILTER COEFFICINETS
# =====================================================

# =====================================================
## TRANSLATION CHANNEL SETTINGS

# HPF Wht 2nd order filter
WASHOUT_HPF_WHT_FC  = .01
WASHOUT_HPF_WHT_Z   = .05

# HPF Wrtzt 1st order filter
WASHOUT_HPF_WRTZT_FC  = .1

# =====================================================
## COORDINATION CHANNEL SETTINGS

# LPF W12 2nd order filter
WASHOUT_LPF_W12_FC  = .1
WASHOUT_LPF_W12_Z   = .2

# =====================================================
## ROTATION CHANNEL SETTINGS

# HPF W11 1st order filter
WASHOUT_HPF_W11_FC  = .1


## ****** END OF USER CONFIGURATIONS ******

## Number of samples in time window
SAMPLE_NUM = int(( IDEAL_SAMPLE_FREQ * TIME_WINDOW ) + 1.0 )


# ===============================================================================
#       FUNCTIONS
# ===============================================================================

def system_model_calc_sens_error(a_ref, w_ref, a_act, w_act):
    
    a_err = [0] * 3
    w_err = [0] * 3
    
    for n in range(3):
        a_err[n] = a_ref[n] - a_act[n]
        w_err[n] = w_ref[n] - w_act[n]

    return a_err, w_err


def system_model_log_data(storage, data):
    for n in range(3):
        storage[n].append( data[n] )


def system_model_plot_a_signals(ax, x, a):
    ax.plot( x, a[0], "r", label="ax")
    ax.plot( x, a[1], "y", label="ay")
    ax.plot( x, a[2], "g", label="az")


def system_model_plot_w_signals(ax, x, w):
    ax.plot( x, w[0], "w", label="wx")
    ax.plot( x, w[1], "tab:orange", label="wy")
    ax.plot( x, w[2], "c", label="wz")


def system_model_plot_signals(ax, x, a, w):
    system_model_plot_a_signals(ax, x, a)
    system_model_plot_w_signals(ax, x, w)

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
#       CLASSES
# ===============================================================================    

## Converstion to driver reference frame
class DriverFrame:

    # ===============================================================================
    # @brief: Initialization of conversion to driver frame
    #
    # @param[in]:    r_ae   - Position of driver head in respect to platform
    # @param[in]:    dt     - Period time 
    # @return:       void
    # ===============================================================================
    def __init__(self, r_ea, dt):
        self.r_ea = r_ea
        self.dt = dt

        # Previous values of rotations & angular rates 
        # Used for derivitive calculations
        self.beta_prev = [0] * 3
        self.w_prev = [0] * 3


    # ===============================================================================
    # @brief: Transform to driver frame
    #
    # @param[in]:    a          - Vector of accelerations
    # @param[in]:    beta       - Vector of rotations
    # @return:       a_d, w_d   - Transformed accelerations and angular velocities
    #                             to driver reference frame  
    # ===============================================================================
    def transform(self, a, beta):
        a_d = [0] * 3
        w_d = [0] * 3
        beta_dot = [0] * 3
        
        # Calculate rotation derivitive
        beta_dot, self.beta_prev = self.__calculate_derivitive_on_vector( beta, self.beta_prev, self.dt )

        # Calculate needed matrixes
        Lia = self.__calculate_lia_matrix( beta[0], beta[1], beta[2] )
        Ra = self.__calculate_ra_matrix( beta[0], beta[1] ) 

        # Calculate driver angular velocities
        w_d = self.__multiply_matrix_and_vector(Ra, beta_dot)

        # Calculate derivitive of angular rate
        _w_A_dot, self.w_prev = self.__calculate_derivitive_on_vector( w_d, self.w_prev, self.dt )

        # Calculate Aea matrix
        Aea = self.__calculate_aea_matrix( w_d[0], w_d[1], w_d[2], _w_A_dot[0], _w_A_dot[1], _w_A_dot[2] )

        # Calculate rotation affect on acceleration
        _a_rot = self.__multiply_matrix_and_vector( Aea, self.r_ea )

        # Apply rotation
        _a_AA = self.__multiply_matrix_and_vector( Lia, a )

        # Add rotations effect
        _a_EA = self.__sum_vectors( _a_AA, _a_rot )

        # Subtract gravity
        a_d = self.__subtract_vectors( _a_EA, ( self.__multiply_matrix_and_vector( Lia, [0,0,9.81] )))

        return a_d, w_d


    # NOTE: p-x axis, q-y axis, r-z axis rotations
    def __calculate_aea_matrix(self, p,  q, r, p_dot, q_dot, r_dot):
        Aps = [[0,0,0], [0,0,0], [0,0,0]]

        Aps[0][0] = -(q**2) - (r**2)
        Aps[0][1] = (p*q) - r_dot
        Aps[0][2] = (p*r) - q_dot

        Aps[1][0] = (p*q) - r_dot
        Aps[1][1] = -(p**2) - (r**2)
        Aps[1][2] = (q*r) - p_dot

        Aps[2][0] = (p*r) - q_dot
        Aps[2][1] = (q*r) - p_dot
        Aps[2][2] = -(p**2) - (q**2)

        return Aps

    
    def __calculate_ra_matrix(self, roll, pitch):
        Ra = [[0,0,0], [0,0,0], [0,0,0]]

        s_roll = np.sin(roll)
        c_roll = np.cos(roll)
        s_pitch = np.sin(pitch)
        c_pitch = np.cos(pitch)

        Ra[0][0] = 1
        Ra[0][1] = 0
        Ra[0][2] = -s_pitch

        Ra[1][0] = 0
        Ra[1][1] = c_roll
        Ra[1][2] = s_roll * c_pitch

        Ra[2][0] = 0
        Ra[2][1] = -s_roll
        Ra[2][2] = c_roll * c_pitch

        return Ra

    def __calculate_lia_matrix(self, roll, pitch, yaw):
        Lia = [[0,0,0], [0,0,0], [0,0,0]]

        s_roll  = np.sin(roll)
        c_roll  = np.cos(roll)
        s_pitch = np.sin(pitch)
        c_pitch = np.cos(pitch)
        s_yaw  = np.sin(yaw)
        c_yaw  = np.cos(yaw)

        Lia[0][0] = c_pitch * c_yaw
        Lia[0][1] = c_pitch * s_yaw
        Lia[0][2] = -s_pitch

        Lia[1][0] = (s_roll*s_pitch*c_yaw) - (c_roll*s_yaw)
        Lia[1][1] = (s_roll*s_pitch*s_yaw) + (c_roll*c_yaw)
        Lia[1][2] = s_roll*c_pitch

        Lia[2][0] = (c_roll*s_pitch*c_yaw) + (s_roll*s_yaw)
        Lia[2][1] = (c_roll*s_pitch*s_yaw) - (s_roll*c_yaw)
        Lia[2][2] = c_roll * c_pitch

        return Lia

    def __multiply_matrix_and_vector(self, matrix, vector):
        res_vector = [0] * 3
        for i in range(3):
            for j in range(3):
                res_vector[i] += matrix[i][j] * vector[j]
        return res_vector

    def __sum_vectors(self, vec_1, vec_2):
        res_vector = [0] * 3
        for i in range(3):
            res_vector[i] = vec_1[i] + vec_2[i]
        return res_vector

    def __subtract_vectors(self, vec_1, vec_2):
        res_vector = [0] * 3
        for i in range(3):
            res_vector[i] = vec_1[i] - vec_2[i]
        return res_vector

    def __calculate_derivitive(self, x, x_prev, dt):
        _dx = ( x - x_prev ) / dt
        return _dx

    def __calculate_derivitive_on_vector(self, vec, vec_prev, dt):
        _dvec = [0] * 3
        for n in range(3):
            _dvec[n] = self.__calculate_derivitive( vec[n], vec_prev[n], dt )
        return _dvec, vec


# ===============================================================================
#       MAIN ENTRY
# ===============================================================================
if __name__ == "__main__":

    """
    # Time array
    _time, _dt = np.linspace( 0.0, TIME_WINDOW, num=SAMPLE_NUM, retstep=True )

    
    # =====================================================================
    #   COMPONENTS OF SYSTEM
    # =====================================================================

    # Wahsout filter
    _filter_washout = WashoutFilter(    Wht=[WASHOUT_HPF_WHT_FC, WASHOUT_HPF_WHT_Z], Wrtzt=WASHOUT_HPF_WRTZT_FC, \
                                        W11=WASHOUT_HPF_W11_FC, W12=[WASHOUT_LPF_W12_FC, WASHOUT_LPF_W12_Z], fs=SAMPLE_FREQ )

    # Vestibular systems
    _vest_sys_test = VestibularSystem()
    _vest_sys_wash = VestibularSystem()


    # =====================================================================
    # SIGNALS OF SYSTEM
    # =====================================================================
    
    # System inputs: Accelerations & Angular velocities
    _a_in = [0] * 3
    _w_in = [0] * 3

    # Output of test vestibular system
    _a_sens_test = [0] * 3
    _w_sens_test = [0] * 3

    # Output of washout filter
    _a_wash = [0] * 3
    _w_wash = [0] * 3

    # Output of washout vestibular system
    _a_wash_sens = [0] * 3
    _w_wash_sens = [0] * 3

    # Sensation error
    _a_sens_err = [0] * 3
    _w_sens_err = [0] * 3

    # Position & rotation output
    _p_out = [0] * 3
    _r_out = [0] * 3



    # =====================================================================
    #   STIMULI COMPONENT/SIGNALS
    # =====================================================================

    # Generate inputs
    _fg = FunctionGenerator( INPUT_SIGNAL_FREQ, INPUT_SIGNAL_AMPLITUDE, INPUT_SIGNAL_OFFSET, INPUT_SIGNAL_PHASE, INPUT_SIGNAL_SELECTION )
    
    # Filter input/output
    _x = [ 0 ] * SAMPLE_NUM
    _x_d = [0]

    # Down sample
    _downsamp_cnt = 0
    _downsamp_samp = [0]
    _d_time = [0]
    
    # Generate stimuli signals
    for n in range(SAMPLE_NUM):
        _x[n] = ( _fg.generate( _time[n] ))


    # =====================================================================
    #   DATA FOR PLOTING
    # =====================================================================

    # System inputs: Accelerations & Angular velocities
    _y_d_a_in = [[0], [0], [0]] * 3
    _y_d_w_in = [[0], [0], [0]] * 3

    # Output of test vestibular system
    _y_d_a_sens_test = [[0], [0], [0]] * 3
    _y_d_w_sens_test = [[0], [0], [0]] * 3

    # Output of washout filter
    _y_d_a_wash = [[0], [0], [0]] * 3
    _y_d_w_wash = [[0], [0], [0]] * 3

    # Output of washout vestibular system
    _y_d_a_wash_sens = [[0], [0], [0]] * 3
    _y_d_w_wash_sens = [[0], [0], [0]] * 3

    # Sensation error
    _y_d_a_sens_err = [[0], [0], [0]] * 3
    _y_d_w_sens_err = [[0], [0], [0]] * 3

    # Position
    _y_d_p = [[0], [0], [0]] * 3
    
    # Rotation
    _y_d_r = [[0], [0], [0]] * 3





    # =====================================================================
    #   SIMULATION
    # =====================================================================
    
    # Apply filter
    for n in range(SAMPLE_NUM):

        # Down sample to SAMPLE_FREQ
        if _downsamp_cnt >= (( 1 / ( _dt * SAMPLE_FREQ )) - 1 ):
            _downsamp_cnt = 0

            # Utils
            _downsamp_samp.append(0)
            _d_time.append( _time[n])
            _x_d.append( _x[n] )


            # =====================================================================
            #   MODEL INPUT SELECTION
            # =====================================================================
            _a_in, _w_in = system_model_route_input_signal( _x[n], INPUT_SIGNAL_ROUTE )

            # =====================================================================
            #   SIMULATE MODEL
            # =====================================================================

            # Vestibular system - REAL SENSATION (reference)
            _a_sens_test, _w_sens_test = _vest_sys_test.update( _a_in, _w_in )

            # Washout filter
            _a_wash, _w_wash = _filter_washout.update( _a_in, _w_in )

            # Vestibular system - WASHOUT SENSATION
            _a_wash_sens, _w_wash_sens = _vest_sys_wash.update( _a_wash, _w_wash )

            # Calculate error in real and washout sensation
            _a_sens_err, _w_sens_err = system_model_calc_sens_error(  _a_sens_test, _w_sens_test, _a_wash_sens, _w_wash_sens )



            # TODO: calculate positions & rotations


            
            # =====================================================================
            #   LOG DATA FOR PLOTING
            # =====================================================================
            system_model_log_data( _y_d_a_in, _a_in )
            system_model_log_data( _y_d_w_in, _w_in )

            system_model_log_data( _y_d_a_sens_test, _a_sens_test )
            system_model_log_data( _y_d_w_sens_test, _w_sens_test )

            system_model_log_data( _y_d_a_wash, _a_wash )
            system_model_log_data( _y_d_w_wash, _w_wash )

            system_model_log_data( _y_d_a_wash_sens, _a_wash_sens )
            system_model_log_data( _y_d_w_wash_sens, _w_wash_sens )

            system_model_log_data( _y_d_a_sens_err, _a_sens_err )
            system_model_log_data( _y_d_w_sens_err, _w_sens_err )

        else:
            _downsamp_cnt += 1
    

    


    
    # =============================================================================================
    ## PLOT CONFIGURATIONS
    # =============================================================================================
    plt.style.use(['dark_background'])
    PLOT_MAIN_TITLE_SIZE    = 18
    PLOT_MAIN_TITLE         = "SYSTEM MODEL SIMULATIONS | fs: " + str(SAMPLE_FREQ) + "Hz" 
    PLOT_TITLE_SIZE         = 16
    PLOT_AXIS_LABEL_SIZE    = 12
    PLOT_ADJUST_LEFT        = 0.06
    PLOT_ADJUST_RIGHT       = 0.98
    PLOT_ADJUST_TOP         = 0.91
    PLOT_ADJUST_BOTTOM      = 0.05

    ## ==============================================================================================
    # Rotation motion plots
    ## ==============================================================================================
    fig, ax = plt.subplots(4, 1, sharex=True)
    fig.suptitle( PLOT_MAIN_TITLE , fontsize=PLOT_MAIN_TITLE_SIZE )

    # Subplot 0
    system_model_plot_signals( ax[0], _d_time, _y_d_a_in, _y_d_w_in )
    ax[0].set_title("Input acceleration & rotation", fontsize=PLOT_TITLE_SIZE)
    ax[0].grid(alpha=0.25)
    ax[0].legend(loc="upper right")
    ax[0].set_ylabel('Acceleration [m/s^2],\nAngular rate [rad/s]', fontsize=PLOT_AXIS_LABEL_SIZE)
        
    # Subplot 1
    system_model_plot_signals( ax[1], _d_time, _y_d_a_wash, _y_d_w_wash )
    ax[1].set_title("Washout filter output", fontsize=PLOT_TITLE_SIZE)
    ax[1].grid(alpha=0.25)
    ax[1].legend(loc="upper right")
    ax[1].set_ylabel('Acceleration [m/s^2],\nAngular rate [rad/s]', fontsize=PLOT_AXIS_LABEL_SIZE)

    # Subplot 2
    system_model_plot_signals( ax[2], _d_time, _y_d_a_wash_sens, _y_d_w_wash_sens )
    system_model_plot_signals( ax[2], _d_time, _y_d_a_sens_test, _y_d_w_sens_test )
    ax[2].set_title("Washout - actual feeling vs. Vastibular - reference feeling", fontsize=PLOT_TITLE_SIZE)
    ax[2].grid(alpha=0.25)
    ax[2].legend(loc="upper right")
    ax[2].set_ylabel('Acceleration [m/s^2],\nAngular rate [rad/s]', fontsize=PLOT_AXIS_LABEL_SIZE)

    # Subplot 3
    system_model_plot_signals( ax[3], _d_time, _y_d_a_sens_err, _y_d_w_sens_err )
    ax[3].set_title("Error in sensation", fontsize=PLOT_TITLE_SIZE)
    ax[3].grid(alpha=0.25)
    ax[3].legend(loc="upper right")
    ax[3].set_ylabel('Acceleration [m/s^2],\nAngular rate [rad/s]', fontsize=PLOT_AXIS_LABEL_SIZE)
    ax[3].set_xlabel('Time [s]', fontsize=PLOT_AXIS_LABEL_SIZE)

    plt.subplots_adjust(left=PLOT_ADJUST_LEFT, right=PLOT_ADJUST_RIGHT, top=PLOT_ADJUST_TOP, bottom=PLOT_ADJUST_BOTTOM)
    plt.show()
    
    """

# ===============================================================================
#       END OF FILE
# ===============================================================================
