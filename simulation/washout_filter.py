# ===============================================================================
# @file:    washout_filter.py
# @note:    This script is evaluation of washout filter algorithm
# @author:  Ziga Miklosic
# @date:    11.01.2021
# @brief:   Evaluation of washout filter design. This evaluation is for designing
#           a working washout filter used for steward platform. 
# ===============================================================================

# ===============================================================================
#       IMPORTS  
# ===============================================================================
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz, butter, cheby1, lfilter, filtfilt, bilinear

from filters.filter_utils import FunctionGenerator
from filters.iir_filter import IIR, calculate_2nd_order_HPF_coeff, calculate_1nd_order_HPF_coeff, calculate_2nd_order_LPF_coeff

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

# ===============================================================================
#       CLASSES
# ===============================================================================    

## Wahsout filter
class WashoutFilter:

    ## TRANSLATION CHANNEL SETTINGS
    # SCALE AND LIMIT
    # [ x, y, z]
    WASHOUT_SCALE_A_T = [ 1.0, 1.0, 1.0 ]
    WASHOUT_LIMIT_A_T = [ 2.0, 2.0, 2.0 ] # Limits are symetrical

    ## COORDINATION CHANNEL SETTINGS
    # SCALE AND LIMIT 
    # [ x, y, z]
    WASHOUT_SCALE_A_C = [ 1.0, 1.0, 1.0 ]
    WASHOUT_LIMIT_A_C = [ 2.0, 2.0, 2.0 ] # Limits are symetrical

    ## ROTATION CHANNEL SETTINGS        
    # SCALE AND LIMIT
    # [ rool, pitch, yaw]
    WASHOUT_SCALE_BETA = [ 1.0, 1.0, 1.0 ]
    WASHOUT_LIMIT_BETA = [ 0.5, 0.5, 0.5 ]

    # Gravity constant
    G = 9.18
    G_INV = 1.0 / G

    # TILT MATRIX
    WASHOUT_TILT_MATRIX = [ [0,         G_INV,  0],
                            [-G_INV,    0,      0],
                            [0,         0,      0] ]

    # TILT LIMIT
    WASHOUT_TILT_LIMIT = [ 0.2, 0.2, 0.2 ] # m/s^2
    WASHOUT_TILT_RATE_LIM_DEG = 10 #deg/s

    # ===============================================================================
    # @brief: Initialization of filter
    #
    # @param[in]:    Wht        - Translation channel HPF coefficient
    # @param[in]:    Wrtzt      - Translation channel HPF (return to zero) coefficient
    # @param[in]:    W11        - Rotation channel HPF coefficient
    # @param[in]:    W12        - Tilt coordination channel LPF coefficient
    # @param[in]:    fs         - Sample frequency
    # @return:       void
    # ===============================================================================
    def __init__(self, Wht, Wrtzt, W11, W12, fs):
        self.dt = 1 / fs

        # -----------------------------------------------
        # Translation channel filters
        # -----------------------------------------------
        self._hpf_wht = [0] * 3
        self._hpf_wrtzt = [0] * 3
        
        # X
        b, a = calculate_2nd_order_HPF_coeff( Wht[0][0], Wht[0][1], fs )
        self._hpf_wht[0] = IIR( a, b, 2 )

        b, a = calculate_1nd_order_HPF_coeff( Wrtzt[0], fs )
        self._hpf_wrtzt[0] = IIR( a, b, 1 )

        # Y
        b, a = calculate_2nd_order_HPF_coeff( Wht[1][0], Wht[1][1], fs )
        self._hpf_wht[1] = IIR( a, b, 2 )

        b, a = calculate_1nd_order_HPF_coeff( Wrtzt[1], fs )
        self._hpf_wrtzt[1] = IIR( a, b, 1 )

        # Z
        b, a = calculate_2nd_order_HPF_coeff( Wht[2][0], Wht[2][1], fs )
        self._hpf_wht[2] = IIR( a, b, 2 )

        b, a = calculate_1nd_order_HPF_coeff( Wrtzt[2], fs )
        self._hpf_wrtzt[2] = IIR( a, b, 1 )
        
        # -----------------------------------------------
        # Coordination channel filters
        # -----------------------------------------------
        self._lpf_w12 = [0] * 3

        # Roll
        b, a = calculate_2nd_order_LPF_coeff( W12[0][0], W12[0][1], fs )
        self._lpf_w12[0] = IIR( a, b, 2 )

        # Pitch
        b, a = calculate_2nd_order_LPF_coeff( W12[1][0], W12[1][1], fs )
        self._lpf_w12[1] = IIR( a, b, 2 )        

        # Previous value of tilt for rate limiter
        self.a_c_tilt_prev = [0] * 2

        # -----------------------------------------------
        # Rotation channel filters
        # -----------------------------------------------
        self._hpf_w11 = [0] * 3

        # Roll
        b, a = calculate_1nd_order_HPF_coeff( W11[0], fs )
        self._hpf_w11[0] = IIR( a, b, 1 )

        # Pitch
        b, a = calculate_1nd_order_HPF_coeff( W11[1], fs )
        self._hpf_w11[1] = IIR( a, b, 1 )

        # Yaw
        b, a = calculate_1nd_order_HPF_coeff( W11[2], fs )
        self._hpf_w11[2] = IIR( a, b, 1 )


    # ===============================================================================
    # @brief: Update washout filter
    #
    # @note:    There are many implementation of washout filter, some of them takes 
    #           angular velocities as input other rotations. They result is equal if
    #           some preconditions are taken into account.
    #
    # @param[in]:    a          - Vector of accelerations
    # @param[in]:    beta       - Vector of rotations
    # @return:       a, beta    - Accelerations and angular rate filtered
    # ===============================================================================
    def update(self, a, beta):

        # Translation channel
        a_t = [0] * 3

        # Coordination channel
        a_c = [0] * 3

        # Rotations channel
        beta_r = [0] * 3

        # Translation channel scaling/limitation/filtering
        for n in range(3):
            a_t[n] = self.__scale_limit( a[n], self.WASHOUT_SCALE_A_T[n], self.WASHOUT_LIMIT_A_T[n] )
            a_t[n] = self._hpf_wht[n].update( a_t[n] )
            a_t[n] = self._hpf_wrtzt[n].update( a_t[n] )

        # Coordingation channel scaling/limitation/filtering
        for n in range(3):
            a_c[n]  = self.__scale_limit( a[n], self.WASHOUT_SCALE_A_C[n], self.WASHOUT_LIMIT_A_C[n] )

        # NOTE: no need for yaw filtering!
        for n in range(2):
            a_c[n] = self._lpf_w12[n].update( a_c[n] )

        # Tilt coordination
        a_c_tilt = [0] * 3
        for n in range(3):
            for j in range(3):
                a_c_tilt[n] += self.WASHOUT_TILT_MATRIX[n][j] * a_c[j]

        # Tilt rate limiter
        for n in range(2):
            a_c_tilt[n] = self.__rate_limit( a_c_tilt[n], self.a_c_tilt_prev[n], (self.WASHOUT_TILT_RATE_LIM_DEG*np.pi/180)  * self.dt )
            self.a_c_tilt_prev[n] = a_c_tilt[n]

        # Rotaion scaling/limitation/filtering
        for n in range(3):
            beta_r[n] = self.__scale_limit( beta[n], self.WASHOUT_SCALE_BETA[n], self.WASHOUT_LIMIT_BETA[n] )
            beta_r[n] = self._hpf_w11[n].update( beta_r[n] )

            # Add tilt to rotation channel
            beta_r[n] += a_c_tilt[n]

        return a_t, beta_r


    # ===============================================================================
    # @brief: Scale and limit input signals
    #
    #   NOTE: For know limiting is very simple, shall be change to poly order 3!!!
    #
    # @param[in]:    x      - Input signal value
    # @param[in]:    scale  - Scale factor
    # @param[in]:    lim    - Limit factor
    # @return:       y      - Scaled and limited value
    # ===============================================================================
    def __scale_limit(self, x, scale, lim):
        y = scale * x

        if y > lim:
            y = lim
        elif y < -lim:
            y = -lim
        else:
            pass

        return y

    # NOTE: usage: __rate_limit( r[0], _roll_lim_prev, 10*np.pi/180 / SAMPLE_FREQ )

    # ===============================================================================
    # @brief: Rate limiter
    #
    # @note: rate_lim parameter takes into account also dt. 
    #
    # @param[in]:    x          - Input signal value
    # @param[in]:    x_prev     - Previous value of input signal
    # @param[in]:    rate_lim   - Rate limit
    # @return:       _x_lim     - Rate limited input signal
    # ===============================================================================
    def __rate_limit(self, x, x_prev, rate_lim):
        _dx = ( x - x_prev )
        _x_lim = x_prev

        if _dx > rate_lim:
            _x_lim += rate_lim
        elif _dx < -rate_lim:
            _x_lim -= rate_lim
        else:
            _x_lim += _dx

        return _x_lim


# ===============================================================================
#       MAIN ENTRY
# ===============================================================================
if __name__ == "__main__":

    # Time array
    _time, _dt = np.linspace( 0.0, TIME_WINDOW, num=SAMPLE_NUM, retstep=True )

    # Filter object
    _filter_washout = WashoutFilter(    Wht=WASHOUT_HPF_WHT_COEFFICIENT, Wrtzt=WASHOUT_HPF_WRTZT_COEFFICIENT, \
                                        W11=WASHOUT_HPF_W11_COEFFICIENT, W12=WASHOUT_LPF_W12_COEFFICIENT, fs=SAMPLE_FREQ )

    # Filter input/output
    _x = [ 0 ] * SAMPLE_NUM
    _x_d = [0]

    # Position
    _y_d_p = [[0], [0], [0]] * 3
    
    # Rotation
    _y_d_r = [[0], [0], [0]] * 3

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
        
        
 
    # Apply filter
    for n in range(SAMPLE_NUM):

        # Down sample to SAMPLE_FREQ
        if _downsamp_cnt >= (( 1 / ( _dt * SAMPLE_FREQ )) - 1 ):
            _downsamp_cnt = 0

            # Utils
            _downsamp_samp.append(0)
            _d_time.append( _time[n])
            _x_d.append( _x[n] )

            p, r = _filter_washout.update( [ _x[n]/10, _x[n], 0 ], [ 0, 0, 0 ] )
            _y_d_p[0].append( p[0] )
            _y_d_p[1].append( p[1] )
            _y_d_p[2].append( p[2] )
            _y_d_r[0].append( r[0]*180/np.pi )
            _y_d_r[1].append( r[1]*180/np.pi )
            _y_d_r[2].append( r[2]*180/np.pi )
        else:
            _downsamp_cnt += 1
    

    plt.style.use(['dark_background'])
    ## ==============================================================================================
    # Rotation motion plots
    ## ==============================================================================================
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle( "WASHOUT FILTERS\n fs: " + str(SAMPLE_FREQ) + "Hz", fontsize=20 )

    """
    ax11 = ax[0].twinx() 
    ax11.plot( _time, _x, "w--", lw=1.0, label="ax" )
   # ax11.set_ylim( -2, 2 )
    ax11.legend(loc="lower right")
    ax11.set_ylabel('Acceleration [m/s^2]', fontsize=14)
    """


    ax[0].set_title("Translations", fontsize=16)
    ax[0].plot( _time, _x, "w--", lw=1.0, label="ax" )
    ax[0].plot( _d_time, _y_d_p[0],         "g.-",    label="x")
    ax[0].plot( _d_time, _y_d_p[1],         "r.-",    label="y")
    ax[0].plot( _d_time, _y_d_p[2],         "y.-",    label="z")
    #ax[0].set_ylim(-80, 80)
    ax[0].set_ylabel('Translation [mm]', fontsize=14)
    ax[0].grid(alpha=0.25)
    ax[0].legend(loc="upper right")

    ax[1].set_title("Rotations", fontsize=16)
    ax[1].plot( _d_time, _y_d_r[0],         "g.-",    label="roll")
    ax[1].plot( _d_time, _y_d_r[1],         "r.-",    label="pitch")
    ax[1].plot( _d_time, _y_d_r[2],         "y.-",    label="yaw")

    ax[1].set_ylabel('Amplitude [deg]')
    ax[1].set_xlabel('Time [s]')
    ax[1].grid(alpha=0.25)
    ax[1].legend(loc="upper right")

    plt.show()
    


# ===============================================================================
#       END OF FILE
# ===============================================================================
