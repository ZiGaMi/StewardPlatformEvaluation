# ===============================================================================
# @file:    washout_filter.py
# @note:    This script is evaluation of washout filter algorithm
# @author:  Ziga Miklosic
# @date:    11.01.2021
# @brief:   Evaluation of washout filter design. This evaluation is for designing
#           a working washout filter used for steward platform. 
# ===============================================================================

# ===============================================================================
#  TODO:
#
#   - Enable/Disable via check box showing pos/vel/acc as output of w filter
#   - Make sliders for setting coefficients of filters
#   - Make 5 additional signal generators to simulate all dimensions
#   - Make easy configuration to change stimuli signal to acceleration or rotation
# ===============================================================================

# ===============================================================================
#       IMPORTS  
# ===============================================================================
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz, butter, cheby1, lfilter, filtfilt, bilinear

from filter_utils import FunctionGenerator, SignalMux
from iir_filter import IIR, calculate_2nd_order_HPF_coeff, calculate_1nd_order_HPF_coeff, calculate_2nd_order_LPF_coeff

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
IDEAL_SAMPLE_FREQ = 20000.0

## Time window
#
# Unit: second
TIME_WINDOW = 2

## Input signal shape
INPUT_SIGNAL_FREQ = 0.1
INPUT_SIGNAL_AMPLITUDE = 0.5
INPUT_SIGNAL_OFFSET = 0.0
INPUT_SIGNAL_PHASE = -0.25

## Mux input signal
INPUT_SIGNAL_SELECTION = SignalMux.MUX_CTRL_RECT

## Number of samples in time window
SAMPLE_NUM = int(( IDEAL_SAMPLE_FREQ * TIME_WINDOW ) + 1.0 )


# =====================================================
## TRANSLATION CHANNEL SETTINGS

# HPF Wht 2nd order filter
WASHOUT_HPF_WHT_FC  = 1.0
WASHOUT_HPF_WHT_Z   = .7071

# HPF Wrtzt 1st order filter
WASHOUT_HPF_WRTZT_FC  = 1.0

# SCALE AND LIMIT
# [ x, y, z]
WASHOUT_SCALE_A_T = [ 1.0, 1.0, 1.0 ]
WASHOUT_LIMIT_A_T = [ 2.0, 2.0, 2.0 ] # Limits are symetrical


# =====================================================
## COORDINATION CHANNEL SETTINGS

# LPF W12 2nd order filter
WASHOUT_LPF_W12_FC  = 1.0
WASHOUT_LPF_W12_Z   = 1.0

# SCALE AND LIMIT 
# [ x, y, z]
WASHOUT_SCALE_A_C = [ 1.0, 1.0, 1.0 ]
WASHOUT_LIMIT_A_C = [ 2.0, 2.0, 2.0 ] # Limits are symetrical

# Gravity constant
G = 9.18
G_INV = 1.0 / G

# TILT MATRIX
WASHOUT_TILT_MATRIX = [ [0,     -G_INV,  0],
                        [G_INV, 0,       0],
                        [0,     0,       0] ]

# TILT LIMIT
WASHOUT_TILT_LIMIT = [ 0.2, 0.2, 0.2 ]


# =====================================================
## ROTATION CHANNEL SETTINGS

# HPF W11 1st order filter
WASHOUT_HPF_W11_FC  = 1.0

# SCALE AND LIMIT
# [ rool, pitch, yaw]
WASHOUT_SCALE_BETA = [ 1.0, 1.0, 1.0 ]
WASHOUT_LIMIT_BETA = [ 0.2, 0.2, 0.2 ]


## ****** END OF USER CONFIGURATIONS ******

# ===============================================================================
#       FUNCTIONS
# ===============================================================================

# ===============================================================================
#       CLASSES
# ===============================================================================    


## Integration
class Integrator:

    def __init__(self, init_val=0):
        self.y = init_val

    def update(self, x):
        self.y += x
        return self.y


## IIR Filter
class Washout:

    def __init__(self, Wht, Wrtzt, W11, W12, fs):

        # Translation channel filters
        b, a = calculate_2nd_order_HPF_coeff( Wht[0], Wht[1], fs )
        self._hpf_wht = [0] * 3
        self._hpf_wht[0] = IIR( a, b, 2 )
        self._hpf_wht[1] = IIR( a, b, 2 )
        self._hpf_wht[2] = IIR( a, b, 2 )

        b, a = calculate_1nd_order_HPF_coeff( Wrtzt, fs )
        self._hpf_wrtzt = [0] * 3
        self._hpf_wrtzt[0] = IIR( a, b, 1 )
        self._hpf_wrtzt[1] = IIR( a, b, 1 )
        self._hpf_wrtzt[2] = IIR( a, b, 1 )

        # Coordination channel filters
        b, a = calculate_2nd_order_LPF_coeff( W12[0], W12[1], fs )
        self._lpf_w12 = [0] * 3
        self._lpf_w12[0] = IIR( a, b, 2 )
        self._lpf_w12[1] = IIR( a, b, 2 )
        self._lpf_w12[2] = IIR( a, b, 2 )

        # Rotation channel filters
        b, a = calculate_1nd_order_HPF_coeff( W11, fs )
        self._hpf_w22 = [0] * 3
        self._hpf_w22[0] = IIR( a, b, 1 )
        self._hpf_w22[1] = IIR( a, b, 1 )
        self._hpf_w22[2] = IIR( a, b, 1 )

        # Translation channel integrators
        self._int_x = [0] * 2
        self._int_y = [0] * 2
        self._int_z = [0] * 2
        for n in range(2):
            self._int_x[n] = Integrator()
            self._int_y[n] = Integrator()
            self._int_z[n] = Integrator()


    # ===============================================================================
    # @brief: Update washout filter
    #
    # @param[in]:    a      - Vector of accelerations
    # @param[in]:    beta   - Vector of angular velocities
    # @return:       p, r   - Positions and rotation of a steward platform
    # ===============================================================================
    def update(self, a, beta):

        # Translation channel
        a_t = [0] * 3

        # Coordination channel
        a_c = [0] * 3

        # Rotations channel
        beta_r = [0] * 3

        # Translation channel scaling/limitation/filtering/integration
        for n in range(3):
            a_t[n] = self.__scale_limit( a[n], WASHOUT_SCALE_A_T[n], WASHOUT_LIMIT_A_T[n] )
            a_t[n] = self._hpf_wht[n].update( a_t[n] )
            a_t[n] = self._hpf_wrtzt[n].update( a_t[n] )

        # Integration
        a_t[0] = self._int_x[0].update(a_t[0])
        a_t[0] = self._int_x[1].update(a_t[0])

        a_t[1] = self._int_y[0].update(a_t[1])
        a_t[1] = self._int_y[1].update(a_t[1])

        a_t[2] = self._int_z[0].update(a_t[2])
        a_t[2] = self._int_z[1].update(a_t[2])

        # Coordingation channel scaling/limitation/filtering
        for n in range(3):
            a_c[n]  = self.__scale_limit( a[n], WASHOUT_SCALE_A_C[n], WASHOUT_LIMIT_A_C[n] )
            a_c[n] = self._lpf_w12[n].update( a_c[n] )

        # Tilt coordination
        a_c_tilt = [0] * 3
        for n in range(3):
            for j in range(3):
                a_c_tilt[n] += WASHOUT_TILT_MATRIX[n][j] * a_c[j]

        # Rotaion scaling/limitation/filtering
        for n in range(3):
            beta_r[n] = self.__scale_limit( beta[n], WASHOUT_SCALE_BETA[n], WASHOUT_LIMIT_BETA[n] )
            beta_r[n] = self._hpf_w22[n].update( beta_r[n] )

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





# ===============================================================================
#       MAIN ENTRY
# ===============================================================================
if __name__ == "__main__":

    # Time array
    _time, _dt = np.linspace( 0.0, TIME_WINDOW, num=SAMPLE_NUM, retstep=True )

    
    # Filter object
    _filter_washout = Washout(  Wht=[WASHOUT_HPF_WHT_FC, WASHOUT_HPF_WHT_Z], Wrtzt=WASHOUT_HPF_WRTZT_FC, \
                                W11=WASHOUT_HPF_W11_FC, W12=[WASHOUT_LPF_W12_FC, WASHOUT_LPF_W12_Z], fs=SAMPLE_FREQ )

    # Filter input/output
    _x = [ 0 ] * SAMPLE_NUM
    _x_d = [0]

    # Position
    _y_d_p = [[0], [0], [0]] * 3
    
    # Rotation
    _y_d_r = [[0], [0], [0]] * 3


    # Generate inputs
    _fg_sine = FunctionGenerator( INPUT_SIGNAL_FREQ, INPUT_SIGNAL_AMPLITUDE, INPUT_SIGNAL_OFFSET, INPUT_SIGNAL_PHASE, "sine" )
    _fg_rect = FunctionGenerator( INPUT_SIGNAL_FREQ, INPUT_SIGNAL_AMPLITUDE, INPUT_SIGNAL_OFFSET, INPUT_SIGNAL_PHASE, "rect" )
    _sin_x = []
    _rect_x = []

    # Signal mux
    # NOTE: Support only sine and rectange
    _signa_mux = SignalMux( 2 )
    
    # Down sample
    _downsamp_cnt = 0
    _downsamp_samp = [0]
    _d_time = [0]
    
    # Generate stimuli signals
    for n in range(SAMPLE_NUM):
        _sin_x.append( _fg_sine.generate( _time[n] ))
        _rect_x.append( _fg_rect.generate( _time[n] ))
 
    # Apply filter
    for n in range(SAMPLE_NUM):
        
        # Mux input signals
        _x[n] = _signa_mux.out( INPUT_SIGNAL_SELECTION, [ _sin_x[n], _rect_x[n] ] )

        # Down sample to SAMPLE_FREQ
        if _downsamp_cnt >= (( 1 / ( _dt * SAMPLE_FREQ )) - 1 ):
            _downsamp_cnt = 0

            # Utils
            _downsamp_samp.append(0)
            _d_time.append( _time[n])
            _x_d.append( _x[n] )

            p, r = _filter_washout.update( [ _x[n], 0, 0 ], [ 0, 0, 0 ] )
            _y_d_p[0].append( p[0] )
            _y_d_p[1].append( p[1] )
            _y_d_p[2].append( p[2] )
            _y_d_r[0].append( r[0] )
            _y_d_r[1].append( r[1] )
            _y_d_r[2].append( r[2] )

        else:
            _downsamp_cnt += 1
    
    # Plot results
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle( "WASHOUT FILTERS\n fs: " + str(SAMPLE_FREQ) + "Hz", fontsize=20 )

    ax11 = ax[0].twinx() 
    ax11.plot( _time, _x, "b--", lw=1.0, label="ax" )
    ax11.set_ylim( -2, 2 )
    ax11.legend(loc="lower right")
    ax11.set_ylabel('Acceleration [m/s^2]', fontsize=14)

    ax[0].set_title("Translations", fontsize=16)
    ax[0].plot( _d_time, _y_d_p[0],         "g.-",    label="x")
    ax[0].plot( _d_time, _y_d_p[1],         "r.-",    label="y")
    ax[0].plot( _d_time, _y_d_p[2],         "y.-",    label="z")
    ax[0].set_ylim(-80, 80)
    ax[0].set_ylabel('Translation [mm]', fontsize=14)
    ax[0].grid()
    ax[0].legend(loc="upper right")

    ax[1].set_title("Rotations", fontsize=16)
    ax[1].plot( _d_time, _y_d_r[0],         "g.-",    label="roll")
    ax[1].plot( _d_time, _y_d_r[1],         "r.-",    label="pitch")
    ax[1].plot( _d_time, _y_d_r[2],         "y.-",    label="yaw")

    ax[1].set_ylabel('Amplitude')
    ax[1].set_xlabel('Time [s]')
    ax[1].grid()
    ax[1].legend(loc="upper right")

    plt.show()
    


# ===============================================================================
#       END OF FILE
# ===============================================================================
