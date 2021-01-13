# ===============================================================================
# @file:    human_perception.py
# @note:    This script is for model design of human perception of movement 
# @author:  Ziga Miklosic
# @date:    13.01.2021
# @brief:   Evaluation of human movement perception model base on 
#           "Vehicle modelling and washout filter tuning for the Chalmers Vehicle
#           Simulator" thesis.
# ===============================================================================


# ===============================================================================
#       IMPORTS  
# ===============================================================================
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz, bilinear

from filters.filter_utils import FunctionGenerator, SignalMux
from filters.iir_filter import IIR

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


## Parameters of the vestibular system
VESTIBULAR_ROLL_TL      = 6.1
VESTIBULAR_ROLL_TS      = 0.1
VESTIBULAR_ROLL_TA      = 30.0

VESTIBULAR_PITCH_TL     = 5.3
VESTIBULAR_PITCH_TS     = 0.1
VESTIBULAR_PITCH_TA     = 30.0

VESTIBULAR_YAW_TL       = 10.2
VESTIBULAR_YAW_TS       = 0.1
VESTIBULAR_YAW_TA       = 30.0

VESTIBULAR_X_TL         = 5.33
VESTIBULAR_X_TS         = 0.66
VESTIBULAR_X_TA         = 13.2
VESTIBULAR_X_K          = 0.4

VESTIBULAR_Y_TL         = 5.33
VESTIBULAR_Y_TS         = 0.66
VESTIBULAR_Y_TA         = 13.2
VESTIBULAR_Y_K          = 0.4

VESTIBULAR_Z_TL         = 5.33
VESTIBULAR_Z_TS         = 0.66
VESTIBULAR_Z_TA         = 13.2
VESTIBULAR_Z_K          = 0.4




## ****** END OF USER CONFIGURATIONS ******

# ===============================================================================
#       FUNCTIONS
# ===============================================================================


# ===============================================================================
# @brief:   Calculate rotation perception coefficients
#           
#   h(s) = (1/Ts * s^2) / ( s^3 + (1/Ta + 1/Tl + 1/Ts)*s^2 + (1/Tl*Ts + 1/Tl*Ta + 1/Ta*Ts)*s + 1/(Ta*Tl*Ts)))
#
# @param[in]:    Tl, Ts, Ta - Coefficient in the semicircular canals sensation model
# @param[in]:    fs         - Sample frequency
# @return:       b,a        - Array of b,a IIR coefficients
# ===============================================================================
def calc_rot_mov_coefficient(Tl, Ts, Ta, fs):

    b, a = bilinear( [1/Ts, 0, 0], [1, (1/Ta + 1/Tl + 1/Ts), (1/Tl*Ts + 1/Tl*Ta + 1/Ta*Ts), 1/(Ta*Tl*Ts)], fs )

    return b, a


# ===============================================================================
# @brief:   Calculate linear movement perception coefficients
#           
#   h(s) = ((K*Ta/(Tl*Ts))*s + (K*Ta/(Tl*Ts))) / ( s^2 + (1/Tl + 1/Ts)*s + 1/(Tl*Ts))
#
# @param[in]:    Tl, Ts, Ta, K  - Coefficients in the otolith model
# @param[in]:    fs             - Sample frequency
# @return:       b,a            - Array of b,a IIR coefficients
# ===============================================================================
def calc_lin_mov_coefficient(Tl, Ts, Ta, K, fs):

    b, a = bilinear( [(K*Ta/(Tl*Ts)), (K*Ta/(Tl*Ts))], [1, (1/Tl + 1/Ts), 1/(Tl*Ts)], fs )

    return b, a


# ===============================================================================
#       CLASSES
# ===============================================================================    






# ===============================================================================
#       MAIN ENTRY
# ===============================================================================
if __name__ == "__main__":

    # Time array
    _time, _dt = np.linspace( 0.0, TIME_WINDOW, num=SAMPLE_NUM, retstep=True )

    # Rotation movement coefficient
    _roll_b, _roll_a    = calc_rot_mov_coefficient( VESTIBULAR_ROLL_TL, VESTIBULAR_ROLL_TS, VESTIBULAR_ROLL_TA, SAMPLE_FREQ )
    _pitch_b, _pitch_a  = calc_rot_mov_coefficient( VESTIBULAR_PITCH_TL, VESTIBULAR_PITCH_TS, VESTIBULAR_PITCH_TA, SAMPLE_FREQ )
    _yaw_b, _yaw_a      = calc_rot_mov_coefficient( VESTIBULAR_YAW_TL, VESTIBULAR_YAW_TS, VESTIBULAR_YAW_TA, SAMPLE_FREQ )

    # Linear movement coefficient
    _x_b, _x_a = calc_lin_mov_coefficient( VESTIBULAR_X_TL, VESTIBULAR_X_TS, VESTIBULAR_X_TA, VESTIBULAR_X_K, SAMPLE_FREQ )
    _y_b, _y_a = calc_lin_mov_coefficient( VESTIBULAR_Y_TL, VESTIBULAR_Y_TS, VESTIBULAR_Y_TA, VESTIBULAR_Y_K, SAMPLE_FREQ )
    _z_b, _z_a = calc_lin_mov_coefficient( VESTIBULAR_Z_TL, VESTIBULAR_Z_TS, VESTIBULAR_Z_TA, VESTIBULAR_Z_K, SAMPLE_FREQ )

    # Filters
    _roll_filt  = IIR( a=_roll_a,   b=_roll_b,  order=3 )
    _pitch_filt = IIR( a=_pitch_a,  b=_pitch_b, order=3 )
    _yaw_filt   = IIR( a=_yaw_a,    b=_yaw_b,   order=3 )

    # Get frequency characteristics
    _roll_w, _roll_h    = freqz( _roll_b,  _roll_a,    4096 * 256 )
    _pitch_w, _pitch_h  = freqz( _pitch_b, _pitch_a,   4096 * 256 )
    _yaw_w, _yaw_h      = freqz( _yaw_b,   _yaw_a,     4096 * 256 )

    _x_w, _x_h = freqz( _x_b, _x_a, 4096 * 256 )
    _y_w, _y_h = freqz( _y_b, _y_a, 4096 * 256 )
    _z_w, _z_h = freqz( _z_b, _z_a, 4096 * 256 )

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


        else:
            _downsamp_cnt += 1
    

    # Calculate frequency response
    _roll_w     = ( _roll_w / np.pi * SAMPLE_FREQ / 2)  # Hz
    _pitch_w    = ( _pitch_w / np.pi * SAMPLE_FREQ / 2) # Hz
    _yaw_w      = ( _yaw_w / np.pi * SAMPLE_FREQ / 2)   # Hz

    _x_w = ( _x_w / np.pi * SAMPLE_FREQ / 2)  # Hz
    _y_w = ( _y_w / np.pi * SAMPLE_FREQ / 2)  # Hz
    _z_w = ( _z_w / np.pi * SAMPLE_FREQ / 2)  # Hz

    # For conversion to rad/s
    _roll_w = 2*np.pi*_roll_w  
    _pitch_w = 2*np.pi*_pitch_w  
    _yaw_w = 2*np.pi*_yaw_w  

    _x_w = 2*np.pi*_x_w  
    _y_w = 2*np.pi*_y_w  
    _z_w = 2*np.pi*_z_w  

    # Calculate phases & convert to degrees
    _roll_angle     = np.unwrap( np.angle(_roll_h) )    * 180/np.pi
    _pitch_angle    = np.unwrap( np.angle(_pitch_h) )   * 180/np.pi
    _yaw_angle      = np.unwrap( np.angle(_yaw_h) )     * 180/np.pi

    _x_angle = np.unwrap( np.angle(_x_h) )     * 180/np.pi
    _y_angle = np.unwrap( np.angle(_y_h) )     * 180/np.pi
    _z_angle = np.unwrap( np.angle(_z_h) )     * 180/np.pi

    # Plot results
    fig, ax = plt.subplots(2, 1)
    fig.suptitle( "ROTATION MOVEMENT MODEL\n fs: " + str(SAMPLE_FREQ) + "Hz", fontsize=20 )

    ax[0].plot(_roll_w,     20 * np.log10(abs(_roll_h)),    'g', label="roll")
    ax[0].plot(_pitch_w,    20 * np.log10(abs(_pitch_h)),   'r', label="pitch")
    ax[0].plot(_yaw_w,      20 * np.log10(abs(_yaw_h)),     'b', label="yaw")

    ax[0].grid()
    ax[0].set_xscale("log")
    ax[0].legend(loc="upper right")
    ax[0].set_xlim(1e-3, SAMPLE_FREQ/2)
    ax[0].set_ylim(-80, 2)
    ax[0].set_ylabel("Magnitude [dB]")


    ax[1].plot(_roll_w,     _roll_angle,    'g', label="roll")
    ax[1].plot(_pitch_w,    _pitch_angle,   'r', label="pitch")
    ax[1].plot(_yaw_w,      _yaw_angle,     'b', label="yaw")

    ax[1].set_ylabel("Angle [degrees]")
    ax[1].set_xscale("log")
    ax[1].grid()
    ax[1].legend(loc="upper right")
    ax[1].set_xlim(1e-3, SAMPLE_FREQ/2)
    ax[1].set_xlabel("Frequency [Hz]")

    fig, ax = plt.subplots(2, 1)
    fig.suptitle( "LINEAR MOVEMENT MODEL\n fs: " + str(SAMPLE_FREQ) + "Hz", fontsize=20 )

    ax[0].plot(_x_w, 20 * np.log10(abs(_x_h)), 'g', label="x")
    ax[0].plot(_y_w, 20 * np.log10(abs(_y_h)), 'r', label="y")
    ax[0].plot(_z_w, 20 * np.log10(abs(_z_h)), 'b', label="z")

    ax[0].grid()
    ax[0].set_xscale("log")
    ax[0].legend(loc="upper right")
    ax[0].set_xlim(1e-3, SAMPLE_FREQ/2)
   # ax[0].set_ylim(-80, 2)
    ax[0].set_ylabel("Magnitude [dB]")

    ax[1].plot(_x_w, _x_angle, 'g', label="x")
    ax[1].plot(_y_w, _y_angle, 'r', label="y")
    ax[1].plot(_z_w, _z_angle, 'b', label="z")

    ax[1].set_ylabel("Angle [degrees]")
    ax[1].set_xscale("log")
    ax[1].grid()
    ax[1].legend(loc="upper right")
    ax[1].set_xlim(1e-3, SAMPLE_FREQ/2)
    ax[1].set_xlabel("Frequency [Hz]")

    plt.show()
    


# ===============================================================================
#       END OF FILE
# ===============================================================================
