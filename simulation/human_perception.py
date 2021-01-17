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

from filters.filter_utils import FunctionGenerator
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
IDEAL_SAMPLE_FREQ = 1000.0

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

    b, a = bilinear( [0, 1/Ts, 0, 0], [1, (1/Ta + 1/Tl + 1/Ts), (1/Tl*Ts + 1/Tl*Ta + 1/Ta*Ts), 1/(Ta*Tl*Ts)], fs )

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

    b, a = bilinear( [0, K*Ta, K], [ Tl*Ts, Tl+Ts, 1], fs )

    return b, a

# ===============================================================================
#       CLASSES
# ===============================================================================    

## Vestibular system
class VestibularSystem:

    # ===============================================================================
    # @brief:   Init vestibular system
    #           
    # @return: void
    # ===============================================================================
    def __init__(self):

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

        # Rotation movement coefficient
        _roll_b, _roll_a    = self.__calc_rot_mov_coefficient( VESTIBULAR_ROLL_TL, VESTIBULAR_ROLL_TS, VESTIBULAR_ROLL_TA, SAMPLE_FREQ )
        _pitch_b, _pitch_a  = self.__calc_rot_mov_coefficient( VESTIBULAR_PITCH_TL, VESTIBULAR_PITCH_TS, VESTIBULAR_PITCH_TA, SAMPLE_FREQ )
        _yaw_b, _yaw_a      = self.__calc_rot_mov_coefficient( VESTIBULAR_YAW_TL, VESTIBULAR_YAW_TS, VESTIBULAR_YAW_TA, SAMPLE_FREQ )

        # Linear movement coefficient
        _x_b, _x_a = self.__calc_lin_mov_coefficient( VESTIBULAR_X_TL, VESTIBULAR_X_TS, VESTIBULAR_X_TA, VESTIBULAR_X_K, SAMPLE_FREQ )
        _y_b, _y_a = self.__calc_lin_mov_coefficient( VESTIBULAR_Y_TL, VESTIBULAR_Y_TS, VESTIBULAR_Y_TA, VESTIBULAR_Y_K, SAMPLE_FREQ )
        _z_b, _z_a = self.__calc_lin_mov_coefficient( VESTIBULAR_Z_TL, VESTIBULAR_Z_TS, VESTIBULAR_Z_TA, VESTIBULAR_Z_K, SAMPLE_FREQ )

        self._roll_filt  = IIR( a=_roll_a,   b=_roll_b,  order=3 )
        self._pitch_filt = IIR( a=_pitch_a,  b=_pitch_b, order=3 )
        self._yaw_filt   = IIR( a=_yaw_a,    b=_yaw_b,   order=3 )

        self._x_filt = IIR( a=_x_a, b=_x_b, order=2 )
        self._y_filt = IIR( a=_y_a, b=_y_b, order=2 )
        self._z_filt = IIR( a=_z_a, b=_z_b, order=2 )

    # ===============================================================================
    # @brief:   Update vesticular system
    #           
    #
    # @param[in]:    a      - Input accelerations
    # @param[in]:    w      - Input angular velocities
    # @return:       af, wf - Sensed accelerations and angular velocities
    # ===============================================================================
    def update(self, a, w):
        
        af_x = self._x_filt.update( a[0] ) 
        af_y = self._y_filt.update( a[1] ) 
        af_z = self._z_filt.update( a[2] ) 
        af = [af_x, af_y, af_z]

        wf_x = self._roll_filt.update( w[0] ) 
        wf_y = self._pitch_filt.update( w[1] ) 
        wf_z = self._yaw_filt.update( w[2] ) 
        wf = [wf_x, wf_y, wf_z]

        return af, wf

    # ===============================================================================
    # @brief:   Calculate rotation perception coefficients
    #           
    #   h(s) = (1/Ts * s^2) / ( s^3 + (1/Ta + 1/Tl + 1/Ts)*s^2 + (1/Tl*Ts + 1/Tl*Ta + 1/Ta*Ts)*s + 1/(Ta*Tl*Ts)))
    #
    # @param[in]:    Tl, Ts, Ta - Coefficient in the semicircular canals sensation model
    # @param[in]:    fs         - Sample frequency
    # @return:       b,a        - Array of b,a IIR coefficients
    # ===============================================================================
    def __calc_rot_mov_coefficient(self, Tl, Ts, Ta, fs):

        b, a = bilinear( [0, 1/Ts, 0, 0], [1, (1/Ta + 1/Tl + 1/Ts), (1/Tl*Ts + 1/Tl*Ta + 1/Ta*Ts), 1/(Ta*Tl*Ts)], fs )

        return b, a


    # ===============================================================================
    # @brief:   Calculate linear movement perception coefficients
    #           
    #   h(s) = ((K*Ta)*s + k) / ((Tl*Ts)*s^2 + (Tl+Ts)*s + 1)
    #
    # @param[in]:    Tl, Ts, Ta, K  - Coefficients in the otolith model
    # @param[in]:    fs             - Sample frequency
    # @return:       b,a            - Array of b,a IIR coefficients
    # ===============================================================================
    def __calc_lin_mov_coefficient(self, Tl, Ts, Ta, K, fs):

        b, a = bilinear( [0, K*Ta, K], [ Tl*Ts, Tl+Ts, 1], fs )

        return b, a


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

    _x_filt = IIR( a=_x_a, b=_x_b, order=2 )

    # Get frequency characteristics
    N = 256
    _roll_w, _roll_h    = freqz( _roll_b,  _roll_a,    4096 * N )
    #_pitch_w, _pitch_h  = freqz( _pitch_b, _pitch_a,   4096 * N )
    #_yaw_w, _yaw_h      = freqz( _yaw_b,   _yaw_a,     4096 * N )

    _x_w, _x_h = freqz( _x_b, _x_a, 4096 * N )
    #_y_w, _y_h = freqz( _y_b, _y_a, 4096 * N )
    #_z_w, _z_h = freqz( _z_b, _z_a, 4096 * N )


    # Vestibular system
    _vest_sys = VestibularSystem()



    # Filter input/output
    _x = [ 0 ] * SAMPLE_NUM
    _x_d = [0]

    
    _y_d_roll = [0]
    _y_d_x = [0]
    

    # Accelerations
    _y_d_a_sens = [[0], [0], [0]] * 3
    
    # Angular rates
    _y_d_w_sens = [[0], [0], [0]] * 3

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
        
        # Mux input signals
        #_x[n] = _signa_mux.out( INPUT_SIGNAL_SELECTION, [ _sin_x[n], _rect_x[n] ] )

        # Down sample to SAMPLE_FREQ
        if _downsamp_cnt >= (( 1 / ( _dt * SAMPLE_FREQ )) - 1 ):
            _downsamp_cnt = 0

            # Utils
            _downsamp_samp.append(0)
            _d_time.append( _time[n])
            _x_d.append( _x[n] )

            
            # Rotation sensed
            _y_d_roll.append( _roll_filt.update( _x[n] ))
            _y_d_x.append( _x_filt.update( _x[n] ))
            

            a_sens, w_sens = _vest_sys.update( [ _x[n], 0, 0 ], [ _x[n], 0, 0 ] )

            for n in range(3):
                _y_d_a_sens[n].append( a_sens[n] )
                _y_d_w_sens[n].append( w_sens[n] )
        else:
            _downsamp_cnt += 1
    

    # Calculate frequency response
    _roll_w     = ( _roll_w / np.pi * SAMPLE_FREQ / 2)  # Hz
    #_pitch_w    = ( _pitch_w / np.pi * SAMPLE_FREQ / 2) # Hz
    #_yaw_w      = ( _yaw_w / np.pi * SAMPLE_FREQ / 2)   # Hz

    _x_w = ( _x_w / np.pi * SAMPLE_FREQ / 2)  # Hz
    #_y_w = ( _y_w / np.pi * SAMPLE_FREQ / 2)  # Hz
    #_z_w = ( _z_w / np.pi * SAMPLE_FREQ / 2)  # Hz

    # For conversion to rad/s
    _roll_w = 2*np.pi*_roll_w  
    #_pitch_w = 2*np.pi*_pitch_w  
    #_yaw_w = 2*np.pi*_yaw_w  

    _x_w = 2*np.pi*_x_w  
    #_y_w = 2*np.pi*_y_w  
    #_z_w = 2*np.pi*_z_w  

    # Calculate phases & convert to degrees
    _roll_angle     = np.unwrap( np.angle(_roll_h) )    * 180/np.pi
    #_pitch_angle    = np.unwrap( np.angle(_pitch_h) )   * 180/np.pi
    #_yaw_angle      = np.unwrap( np.angle(_yaw_h) )     * 180/np.pi

    _x_angle = np.unwrap( np.angle(_x_h) )     * 180/np.pi
    #_y_angle = np.unwrap( np.angle(_y_h) )     * 180/np.pi
    #_z_angle = np.unwrap( np.angle(_z_h) )     * 180/np.pi

    

    plt.style.use(['dark_background'])
    ## ==============================================================================================
    # Rotation motion plots
    ## ==============================================================================================
    fig, ax = plt.subplots(2, 1)
    fig.suptitle( "ROTATION MOVEMENT MODEL\n fs: " + str(SAMPLE_FREQ) + "Hz", fontsize=16 )

    
    ax[0].plot(_roll_w, 20 * np.log10(abs(_roll_h)), "w")

    ax[0].grid(alpha=0.25)
    ax[0].set_xscale("log")
    ax[0].set_xlim(1e-3, SAMPLE_FREQ/2)
    ax[0].set_ylim(-80, 2)
    ax[0].set_ylabel("Magnitude [dB]", color="w", fontsize=14)
    ax[0].set_xlabel("Frequency [rad/s]", fontsize=14)

    ax_00 = ax[0].twinx()

    ax_00.plot(_roll_w, _roll_angle, "r")
    ax_00.set_ylabel("Phase [deg]", color="r", fontsize=14)
    ax_00.set_xscale("log")
    ax_00.grid(alpha=0.25)
    ax_00.set_xlim(1e-3, SAMPLE_FREQ/2)
    ax_00.set_xlabel("Frequency [rad/s]", fontsize=14)
    

    ax[1].plot( _time, _x,                  "r",    label="Input" )
    ax[1].plot( _d_time, _y_d_roll,         ".-y",    label="Sensed")
    ax[1].plot( _d_time, _y_d_w_sens[0],    "--w",    label="Sensed")
    ax[1].grid(alpha=0.25)
    ax[1].set_xlim(0, 8)
    ax[1].legend(loc="upper right")
    ax[1].grid(alpha=0.25)
    ax[1].set_xlabel("Time [s]", fontsize=14)
    ax[1].set_ylabel("rotation, sensed rotation [rad/s]", fontsize=14)

    ## ==============================================================================================
    # Linear motion plots
    ## ==============================================================================================
    fig, ax = plt.subplots(2, 1)
    fig.suptitle( "LINEAR MOVEMENT MODEL\n fs: " + str(SAMPLE_FREQ) + "Hz", fontsize=16 )

    
    ax[0].plot(_x_w, 20 * np.log10(abs(_x_h)), 'w')
    ax[0].grid(alpha=0.25)
    ax[0].set_xscale("log")
    ax[0].set_xlim(1e-3, SAMPLE_FREQ/2)
    ax[0].set_ylim(-40, 1)
    ax[0].set_ylabel("Magnitude [dB]", color="w", fontsize=14)
    ax[0].set_xlabel("Frequency [rad/s]", fontsize=14)

    ax_00 = ax[0].twinx()

    ax_00.plot(_x_w, _x_angle, "r")

    ax_00.set_ylabel("Phase [deg]", color="r", fontsize=14)
    ax_00.set_xscale("log")
    ax_00.grid(alpha=0.25)
    ax_00.set_xlim(1e-3, SAMPLE_FREQ/2)
    ax_00.set_xlabel("Frequency [rad/s]")
    


    ax[1].plot( _time, _x,                  "r",    label="Input" )
    ax[1].plot( _d_time, _y_d_x,            ".-y",    label="Sensed")
    ax[1].plot( _d_time, _y_d_a_sens[0],    "--w",    label="Sensed")
    ax[1].grid(alpha=0.25)
    ax[1].set_xlim(0, 8)
    ax[1].legend(loc="upper right")
    ax[1].grid(alpha=0.25)
    ax[1].set_xlabel("Time [s]", fontsize=14)
    ax[1].set_ylabel("acceleration, sensed force [mm^2]", fontsize=14)

    plt.show()
    


# ===============================================================================
#       END OF FILE
# ===============================================================================
