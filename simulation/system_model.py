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
IDEAL_SAMPLE_FREQ = 20000.0

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
# WASHOUT FILTER COEFFICINETS
# =====================================================

# =====================================================
## TRANSLATION CHANNEL SETTINGS

# HPF Wht 2nd order filter
WASHOUT_HPF_WHT_FC  = 1.0
WASHOUT_HPF_WHT_Z   = .7071

# HPF Wrtzt 1st order filter
WASHOUT_HPF_WRTZT_FC  = 1.0

# =====================================================
## COORDINATION CHANNEL SETTINGS

# LPF W12 2nd order filter
WASHOUT_LPF_W12_FC  = 1.0
WASHOUT_LPF_W12_Z   = 1.0

# =====================================================
## ROTATION CHANNEL SETTINGS

# HPF W11 1st order filter
WASHOUT_HPF_W11_FC  = 1.0



## ****** END OF USER CONFIGURATIONS ******

# ===============================================================================
#       FUNCTIONS
# ===============================================================================

# ===============================================================================
#       CLASSES
# ===============================================================================    


# ===============================================================================
#       MAIN ENTRY
# ===============================================================================
if __name__ == "__main__":

    # Time array
    _time, _dt = np.linspace( 0.0, TIME_WINDOW, num=SAMPLE_NUM, retstep=True )

    
    # Filter object
    _filter_washout = WashoutFilter(  Wht=[WASHOUT_HPF_WHT_FC, WASHOUT_HPF_WHT_Z], Wrtzt=WASHOUT_HPF_WRTZT_FC, \
                                W11=WASHOUT_HPF_W11_FC, W12=[WASHOUT_LPF_W12_FC, WASHOUT_LPF_W12_Z], fs=SAMPLE_FREQ )

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
        _x[n] = ( _fg.generate( _time[n] ))
 
    # Apply filter
    for n in range(SAMPLE_NUM):

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
