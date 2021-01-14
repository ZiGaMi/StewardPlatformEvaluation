# ===============================================================================
# @file:    filter_utils.py
# @note:    This script is has various utility classes/function
# @author:  Ziga Miklosic
# @date:    10.01.2021
# @brief:   Classes/Functions for general usage to evaluate filter
# ===============================================================================

# ===============================================================================
#       IMPORTS  
# ===============================================================================
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ===============================================================================
#       CONSTANTS
# ===============================================================================

# ===============================================================================
#       CLASSES
# ===============================================================================

# Signal fucntion generator
class FunctionGenerator:

    FG_KIND_SINE = "sine"
    FG_KIND_RECT = "rect"
    FG_KIND_TRIANGLE = "triangle"

    # ===============================================================================
    # @brief: Initialize function generator object
    #
    # @param[in]:    freq    - Frequency of signal  
    # @param[in]:    amp     - Amplitude of signal
    # @param[in]:    off     - DC offset of signal
    # @param[in]:    phase   - Phase of signal
    # @return:       Generated signal
    # ===============================================================================
    def __init__(self, freq, amp, off, phase, kind):

        self.freq = freq
        self.amp = amp
        self.off = off
        self.phase = phase
        self.kind = kind

    # ===============================================================================
    # @brief: Generate signal selected at init time
    #
    # @param[in]:    time    - Linear time  
    # @return:       Generated signal
    # ===============================================================================
    def generate(self, time):
        
        _sig = 0

        if self.kind == "sine":
            _sig =  self.__generate_sine(time, self.freq, self.amp, self.off, self.phase)
        elif self.kind == "rect":
            _sig = self.__generate_rect(time, self.freq, self.amp, self.off, self.phase)
        elif self.kind == "triangle":
            _sig = self.__generate_tringle(time, self.freq, self.amp, self.off, self.phase)
        else:
            raise AssertionError

        return _sig

    # ===============================================================================
    # @brief: Generate sine signal
    #
    # @param[in]:    time    - Linear time  
    # @param[in]:    amp     - Amplitude of sine
    # @param[in]:    off     - DC offset of sine
    # @param[in]:    phase   - Phase of sine
    # @return:       Generated signal
    # ===============================================================================
    def __generate_sine(self, time, freq, amp, off, phase):
        _sig = (( amp * np.sin((2*np.pi*freq*time) + phase )) + off )
        return _sig

    # ===============================================================================
    # @brief: Generate rectangle signal
    #
    # @param[in]:    time    - Linear time  
    # @param[in]:    amp     - Amplitude of rectange
    # @param[in]:    off     - DC offset of rectangle
    # @param[in]:    phase   - Phase of rectangle
    # @return:       Generated signal
    # ===============================================================================
    def __generate_rect(self, time, freq, amp, off, phase):
        _sig = ( amp * signal.square((2*np.pi*freq*time + phase), duty=0.5)) 
        return _sig

    # ===============================================================================
    # @brief: Generate triangle signal
    #
    # @param[in]:    time    - Linear time  
    # @param[in]:    amp     - Amplitude of rectange
    # @param[in]:    off     - DC offset of rectangle
    # @param[in]:    phase   - Phase of rectangle
    # @return:       Generated signal
    # ===============================================================================
    def __generate_tringle(self, time, freq, amp, off, phase):
        _sig = ( amp * signal.sawtooth((2*np.pi*freq*time + phase), width=0.5))
        return _sig
        

## Circular buffer
class CircBuffer:

    def __init__(self, size):
        self.buf = [0.0] * size
        self.idx = 0
        self.size = size
    
    def __manage_index(self):
        if self.idx >= (self.size - 1):
            self.idx = 0
        else:
            self.idx = self.idx + 1

    def set(self, val):
        if self.idx < self.size:
            self.buf[self.idx] = val
            self.__manage_index()
        else:
            raise AssertionError

    def get(self, idx):
        if idx < self.size:
            return self.buf[idx]
        else:
            raise AssertionError
    
    def get_tail(self):
        return self.idx

    def get_size(self):
        return self.size

    def get_whole_buffer(self):
        return self.buf

    """
        Returns array of time sorted samples in buffer
        [ n, n-1, n-2, ... n - size - 1 ]
    """
    def get_time_ordered_samples(self):

        _ordered = [0.0] * self.size
        _start_idx = 0

        _start_idx = self.idx - 1
        if _start_idx < 0:
            _start_idx += self.size

        # Sort samples per time
        for n in range( self.size ):
            _last_idx = _start_idx - n
            if _last_idx < 0:
                _last_idx += self.size 
            _ordered[n] = self.buf[_last_idx]

        return _ordered


# ===============================================================================
#       END OF FILE
# ===============================================================================

## Time window
#
# Unit: second
TIME_WINDOW = 2.5

## Input signal shape
INPUT_SIGNAL_FREQ = 1.0
INPUT_SIGNAL_AMPLITUDE = 1.0
INPUT_SIGNAL_OFFSET = 1.0
INPUT_SIGNAL_PHASE = 0.0

# Ideal sample frequency
#   As a reference to sample rate constrained embedded system
#
# Unit: Hz
IDEAL_SAMPLE_FREQ = 10000.0

## Number of samples in time window
SAMPLE_NUM = int(( IDEAL_SAMPLE_FREQ * TIME_WINDOW ) + 1.0 )


## Only for testing
if __name__ == "__main__":

    # Time array
    _time, _dt = np.linspace( 0.0, TIME_WINDOW, num=SAMPLE_NUM, retstep=True )



    # Generate inputs
    _fg_sine        = FunctionGenerator( INPUT_SIGNAL_FREQ, INPUT_SIGNAL_AMPLITUDE, INPUT_SIGNAL_OFFSET, INPUT_SIGNAL_PHASE, "sine" )
    _fg_rect        = FunctionGenerator( INPUT_SIGNAL_FREQ, INPUT_SIGNAL_AMPLITUDE, INPUT_SIGNAL_OFFSET, INPUT_SIGNAL_PHASE, "rect" )
    _fg_triangle    = FunctionGenerator( INPUT_SIGNAL_FREQ, INPUT_SIGNAL_AMPLITUDE, INPUT_SIGNAL_OFFSET, INPUT_SIGNAL_PHASE, "triangle" )
    _sin_x = []
    _rect_x = []
    _triangle_x = []


    # Generate stimuli signals
    for n in range(SAMPLE_NUM):
        _sin_x.append( _fg_sine.generate( _time[n] ))
        _rect_x.append( _fg_rect.generate( _time[n] ))
        _triangle_x.append( _fg_triangle.generate( _time[n] ))


    plt.style.use(['dark_background'])
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(_time,     _sin_x,    'y',lw=2)
    ax[0].grid(alpha=0.25)

    ax[1].plot(_time,     _rect_x,    'r', lw=2)
    ax[1].plot(_time,     _triangle_x, 'b', lw=2)
    ax[1].grid(alpha=0.25)


    plt.show()

