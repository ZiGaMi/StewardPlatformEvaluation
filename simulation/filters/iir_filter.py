# ===============================================================================
# @file:    iir_filter.py
# @note:    This script is evaluation of IIR filter algorithm
# @author:  Ziga Miklosic
# @date:    06.01.2021
# @brief:   Evaluation of IIR filter design. 
# ===============================================================================

# ===============================================================================
#       IMPORTS  
# ===============================================================================
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz, butter, cheby1, lfilter, filtfilt, bilinear

from filters.filter_utils import CircBuffer

# ===============================================================================
#       CONSTANTS
# ===============================================================================

# ===============================================================================
#       FUNCTIONS
# ===============================================================================

# ===============================================================================
# @brief:   calculate 2nd order high pass filter based on following 
#           transfer function:
#           
#               h(s) = s^2 / ( s^2 + 2*z*w*s + w^2 )  --- bilinear ---> h(z)
#
# @param[in]:    fc     - Corner frequenc
# @param[in]:    z      - Damping factor
# @param[in]:    fs     - Sample frequency
# @return:       b,a    - Array of b,a IIR coefficients
# ===============================================================================
def calculate_2nd_order_HPF_coeff(fc, z, fs):
    
    # Calculate omega
    w = 2*np.pi*fc

    # Make bilinear transformation
    b, a = bilinear( [1,0,0], [1,2*z*w,w**2], fs )

    return b, a


# ===============================================================================
# @brief:   calculate 1nd order high pass filter based on following 
#           transfer function:
#           
#               h(s) = s / ( s + w )  --- bilinear ---> h(z)
#
# @param[in]:    fc     - Corner frequenc
# @param[in]:    fs     - Sample frequency
# @return:       b,a    - Array of b,a IIR coefficients
# ===============================================================================
def calculate_1nd_order_HPF_coeff(fc, fs):
    
    # Calculate omega
    w = 2*np.pi*fc

    # Make bilinear transformation
    b, a = bilinear( [1,0], [1,w], fs )

    return b, a


# ===============================================================================
# @brief:   calculate 2nd order low pass filter based on following 
#           transfer function:
#           
#               h(s) = w^2 / ( s^2 + 2*z*w*s + w^2 ) --- bilinear ---> h(z)
#
# @param[in]:    fc     - Corner frequenc
# @param[in]:    z      - Damping factor
# @param[in]:    fs     - Sample frequency
# @return:       b,a    - Array of b,a IIR coefficients
# ===============================================================================
def calculate_2nd_order_LPF_coeff(fc, z, fs):

    # Calculate omega
    w = 2*np.pi*fc

    # Using bilinear transformation
    b, a = bilinear( [0,0,w**2], [1,2*z*w,w**2], fs )

    return b, a


# ===============================================================================
# @brief:   calculate 2nd order notch filter. This code is from 
#           "Second-order IIR Notch Filter Design and implementation of digital
#           signal processing system" acticle
#
# @param[in]:    fc     - Corner frequenc
# @param[in]:    fs     - Sample frequency
# @return:       b,a    - Array of b,a IIR coefficients
# ===============================================================================
def calculate_2nd_order_notch_coeff(fc, fs, r):
   
    _w = 2 * np.pi * fc / fs

    # Calculate coefficient
    a2 = r*r
    a1 = -2*r*np.cos( _w ) 
    a0 = 1
    
    b2 = 1
    b1 = -2*np.cos( _w )
    b0 = 1
    
    # Fill array
    a = [ a0, a1, a2 ] 
    b = [ b0, b1, b2 ] 

    return b, a

# ===============================================================================
#       CLASSES
# ===============================================================================    

## IIR Filter
class IIR:

    def __init__(self, a, b, order):

        # Store tap number and coefficient
        self.order = order
        self.a = a
        self.b = b

        # Create circular buffer
        self.x = CircBuffer(order+1)
        self.y = CircBuffer(order+1)


    def update(self, x):
        
        # Fill input
        self.x.set( x )

        # Get input/outputs history
        _x = self.x.get_time_ordered_samples()
        _y = self.y.get_time_ordered_samples()

        # Calculate new value
        y = 0.0
        for j in range(self.order+1):

            y = y + (self.b[j] * _x[j])

            if j > 0:
                y = y - (self.a[j] * _y[j-1])

        y = ( y * ( 1 / self.a[0] ))

        # Fill output
        self.y.set(y)

        return y

# ===============================================================================
#       END OF FILE
# ===============================================================================
