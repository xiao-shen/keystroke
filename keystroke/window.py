import numpy as np
import math
import matplotlib.pyplot as plt

class Windowing:
    """
    The window function splits a possibly infinitely long sample stream in 
    chunks of data. The length of a chunk is windowSize.
    """

    def __init__(self, windowSize=4096, stepSize = 2048, windowFunc = np.hamming):
        """
        Create a new instance with the given windowSize and stepSize. 
        @param windowSize: specifies the size of the window to generate
        @param stepSize: specifies the step size of this windowing function. Each iteration
        the window is moved by stepSize steps. The stepSize could be used let the 
        windows overlap.
        @param windowFunc: The windowing function to generate this window. 
        It accepts the size of the window as first parameter.
        """
        self.windowSize = windowSize
        self.stepSize = stepSize
        self.windowFunc =  windowFunc(self.windowSize)
    
    def apply(self,y_in):
        return self.applybis(y_in)
        print "begin windowing"
        """
        Apply the windowing function to the array y_in. 
        
        @return: A new array with the content of the window in the x direction
        and all windows in the y direction.
        
        @example:
        Assume an y_in of [1111222233334444] and a window size of 4 and a step size of 4
        Then the output will be 
        [1111,
         2222,
         3333,
         4444]
        """
        numberOfSteps = int(math.floor((len(y_in) - self.windowSize)/self.stepSize))
        
        #print "Number of steps: %f" % numberOfSteps
        
        out = np.ndarray((numberOfSteps,self.windowSize),dtype = y_in.dtype)
        
        for i in xrange(numberOfSteps):
            offset = i * self.stepSize
            out[i] = y_in[offset : offset + self.windowSize] #* self.windowFunc
        print " end windowing"
        return out

    def applybis(self,y_in):
        """
        This windowing function seeks first for the beginning of a keystroke and then returns a list of fixed-sized windows in a matrix
        """
        print "begin detection"
        Fs = float(44100)
        
        peakWindowSize = int(0.050 * Fs) # the common peak size is 30 ms
        peakStepSize = int(0.060 * Fs)   # the iteration step is larger than the peak analysis window
        """
        The power threshold determines where a peak begins
        """
        pwr_threshold = 100000 # dB.s

        keystrokeWindowSize = int(0.300 * Fs) # = 13230 = 6615*2
        keystrokeMargin = int(0.035 * Fs)

        maxWindows = int(math.floor((len(y_in) - keystrokeWindowSize)/keystrokeWindowSize)) # the maximum number of samples which can be extracted
        out = np.ndarray((maxWindows,keystrokeWindowSize),dtype = y_in.dtype)

        offset = keystrokeMargin # initialize the time origin
        count = 0
        end_offset = len(y_in) - keystrokeWindowSize
        while offset < end_offset:
            peak = y_in[offset : offset + peakWindowSize]
	    # no pre-implemented function to calculate power?
            peak_pwr=pwr_log(peak, peakWindowSize)
            
            if peak_pwr > pwr_threshold:
                # we suppose we are on the beginning of a keystroke
                offset = offset - keystrokeMargin
                sample_time = float(offset) / Fs
                sample = y_in[offset : offset + keystrokeWindowSize]
                sample_x = xrange(offset, offset + keystrokeWindowSize)
                plt.plot(sample_x, sample)
                out[count] = sample
                count = count + 1
                offset = offset + keystrokeWindowSize + keystrokeMargin + peakStepSize * 2
            else:
                offset = offset + peakStepSize

        print "%d keystrokes detected" % count
        plt.show()
	# delete zero rows
        np.delete(out, xrange(count, maxWindows))
        print " end detection"
        return out

def pwr_log(x, len):
    y=20.0*np.log10(abs(x)+1)
    return sum(y)

def pwr(x, len):
    # x_sqr=pow(abs(x),2)
    # return np.mean(x_sqr)
    pw=0.0
    for i in xrange(len):
        pw=pw + float(x[i])**2
    return pw / len

def pwr_abs(x, len):
    return np.sum(np.abs(x)) / len
