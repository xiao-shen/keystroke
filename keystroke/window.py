import numpy as np
import math

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
         
        return out