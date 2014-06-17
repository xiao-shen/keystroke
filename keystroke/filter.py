import numpy as np

class ThresholdFilter:
    """
    This filter removes windows which are silent.
    The window is called silent when its average amplitude
    is below a certain threshold.
    """
    def __init__(self, threshold = 100):
        self.threshold = threshold
        
    def apply(self, y_in):
        """
        This function applys the thresholding to the input.
        @param y_in: A matrix with the sampled data. The maxtrix
        has the following shape: 
            - In the x direction are the raw samples of the winodw
            - In the y direction are the windows of the audio to analyze
        @returns: the Matrix without silent windows
        """
    
        x_len, y_len = y_in.shape
        
        linesToRemove = []
        
        for x in xrange(x_len):
            integral = np.sum(np.abs(y_in[x])) / y_len
            
            if integral < self.threshold:
                linesToRemove.append(x)
        
        return np.delete(y_in, linesToRemove, 0)
