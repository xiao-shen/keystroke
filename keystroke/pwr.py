# audio power calculus functions

def pwr_log(x, len):
    y=20.0*np.log10(abs(x)+1)
    return sum(y)

def pwr_2(x, len):
    # x_sqr=pow(abs(x),2)
    # return np.mean(x_sqr)
    pw=0.0
    for i in xrange(len):
        pw=pw + float(x[i])**2
    return pw / len

def pwr_abs(x, len):
    return np.sum(np.abs(x)) / len
