import numpy as np
r2 = np.sqrt(2.)
r6 = np.sqrt(6.)
def main(fn='PCYS.OUT'):
    dat = np.loadtxt(fn,skiprows=1).T
    S1, S2 = dat[0],dat[1]
    D1, D2 = dat[2],dat[3]
    s11 = np.zeros((len(S1)+1,))
    s22 = s11.copy()
    d11 = s11.copy()
    d22 = s11.copy()
    ## find s11 and s22
    s1 = - (S1 * r2 + S2 * r6)/2.
    s2 =   (S1 * r2 - S2 * r6)/2.

    d1 = -r6/3. * D2 - r2 * D1
    d2 = -r6/3. * D2 + r2 * D1
    d1 = d1 / 2.
    d2 = d2 / 2.


    for i in xrange(len(s1)):
        s11[i] = s1[i]
        s22[i] = s2[i]
        d11[i] = d1[i]
        d22[i] = d2[i]

    s11[-1] = s1[0]
    s22[-1] = s2[0]
    d11[-1] = d11[0]
    d22[-1] = d22[0]

    return s11,s22,d11,d22


def int_stress_at_th(s11,s22,d11,d22,th):
    ths = []
    for i in xrange(len(s11)):
        _th_ = np.arctan2(d22[i],d11[i])
        ths.append(_th_)

    i0 = find_nearest_ind(ths, th)

    v1=abs(th - ths[i0+1])
    v2=abs(th - ths[i0-1])
    if v1<v2: i1 = i0+1
    if v1>v2: i1 = i0-1


    ## interpolate...
    # xp = [ths[i0], ths[i1]]
    # y1 = [s11[i0], s11[i1]]
    # y2 = [s22[i0], s22[i1]]
    # s1=np.interp(th,xp, y1)
    # s2=np.interp(th,xp, y2)

    s1 = (s11[i0]+s11[i1]) / 2.
    s2 = (s22[i0]+s22[i1]) / 2.

    # d1 = (d11[i0]+d11[i1]) / 2.
    # d2 = (d22[i0]+d22[i1]) / 2.

    return s1,s2



def find_nearest_ind(array,value):
    array = np. array(array)
    index = (np.abs(array-value)).argmin()
    return index
