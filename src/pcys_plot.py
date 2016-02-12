"""
Plot PCYS and STR-STR curves resulting from RGVB simulations
"""

import numpy as np
from glob import glob
import matplotlib.pyplot as plt
sqrt=np.sqrt

def _plot_(fn,ax1,ax2,ax3,norm):
    """
    """
    dat=np.loadtxt(fn,skiprows=1).T
    s1=dat[0]
    s2=dat[1]

    """
    s22-s11 = s1*sqrt(2)
    -s22-s11 = s2*sqrt(6)
    """
    s11 = (-s1*sqrt(2)-s2*sqrt(6))/2.
    s22 = s1*sqrt(2) + s11

    ## Project on to the plane-stress space
    if ax1==None or ax2==None:
        fig=plt.figure(figsize=(9,3))
        ax1=fig.add_subplot(131)
        ax2=fig.add_subplot(132)
        ax3=fig.add_subplot(133)
        ax1.set_aspect('equal')
        ax1.grid('on')
        ax2.set_aspect('equal')
        ax2.grid('on')
        ax3.set_aspect('equal')
        ax3.grid('on')

    l,=ax1.plot(s11/norm,s22/norm,'-x')
    ax2.plot(s1/norm,s2/norm,'-x',color=l.get_color())
    ax3.plot(s1,s2,'-x',color=l.get_color())
    return ax1,ax2,ax3

def read(fn):
    """
    """
    s=open(fn,'r').read()
    lines=s.split('\n')
    D=[]
    ib = []
    nb = 0
    for i in xrange(len(lines)):
        l = lines[i]
        try:
            dat=map(float,l.split())
            dat[0]
        except:
            if nb!=0: ib.append(nb_ln)
            nb = nb + 1
            nb_ln = 0
            pass
        else:
            nb_ln = nb_ln + 1
            D.append([dat[0],dat[1],
                      dat[2],dat[3],
                      dat[4],dat[5],
                      dat[6],dat[7]])

    nb = nb-1
    # print ib
    # print nb

    D = np.array(D).T
    dat=np.zeros((nb,8,max(ib)))
    # print 'D.shape:',D.shape
    # print 'd.shape:', dat.shape

    dat[::] = np.nan
    i0 = 0
    for i in xrange(nb):
        i1 = i0+ib[i]
        n = i1-i0
        # print 'i,i0,i1,n'
        # print i, i0, i1, n
        # raw_input()
        # print D[0][i0:i1]
        # raw_input()
        #dat[i,0,0:n] = D[0][i0:i1]
        #dat[i,1,0:n] = D[1][i0:i1]

        for j in xrange(8):
            dat[i,j,0:n] = D[j][i0:i1]

        i0 = i0+ib[i]

    return dat

def main():
    """
    Main function for this module.
    Load up data from STR_STR.OUT and PCYS.OUT
    to dynamically allocated arrays
    """
    from matplotlib.ticker import MaxNLocator
    from matplotlib import gridspec
    GS = gridspec.GridSpec
    gs = GS(10,50,top=0.95,bottom=0.05,left=0.1,wspace=1,hspace=0)

    fig=plt.figure(figsize=(16,3))

    fig.add_subplot(gs[1:9, 0:7])
    fig.add_subplot(gs[1:9,10:17])
    fig.add_subplot(gs[1:9,20:27])
    fig.add_subplot(gs[1:9,30:37])
    fig.add_subplot(gs[1:9,40:47])

    for ax in fig.axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.minorticks_on()

    ax1,ax2,ax3,ax4,ax5=fig.axes

    # ax1=fig.add_subplot(151)
    # ax2=fig.add_subplot(152)
    # ax3=fig.add_subplot(153)
    # ax4=fig.add_subplot(154)
    # ax5=fig.add_subplot(155)

    # ax1.set_aspect('equal')
    ax1.grid('on')
    # ax2.set_aspect('equal')
    ax2.grid('on')
    ax3.grid('on')

    D=read(fn='PCYS.OUT')
    S=read(fn='STR_STR.OUT')
    print D.shape, S.shape
    nlev=len(D)
    for i in xrange(nlev):
        norm = S[i,1,:]
        flt = np.isnan(norm)==False
        norm=norm[flt]
        norm = norm[-1]
        print norm
        # norm = 1.
        # print norm
        x=D[i,0,:]
        y=D[i,1,:]

        l, = ax3.plot(x/norm,y/norm,'-')
        if i==0: ax3.set_title(r'YS/$\sigma^\mathrm{VM}$')
        ax2.plot(x,y,'-',color=l.get_color())
        if i==0: ax2.set_title(r'YS')

        ax1.plot(S[i][0,:][flt],S[i][1,:][flt],'-',color=l.get_color())
        ax1.plot(S[i][0,:][0],S[i][1,:][0],'+',color=l.get_color())
        #ax1.plot(S[i][0,:][1],S[i][1,:][1],'x',color=l.get_color())
        ax1.plot(S[i][0,:][flt][-1],S[i][1,:][flt][-1],'o',color=l.get_color())

        if i==0: ax1.set_title(r'Flow stress-strain $\sigma^{VM} vs. \epsilon^{VM}$')

        s1=x; s2=y
        s11 = (-s1*sqrt(2)-s2*sqrt(6))/2.
        s22 = s1*sqrt(2) + s11
        ax4.plot(s11,s22,color=l.get_color())
        ax5.plot(s11/norm,s22/norm,color=l.get_color())


        if i==0: ax4.set_title(r'YS plane stress')
        if i==0: ax5.set_title(r'Normalized YS')

    ax1.set_ylim(0,)
    ax2.set_xlim(-400,400)
    ax2.set_ylim(-400,400)

    ax3.set_xlim(-0.8,0.8)
    ax3.set_ylim(-0.8,0.8)

    ax4.set_xlim(-600,600)
    ax4.set_ylim(-600,600)

    ax5.set_xlim(-1.2,1.2)
    ax5.set_ylim(-1.2,1.2)
