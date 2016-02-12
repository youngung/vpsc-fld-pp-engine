## test performance

import numpy as np
import matplotlib.pyplot as plt

def reader(fn='20151001_T6Uwru_FLD_B_pp_time_score.txt'):
    lines = open(fn).read().split('\n')
    lines=lines[:-1]

    ## check parsings...
    # print lines[-9] ## Real-Time-Monitor in use?
    # print lines[-8] ## Numerical method (NR or DA)
    # print lines[-1] ## efficiency
    # print lines[-2] ## Speed-up
    # print lines[-3] ## NCPUS
    # print lines[-4] ## WallClock
    # print lines[-5] ## CPU time

    rtm     = lines[-9].split(':')[-1]
    rtm     = rtm.strip()

    nu      = int(lines[-8].split(':')[-1])
    eff     = float(lines[-1].split(':')[-1].split()[0])
    speedup = float(lines[-2].split(':')[-1])

    wall_clock = float(lines[-4].split(',')[-1])
    cpu_run    = float(lines[-5].split(',')[-1])
    ncpus   = int(lines[-3].split(':')[-1])


    nconds = int(lines[-12].split(':')[-1])
    nconds = nconds * int(lines[-12].split(':')[-1])

    return rtm,nu,eff,speedup,ncpus,wall_clock,cpu_run


def main(date=20151028,hash_a='*'):
    import os
    from glob import glob
    fns = glob('archive/FLD_A/*/FLD_%s/%i_*time_score.txt'%(hash_a,date))
    #fns = glob('%i_*time_score.txt'%date)
    # print fns
    # print 'Number of collected data:',len(fns)
    Dat_sort_rtm = [[],[]]
    Dat_sort_none = [[],[]]
    Dat_sort_nu0 = [[],[]]
    Dat_sort_nu1 = [[],[]]
    fig=fig=plt.figure(figsize=(6,6))
    ax1=fig.add_subplot(221)
    ax3=fig.add_subplot(223)
    ax4=fig.add_subplot(224)

    k=1
    fns.sort()

    #fns[i], rtm, icase, 'cx', label, ncpu
    print ('%6s '*6)%('fns','rtm','icase','CX','label','ncpu')
    fmt = '%6s %6s %6i %6s %6s %6i'

    for i in xrange(len(fns)):
        try:
            rtm, nu, eff, speedup, ncpu, wall_clock, cpu_run = reader(fns[i])
        except:
            print 'Failed to read'
            pass
        else:

            if rtm!='True' and nu==0:
                label='N/R'
                cx='b'
                icase = 0
            if rtm!='True' and nu==1:
                label='D/A'
                cx='y'
                icase = 2
            if rtm=='True' and nu==0:
                label='N/R+M'
                cx='r'
                icase = 1
            if rtm=='True' and nu==1:
                label='D/A+M'
                cx='g'
                icase = 3

            print fmt%(os.path.split(fns[i])[-1][9:15], rtm, icase, cx, label, ncpu)

            if nu==0: marker='x'
            if nu==1: marker='s'

            lab=None
            if ncpu==64:
                lab=label
                ax3.bar(icase+1,wall_clock/60.,align='center',color=cx)
                ax4.bar(icase+1,cpu_run/3600.,align='center',color=cx)
                k = k + 1

            ax1.plot(ncpu,speedup,mec=cx,mfc='None',marker=marker,label=lab,ls='None')

    ax1.plot([0,70],[0,70],'--',color='gray')
    ax1.set_xlabel('Number of CPU core unit')
    ax1.set_ylabel('Speed-up by parallelization')
    ax1.legend(loc='best',numpoints=1,fontsize=7)

    ax3.set_ylabel('Wall Clock time [min]')
    ax4.set_ylabel('CPU run time [Hour]')
    for ax in [ax3,ax4]:
        ax.set_xlim(0,5)
        ax.set_xticks([1,2,3,4])
        pass

    fig.tight_layout()#bbox_inches='tight')
