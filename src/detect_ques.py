from os import popen
import numpy as np
import time
from MP import progress_bar as pb
uet = pb.update_elapsed_time

def find_availability(njobs=19,allow_fq=True,min_cpu=0):
    """
    Find available resources based on qstat in
    NIST CTCMS cluster.

    Arguments
    =========
    njobs = 19
    allow_fq = False  (allow the fast que or not)
    min_cpu  = 1  (a que should have more than min_cpu
                   of available cores)
    """
    f = popen('qstat -f');
    lines=f.read();lines=lines.split('\n')
    n = 0
    cpus = 0
    unit = []
    que_kinds = []
    for i in xrange(len(lines)):
        datum = lines[i].split()
        ## operativity?
        iop=True

        try:
            if datum[-1]=='adu':
                iop=False
            if datum[-1]=='d':
                iop=False
            if datum[-1]=='au':
                iop=False
            if datum[-1]=='a':
                iop=False
            if datum[-1]=='dE':
                iop=False
        except: pass
        # print 'operativity: ',iop

        ##
        if iop and len(datum)>0 and datum[0].split('@')[0]=='wide64':
            dum, i1,i2=map(int, lines[i].split()[2].split('/'))
            ncore = i2-i1

            if ncore>=min_cpu:
                n = n + 1
                unit.append(i2-i1)
                cpus = cpus + i2-i1
                que_kinds.append('wide64')

        if iop and len(datum)>0 and datum[0].split('@')[0]=='wide2':
            dum, i1,i2=map(int, lines[i].split()[2].split('/'))
            ncore = i2-i1
            if ncore>=min_cpu:
                n = n + 1
                unit.append(i2-i1)
                cpus = cpus + i2-i1
                que_kinds.append('wide2')
        if iop and len(datum)>0 and datum[0].split('@')[0]=='wide3':
            dum, i1,i2=map(int, lines[i].split()[2].split('/'))
            ncore = i2-i1
            if ncore>=min_cpu:
                n = n + 1
                unit.append(i2-i1)
                cpus = cpus + i2-i1
                que_kinds.append('wide3')

        if iop and len(datum)>0 and allow_fq and datum[0].split('@')[0]=='fast':
            dum, i1,i2=map(int, lines[i].split()[2].split('/'))
            ncore = i2-i1
            if ncore>=min_cpu:
                n = n + 1
                unit.append(i2-i1)
                cpus = cpus + i2-i1
                que_kinds.append('fast')

    if n==0:
        return 0, [0], ['None']

    ## sorting
    if len(unit)>0:
        print unit, que_kinds
        unit, que_kinds = zip(*sorted(zip(unit,que_kinds)))
        unit = unit[::-1]
        que_kinds=que_kinds[::-1]

    print '%5s %5s'%('Ques', 'cpus')
    print '%5i %5i'%(n, cpus)

    # for i in xrange(len(unit)):
    #     print '%5i '%unit[i],
    # print
    # for i in xrange(len(unit)):
    #     print '%5.2f '%(unit[i]/float(cpus)),
    # print

    ntot = 0
    remaining = njobs
    nassign = np.zeros(len(unit))
    for i in xrange(len(unit)):
        n = int(njobs * unit[i]/float(cpus))
        remaining = remaining - n
        #print '%5i '%n,
        ntot = ntot + n
        nassign[i] = n
    #print 'sum:', ntot

    ## distribute the remaining jobs if any
    if remaining!=0:
        ## 1). distribute it preferrentially to thos nodes with no jobs assigned.
        for i in xrange(len(nassign)):
            if nassign[i]==0:
                nassign[i]=nassign[i]+1
                ntot = ntot + 1
                remaining = remaining - 1
            if remaining==0:
                break
        ## 2). distribute equally in the order.
        for i in xrange(len(unit)):
            if remaining==0: break
            nassign[i] = nassign[i] + 1
            remaining = remaining - 1
            ntot = ntot + 1
        ## 3). randomly distribute the remaining until it is finished.
        if remaining!=0:
            irand = np.random.randint
            while remaining>0:
                k = irand(0,len(nassign))
                nassign[k] = nassign[k] + 1
                remaining = remaining - 1
                ntot = ntot + 1

    print 'final:'

    print '%8s'%'QueType',
    for i in xrange(len(unit)):
        print '%6s '%que_kinds[i],
    print '\n%8s'%'Cores',
    for i in xrange(len(unit)):
        print '%6i '%unit[i],
    print '\n%8s'%'# Jobs',
    for i in xrange(len(unit)):
        print '%6i '%nassign[i],

    ntot = nassign.sum()
    remaining = njobs - ntot
    print '\n\n--------------'
    print 'remaining:', remaining
    return unit, nassign, que_kinds


## Modulues within the main functions
def renew_resource(njobs, allow_fq, min_cpu, max_cpu):
    """
    Renew the list of available cores

    arguments
    =========
    njobs
    allow_fq
    min_cpu
    max_cpu
    """
    cores, nj, que_kinds = find_availability(
        njobs=njobs,allow_fq=allow_fq,min_cpu=min_cpu,
        )
    if cores==0: avail_cpu=0
    else: avail_cpu = cores[0]

    if avail_cpu==0:
        t0 = time.time()
        print 'sleep begins in renew_resources'
        dt = 1.; elapsed=0
        istay=True
        while (istay):
            t1=time.time()
            time.sleep(dt)
            elapsed = elapsed + dt
            uet(elapsed)
            if t1-t0>15.:
                cores, nj, que_kinds = find_availability(
                    njobs=njobs,allow_fq=allow_fq,
                    min_cpu=min_cpu)
                if cores!=0:
                    avail_cpu=cores[0]
                    istay=False
                else: pass
            else: pass
        print '\nsleep ends'
        pass

    if avail_cpu>max_cpu: use_cpu = max_cpu
    else: use_cpu = avail_cpu
    return que_kinds[0], use_cpu


def timer(buffer=10.):
    """
    Run optimize_jobs after 7200 seconds.
    """
    import glob
    iexit=False
    t0 = time.time()

    while not(iexit):

        t1=time.time()
        uet(t1-t0)
        fns = glob.glob('fld_a_rst_2015030*.tar')
        if len(fns)!=0:
            iexit=True
        else:
            time.sleep(buffer)

    fn = fns[0]
    optimize_jobs(
        fn_FLDa=fn,isub=True,min_cpu=16,max_cpu=16,
        wait_sec=15.,allow_fq=True)


def optimize_jobs(
    h0 = 0.995,
    psi0s   = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90],
    fn_FLDa = None,isub=False,time_pause=60,allow_fq=False,
    max_cpu=16,min_cpu=8,wait_sec=15.):
    """
    Given the condition and the resources,
    break a parallel-job into severals.
    Then, prepare an ample number of job files

    Arguments
    =========
    h0      = 0.990
    psi0s   = [0,5,10,15 ... ]
    fn_FLDa = 'fld_a_rst_20141023.tar'
    """
    import os, time
    if not(os.path.isfile(fn_FLDa)):
        raise IOError, 'Could not find %s'%fn_FLDa

    FLDA_date = fn_FLDa.split('_')[3][:8]
    FLDA_hash = os.path.split(fn_FLDa)[-1].split('.tar')[0][::-1][:6][::-1]


    # f = popen('tar -tf %s'%fn_FLDa)
    # lines=f.readlines()
    # npath = len(lines)

    bash_fns = []
    for i in xrange(len(psi0s)):
        ## renew core
        que_type, use_cpu = renew_resource(
            19, allow_fq, min_cpu, max_cpu)

        fn = 'FLB_%2.2i.sh'%(i+1)
        bash_fns.append(fn)
        write_bash_FLDb(que_type,fn,use_cpu,psi0s[i],h0,FLDA_date=FLDA_date,
                        FLDA_hash=FLDA_hash)
        if isub:
            std=os.popen('qsub %s'%fn)
            pid = std.readline().split('job')\
                [1].split()[0]
            print pid
            t0 = time.time()
            print 'sleep begins'
            istay=True
            dt = 1.; elapsed=0
            while (istay):
                t1=time.time()
                time.sleep(dt)
                elapsed = elapsed + dt
                uet(elapsed)
                if t1-t0>wait_sec:
                    istay=False
                else: pass
            print '\nsleep ends'

    return bash_fns

def write_bash_FLDb(que_kind,fn,ncore,psi0,f0,FLDA_date,FLDA_hash):
    f = open(fn,'w')
    f.write('#!/bin/bash\n')
    f.write('## Forming Limit Diagram Calculation based on VPSC-MK\n')

    if que_kind=='fast':
        f.write('#$ -q fast\n')
        f.write('#$ -l short=TRUE\n')

    f.write('#$ -pe nodal %i\n'%ncore)
    f.write('#$ -M ynj\n')
    f.write('#$ -m be\n')
    f.write('#$ -cwd\n\n')
    f.write('# Job command\n')
    f.write('/users/ynj/anaconda/bin/python FL.py ')
    f.write("--h0s '%5.3f' --ncpus %i"%(f0,ncore))
    f.write(" --ahash %s"%FLDA_hash)
    f.write(" --psi0s '%2.2i'  --d %s\n"%(psi0,FLDA_date))
    f.close()

if __name__=='__main__':
    find_availability()

