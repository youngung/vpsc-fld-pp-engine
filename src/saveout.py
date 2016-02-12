"""
Managing output/input files to vpsc
"""
from glob import glob
import os

def find_gnu_tar():
    """
    Find GNU tar command

    trials: ['tar','gtar']
    """
    import subprocess as sp
    from tempfile import mkstemp

    trials = ['tar','gtar']
    for cmd in trials:
        line = '%s --version'%cmd
        # print 'Line:\n-----\n',line

        _i_, fstdo = mkstemp(dir='/tmp'); fstdo = open(fstdo,'w')
        p=sp.Popen([line],shell=True,stdout=fstdo)
        p.poll();p.wait();fstdo.close()
        dum=open(fstdo.name,'r')
        # print dum.read();
        dum.close()
        iexit = p.returncode

        # iexit=os.system(line)

        if iexit==0:
            f=os.popen(line,'r')
            if f.read().find('GNU')!=-1:
                return cmd
            f.close()
    return None

_tar_ = find_gnu_tar()
if type(_tar_)==type(None): raise IOError, 'Could not find GNU tar'

def main_tar(fn_tar=None,
             wild_cards=['*.out','*.OUT','*.bin',
                         'FLD_nu.in',
                         'vpsc7.in','vpsc'],
             path0='',iopt=0):
    """
    Write a tar command and return together
    using the given (or suggested) tar file name
    (fn_tar)

    Arguments
    =========
    fn_tar     = None
    wild_cards = [...]
    paths0     = ''
    iopt       = 0

    ## if iopt==0: preppend path0 to command
    ## if iopt==1: don't preppend path0 to command
    """
    import time
    t = time.asctime()
    date = time.strftime('%Y%m%d')
    dat, month, dum, time, year = t.split()
    hr, mn, sec = time.split(':')

    if not(os.path.isdir('./archive')):
        os.mkdir('archive')

    if fn_tar==None:
        fn_tar = 'archive%s%s_%sh%sm.tar'%(
            os.sep,date,hr,mn)

    # add file names
    fns = [] # targetted file to tar
    for ic in xrange(len(wild_cards)):
        #print 'wild_cards', wild_cards[ic]
        if path0!='':
            fwc='%s%s%s'%(path0,os.sep,wild_cards[ic])
            #print 'fwc:',fwc
        else:
            fwc=wild_cards[ic]

        fs = glob(fwc)
        for i in xrange(len(fs)):
            fns.append(fs[i])

    fns = fns + find_files() # input files to vpsc # make it optional
    cmd = '%s --warning=none -cf %s'%(_tar_,fn_tar)

    ## if iopt==0: preppend path0 to command
    ## if iopt==1: don't preppend path0 to command
    if iopt==0:
        if path0!='': path0 = '%s%s'%(path0,os.sep)
        for i in xrange(len(fns)):
            _fn_ = os.path.join(path0,fns[i])
            cmd = cmd + ' %s%s'%(path0,fns[i])

    elif iopt==1:
        for i in xrange(len(fns)):
            if os.path.isfile(fns[i]):
                cmd = cmd + ' %s'%(fns[i])

    return cmd, fn_tar

def main(a,b,test=True):
    """
    wild card a to wild card b
    """
    from shutil import move
    fns = glob('*%s*'%a)

    for i in xrange(len(fns)):
        new_nf = fns[i].replace(a,b)
        if test: print i,fns[i], new_nf
        else:
            move(fns[i],new_nf)

def find_files(fn='vpsc7.in'):
    """
    Find files that are listed in 'vpsc7.in'
    single crystal (fsx)
    and texture files (ftex)
    """
    dl = open(fn,'r').readlines()
    nph = int(dl[1].split()[0])
    ftex=[]; fsx=[];
    fmaster = []

    ihead = 3
    nbl = 10
    for i in xrange(nph):
        icl0 = ihead + nbl*i
        itex = icl0 + 5
        isx  = icl0 + 7
        ftex.append(dl[itex][:-1].split('\r')[0])
        fsx.append(dl[isx][:-1].split('\r')[0])
        fmaster.append(dl[itex][:-1].split('\r')[0])
        fmaster.append(dl[isx][:-1].split('\r')[0])

    return fmaster

