"""
Prepare vpsc run on /tmp/ folder

1). Make vpsc/date folder
2). Migrate the executable and relevant input files
"""
import os
import shutil
import saveout
import time

import saveout

def fld_b_submits(
    tar_a='fld_a_rst_20140703.tar',
    rho0=-0.7,rho1=2.7,n=35,
    a0=0,a1=90.01,dang=15,fs=[0.980,0.990,0.995],
    i_fque=True):
    """
    Create a bunch of 'bash' scripts to run in SGE cluster
    for region B forming limit calculations
    using the give region A tar (tar_a)

    ! create paths/condition files on designated paths
    ! start parallel running
    """
    for i in xrange(len(fs)):
        run_paths,bash_fns = fld_b_prepare(
            wc=[tar_a,'fld.py','saveout.py','run_tmp.py',
                'find_rhos'],
            rho0=rho0,rho1=rho1,n=n,
            a0=a0,a1=a1,dang=dang,f0=fs[i],i_fque=i_fque)
        for j in xrange(len(run_paths)):
            path0=os.getcwd()
            os.chdir(run_paths[j])
            os.system('qsub %s'%bash_fns[j])
            os.chdir(path0)

def fld_b_prepare(
    wc=['fld_a_rst_20140623.tar','fld.py','saveout.py','run_tmp.py'],
    a0=0,a1=90.01,dang=15,f0=0.99,i_fque=True,
    rho0=-0.7,rho1=2.7,n=35
    ):
    """
    Prepare folder in which a particular forming limit
    probing condition is configured.
     * psi angle, f0 and the given fld_a_rst file
       should be provided
    """
    import numpy as np
    import fld
    angs = np.arange(a0,a1,dang,dtype='int')
    bash_fns  = []
    run_paths = []
    for i in xrange(len(angs)):
        wwc=wc[:]
        fn,cwd = fld.write_fld_bash(
            iopt=1,psi0=angs[i],

            rho0=rho0,rho1=rho1,n=n,

            f0=f0,isave=True,
            i_fque=i_fque)
        bash_fns.append(fn)
        wwc.append(fn)
        dst = '/users/ynj/repo/vpsc-fld_psi%3.3i_f%5.3f'%(angs[i],f0)
        run_paths.append(dst)
        main(dst=dst,iopt=1,wc=wwc)
    return run_paths, bash_fns

def read_tar(wc=[],wild_cards=['*.out','*.OUT','*.bin',
                               'vpsc7.in','vpsc']):
    """
    Making a tar file that contains relevant file names
    based on the wild_cards
    Read tar file name

    Arguments
    =========
    wc=[]
    wild_cars=['*.out','*.OUT','*.bin','vpsc7.in','vpsc']
    """
    # default wild cards
    for i in xrange(len(wc)):
        wild_cards.append(wc[i])
    cmd, fn_tar = saveout.main_tar(
        wild_cards=wild_cards)
    return cmd, fn_tar

def main(dst='/tmp',wc=[],
         wild_cards=['region_a.bin','FLD_nu.in',
                     'vpsc7.in','vpsc']):
    """
    To the given destined folder, move relevant files.

    Arguments
    =========
    dst        = '/tmp'
    wc         = []
    wild_cards = ['region_a.bin','FLD_nu.in',
                  'vpsc7.in','vpsc']
    """
    cmd, fn_tar = read_tar(wc,wild_cards)
    os.system(cmd) ## Create the tape archive (fn_tar)

    date, time = fn_tar.split(os.sep)[-1].split('_')[0:2]
    h = time.split('h')[0]
    m = time.split('h')[1].split('m')[0]
    if not(os.path.isdir(dst)):
        raise IOError, \
            'Destined folder %s does not exist'%dst
    ## move the created tar files to destined folder
    shutil.move(fn_tar, dst)
    fn_tar = fn_tar.split(os.sep)[-1]
    path0  = os.getcwd()
    ## get to the destined folder and extract/remove
    ## the tar file
    os.chdir(dst)
    try:
        os.system('%s -xf %s'%(saveout._tar_,fn_tar))
        # os.remove(fn_tar)
    except:
        os.chdir(path0)
        raise IOError, \
            'Failed to extract %s'%fn_tar
    os.chdir(path0)

def run_vpsc(dst='/tmp',cmd='./vpsc'):
    """
    In the designated path run vpsc using os.system

    Arguments
    =========
    dst
    cmd
    """
    path0 = os.getcwd()
    os.chdir(dst)
    os.system(cmd)
    os.chdir(path0)

def tar_rst(dst='/tmp',fn_tar=None):
    """
    dst='/tmp'
    fn_tar=None
    """
    path0 = os.getcwd()
    os.chdir(dst)
    if fn_tar==None:
        date,t = get_stime()
        fn_tar = 'run_tmp_rst_%s_%s.tar'%(date,t)
    saveout.main_tar(fn_tar=fn_tar)
    full_cwd = os.getcwd()
    full_fn_tar = '%s%s%s'%(full_cwd,os.sep,fn_tar)
    os.chdir(path0)
    return full_fn_tar

def get_stime():
    """
    date/time format for naming convention in VPSC-FLD
    """
    import time
    t=time.asctime()
    date = time.strftime('%Y%m%d')
    dat,month,dum,time,year = t.split()
    hr,mn,sec = time.split(':')
    return date,hr+mn+sec
