"""
A Python script that runs multi-threaded VPSC-based
Forming Limit Diagram predictions

----------------------------------------------
Youngung Jeong, PhD
youngung.jeong@gmail.com
----------------------------------------------
2016March-Present, younguj@clemson.edu
International Center for Automotive Research
Clemson University
----------------------------------------------
2014March-2016Feb, youngung.jeong@nist.gov
Center for Automotive Lightweighting
National Institute of Standards and Technology
----------------------------------------------

"""
import os, saveout, thread, run_tmp, time, shutil
import numpy as np
from multiprocessing import Process
from MP import progress_bar
uet = progress_bar.update_elapsed_time
_tar_ = saveout._tar_

def find_tmp():
    """
    Find the relevant temp folder
    in compliance with the CTCMS cluster policy,
    The rule is if there's /data/
    create files there and run vpsc there.

    Returns
    -------
    _tmp_
    """
    ## Find /data/
    if os.path.isdir('/local_scratch/'): ## Palmetto@Clemson
        _tmp_ = '/local_scratch/'
    elif os.path.isdir('/data/'): ## CTCMS cluster@NIST
        # _tmp_='/data/ynj/'
        _tmp_='/data/ynj/scratch/'
    else:
        _tmp_='/tmp/ynj/'
    if not(os.path.isdir(_tmp_)):
        os.mkdir(_tmp_)
    return _tmp_
_tmp_ = find_tmp()

def email_notification(
    subj='',
    msg='',
    recipient=''
    ):
    cmd = os.popen('which mail').read().split('\n')[0]
    os.popen("%s -s '%s' ynj <<< '%s' "%(
        cmd, subj, msg))

def write_fld_bash(
    iopt  = 0,
    n     = 35,
    n_cpus=64,
    i_fque = True,
    isave = True,

    # region A
    rho0  = -0.5,
    rho1  =  2.5,
    cvm   =  1e-3,
    deij  =  1e-3,
    evm_l =  1.5,

    # region B
    f0    =  0.990,
    psi0  =  0.0,

    # region B (p)
    f0s   = [0.990],
    psi0s = [0.0],
    ):
    """
    Making bash script for submitting to cluster

    Argument
    ========

    * Common
    iopt  =  0     (0: A region; 1: B region, 2: B region (eff) )
    n     = 35     (number of prefixed probings)
    n_cpus= 64
    i_fque= True
    isave = False

    * Region A specific
    rho0  =  -0.7   (initial rho value)
    rho1  =   2.7   (final rho value - 2.7 corresponding to -0.7 for axis 2)
    cvm   =   0.001   (Von Mises strain rate given to region A)
    deij  =   0.00005 (incremental strain for major axis in case of region A)
    evm_l =   2.0   (limit VM strain for probing in region A)

    # Region B specific
    f0     =   0.99  (inhomogeneity typically 0.99)
    psi0   =   0.0   (psi angle)

    # Region B (p) specific
    f0s    =  [0.99]
    psi0s  =  [90.0]
    """
    p = os.popen('which python'); d = p.read();

    # common keys
    _python_ = d[:-1]
    _bash_='#!/bin/bash\n'
    _header_='## Forming Limit Diagram Calculation based on VPSC-FLD\n'

    if iopt==0 or iopt==1:
        d= np.linspace(rho0,rho1,n)
        for i in xrange(len(d)):
            print '%5.2f'%d[i],
        print
        if i_fque: _qsub_args_=['-pe nodal %i'%n,
                                '-q fast','-l short=TRUE',
                                '-M ynj','-m be','-cwd']
        if not(i_fque): _qsub_args_=['-pe nodal %i'%n,
                                     '-M ynj','-m be','-cwd']
    if iopt==2:
        if i_fque: _qsub_args_=['-pe nodal %i'%n_cpus,
                                '-q fast','-l short=TRUE',
                                '-M ynj','-m be','-cwd']
        if not(i_fque): _qsub_args_=['-pe nodal %i'%n_cpus,
                                     '-M ynj','-m be','-cwd']
    if iopt==0 or iopt==1: _fld_ = 'fld.py'
    if iopt==2: _fld_ = 'FL.py'
    cmd = '%s %s'%(_python_, _fld_)

    _o_ = '%i'%iopt
    _n_ = '%3.3i'%n
    _u_ = '%i'%n_cpus
    # region A specifics
    _i_ = '%4.2f'%rho0
    _f_ = '%4.2f'%rho1
    _r_ = '%6.4f'%cvm
    _d_ = '%6.4f'%deij
    _l_ = '%4.2f'%evm_l
    key_a=dict(o=_o_,n=_n_,i=_i_,f=_f_,
               r=_r_,d=_d_,l=_l_,u=_u_)
    # region B specifics
    _h_ = '%5.3f'%f0
    _p_ = '%2.2i'%psi0
    key_b=dict(o=_o_,n=_n_,h=_h_,p=_p_,u=_u_)

    # region B (p) specifics
    _h0s_ = "'"; _psi0s_ = "'";
    for i in xrange(len(f0s)):
        _h0s_ = '%s %5.3f'%(_h0s_,f0s[i])
    _h0s_ = "%s'"%_h0s_
    for i in xrange(len(psi0s)):
        _psi0s_ = '%s %5.3f'%(_psi0s_,psi0s[i])
    _psi0s_ = "%s'"%_psi0s_
    key_bp=dict(n=_n_,h0s=_h0s_, psi0s=_psi0s_,u=_u_)

    if iopt==0: # region A
        fn = 'FLa'
        fn = '%s.sh'%fn
        for i in xrange(len(key_a.keys())):
            key   = key_a.keys()[i]
            value = key_a.get(key)
            cmd = '%s -%s %s'%(cmd, key, value)

    if iopt==1: # region B
        fn = 'FLb'
        fn = '%s%s%s.sh'%(fn,'%3.3i'%(f0*1000),_p_)
        for i in xrange(len(key_b.keys())):
            key   = key_b.keys()[i]
            value = key_b.get(key)
            cmd = '%s -%s %s'%(cmd, key, value)

    if iopt==2: # region B (p)
        fn = 'FLb_p'
        fn = '%s%2.2i%2.2i.sh'%(fn, len(f0s),len(psi0s))
        for i in xrange(len(key_bp.keys())):
            key = key_bp.keys()[i]
            value = key_bp.get(key)
            cmd = "%s --%s %s"%(cmd,key,value)

    if isave:
        f=open(fn,'w')
        f.write(_bash_)
        f.write('\n')
        f.write(_header_)
        f.write('\n')
        for i in xrange(len(_qsub_args_)):
            f.write('#$ ')
            f.write(_qsub_args_[i])
            f.write('\n')
        f.write('\n')

        f.write('# job command\n')
        f.write(cmd)
        f.write('\n')
    return fn, cmd

def main(
        # for region A
        ialph=False,
        rho0=-0.5,rho1=1.0,
        cvm = 1e-3,deij=1e-3,
        evm_limit=0.01,
        theta=0.,
        delay = 0.,

        # for region B
        f0=0.99,psi0=45.,
        FLDa_date='20140304', # my starting date at NIST
        FLDa_hash=None,n_probe=None,
        ipro=-32,

        ## iopt 0 or 1 respectively for regions A and B
        iopt=None,path_hub=None,

        ## Snapshots
        fn_ss0=None,
        isnapshot1=False):
    """
    Remove iopt=1 case and dedicate this function
    only for region A simulations.

    Prepare for a set of given strain paths.

    Prepare a transient temp folder -> move files from <path_home>
    to <path_hub> then share the files to individual working
    directories this set-up might be computationally more
    efficient to a case where <path_home> serves as the <path_hub> in
    case the <path_home> is not under a temp folder. Moreover, the
    <path_home> remains uncoupled from <path_hub> so that submit
    various jobs from <path_home> becomes plausible.

    Arguments
    =========
    # for region A
    ialph=False (if True, run with stress ratio - alpha - control)
    rho0=-0.5
    rho1=1.0
    n_probe=4
    cvm = 1e-3 deij=1e-3
    evm_limit=0.01
    delay=0.
    theta=0. : Theta determines the misorientation bewteen
               the texture file and the 'stretching' axes.
               - Usually, texture file is aligns with
                 the anisotropic orthotropic axes from rolling
                 (such as RD/TD/ND align with 1, 2, and 3 basis axes)
               - None-zero theta would invoke a case where
                 the stretching axes deviate from RD/TD/ND thus
                 inducing a significant amount of 'shear' components (either L12 or S12)
               - If theta is given, rotate the 'designated texture' file..

    # for region B
    f0=0.99 psi0=45.
    FLDa_date='20140304'  # my starting date at NIST
    ipro = 32 (NR) or -32 (DA)

    ## iopt 0 or 1 respectively for regions A and B
    iopt=None
    path_hub = None
        if path_hub is given (not None), then
            the given path_hub is used as the hub

    ## Snapshots
    fn_ss0     = None    - Initial snapshot file name
    isnapshot1 = False   - Switch for snapshot after the main proc

    Returns
    =======
    path_hub, paths_run, main_fn_tar, ngrs, fn_ss1
    """
    import vpscTools.parsers.vpsc_parsers as vp_parsers
    import vpscTools.wts_rotate as wts_rotate
    vp_in = vp_parsers.in_parser
    path_home = os.getcwd() # path home

    from tempfile import NamedTemporaryFile as ntf
    from tempfile import mkdtemp
    if iopt==None:
        raise IOError, 'Unspecified FDL simulation option'

    ## Treatment for the case theta!=0
    tiny = 1.e-5
    ngt=None
    if iopt==0 and abs(theta)>tiny:
        print "Rotate the texture file..."
        with open('vpsc7.in','r') as fo:
            vpsc_in_ref = fo.read()
            vpsc_in_lines = vpsc_in_ref.split('\n')

        tex_fns, tex_i_lines = vp_in('vpsc7.in',iopt=1)
        ## count the number of grains in the texture file
        ngt = 0
        for i in xrange(len(tex_fns)):
            ngt = ngt + int(os.popen('wc -l %s'%tex_fns[i]\
            ).read().split()[0]) - 4

        ## Rotate all of the tex_fns and
        ## save them to other names...
        for i in xrange(len(tex_fns)):
            tex_fn_old = tex_fns[i]
            tex_i_line = tex_i_lines[i]
            tex_fn_new = wts_rotate.main(tex_fn_old, theta)
            ## replace i_tex_lines[i] with tex_fn_new...
            # print tex_fn_new
            tex_fn_new = tex_fn_new.split(os.getcwd())[-1][1:]
            if not(os.path.isfile(tex_fn_new)):
                raise IOError, 'tex_fn_new was not found'
            print tex_fn_new
            vpsc_in_lines[tex_i_line] = tex_fn_new

        ## swap 'vpsc7.in' ...
        # for i in xrange(len(vpsc_in_lines)):
        #     print vpsc_in_lines[i]
        _temp_vpscin_theta_ = ntf(delete=True).name
        print 'vpscin before that: ', _temp_vpscin_theta_
        shutil.copy('vpsc7.in',_temp_vpscin_theta_)
        with open('vpsc7.in','w') as fo:
            for i in xrange(len(vpsc_in_lines)):
                fo.write('%s\n'%vpsc_in_lines[i])

    ipath_hub = False
    if type(path_hub)==type(None):
        ipath_hub = True ## initial path hub given
        path_hub  = mkdtemp(
            dir=_tmp_,
            prefix='VPSCFLD_hub_iopt%i_'%iopt) # path hub
        print 'path_hub:', path_hub

    ## --------------------------------------------------------
    ## Create 'Hub'
    ## Move tar file that contains the prepped tar for
    ## the first path (say, rho=-0.5)
    # print iopt, ipath_hub
    ngrs=[]
    if iopt==0:
        ## Arbitrary  arguments to path_hub ...
        fld_pre_probe(
            ialph=ialph,
            rho=-0.5,cvm=1e-3,evm_limit=0.01,
            deij=1e-4,dst=path_hub)
    elif iopt==1 and ipath_hub:
        fn_a_tar, _theta_ = most_recent_tar(
            iopt=1,FLDa_hash=FLDa_hash)

        print fn_a_tar
        print 'fn_a_tar:', fn_a_tar
        shutil.copy(fn_a_tar, path_hub)

        ## in newer versions (later than 20150925)
        ## fld_a tar file contains 'vpsc7.in'
        ## use those files rather than the locally found one.
        cmd = 'tar -tf %s'%fn_a_tar
        elements = os.popen(cmd).read().split('\n')

        ## use 'vpsc7.in' contained in fld_a*.tar file.
        if 'vpsc7.in' in elements:
            _temp_vpscin_ = ntf(delete=True,dir=_tmp_).name
            print _temp_vpscin_
            shutil.move('vpsc7.in',_temp_vpscin_)
            cmd = 'tar -xf %s vpsc7.in'%fn_a_tar
            os.system(cmd)
            fld_post_probe(
                f0=f0,psi0=psi0,dst=path_hub,
                wc=['find_rhos','FLD_nu.in'],ipro=ipro)
                # fn_ss0=fn_ss0,
                # isnapshot1=isnapshot1)

            tex_fns, tex_i_lines = vp_in(_temp_vpscin_,iopt=1)
            ## count the number of grains in the texture file

            ngt = 0
            for i in xrange(len(tex_fns)):
                ngrs.append(int(os.popen(
                    'wc -l %s'%tex_fns[i]).\
                                read().split()[0]) - 4)
                ngt = ngt + ngrs[i]

            shutil.copy(_temp_vpscin_,'vpsc7.in')
        else:
            print '-'*40
            print 'Warning !!'
            print 'Your FLD-a tar does not contain vpsc7.in'
            print 'Your local vpsc7.in as-it-is will be used'
            print 'as a reference for <fld.fld_post_probe>'
            print '-'*40
            raise IOError, 'Consider using a recent version of VPSC-FLD'
            # ## out-dated way
            # fld_post_probe(
            #     f0=f0,psi0=psi0,dst=path_hub,
            #     wc=['find_rhos','FLD_nu.in'],ipro=ipro)

    elif iopt==1 and not(ipath_hub):
        ## use ipath_hub
        if type(n_probe)==None:
            raise IOError, \
                'n_probe should be given'+\
                ' for the configured case'
    else: raise IOError, 'Unexpected iopt was given'
    ## --------------------------------------------------------

    os.chdir(path_hub) ## enter hub directory

    P       = []; stdouts   = []; stderrs    = [];
    fntars  = []; paths_run = [];
    activ_p = []; n_active  = 0 ; #rho_values = [];


    fn_ss1 = []
    for i in xrange(n_probe):
        if iopt==0: # region A
            if n_probe>1:
                rho = rho0 + (rho1-rho0)/float(n_probe-1.) * i
            elif n_probe==1:
                rho = rho0
            path_run = mkdtemp(
                dir=_tmp_,prefix='VPSCFLD_a_%2.2i_'%i)

            evm_type =type(evm_limit).__name__
            if 'float' in evm_type:
                evm=evm_limit
            elif evm_type=='list' or evm_type=='ndarray':
                ## variable evm_limits ...
                evm=evm_limit[i]
            else:
                print 'evm_limit type:',type(evm_limit).__name__
                raise IOError, 'wrong type'

            _fs1_ = fld_pre_probe(
                ialph=ialph,
                rho=rho,cvm=cvm,
                evm_limit=evm,
                deij=deij,dst=path_run,
                fn_ss0=fn_ss0,
                isnapshot1=isnapshot1)
            fn_ss1.append(_fs1_)

        elif iopt==1: # region B
            path_run = mkdtemp(
                dir=_tmp_,prefix='VPSCFLD_b_%2.2i_%3.3i_%i_'%(
                    i,psi0,f0*1000))
            print 'path_run:', path_run
            shutil.copy('FLD_nu.in',path_run)
            # shutil.copy('vpsc',path_run)
            # Extract region_a.bin ---------
            fld_post_tarx(FLDa_date=FLDa_date,i=i,
                          FLDa_hash=FLDa_hash)
            # ------------------------------
            _fs1_ = fld_post_probe(
                f0=f0,psi0=psi0,dst=path_run,
                ipro=ipro,fn_ss0=fn_ss0,
                isnapshot1=isnapshot1)
            fn_ss1.append(_fs1_)
            shutil.move('region_a.bin',path_run)
            ## shutil.copy('vpsc7.in',path_run)

        else: raise IOError, 'Unexpected iopt was given'

        time.sleep(delay)

        ## naming convention for the tar file.
        if iopt==0: lab='a'
        if iopt==1: lab='b'
        fn_tar_member = 'fld_%s_rst_%2.2i.tar'%(lab,i)
        fn_tar_member = os.path.join(path_run,fn_tar_member)

        fntars.append(fn_tar_member)
        paths_run.append(path_run)

        # print 'VPSC-FLD run prepared for the %s path'%n2str(i+1)
        ## end of n_probe for-loop

    ## recover the original vpsc7.in in case that!=0.
    if abs(theta)>tiny and iopt==0:
        shutil.copy(_temp_vpscin_theta_,
                    os.path.join(path_home,'vpsc7.in'))

    os.chdir(path_home) # exit from path_hub to path_home
    ### Exit for more general parallel option
    # if iopt==0: return path_hub, paths_run, fntars
    # if iopt==1: return path_hub, paths_run, fntars, rho_values
    return path_hub, paths_run, fntars, ngt, fn_ss1

def fld_vpsc_run(dst=_tmp_,iwait=True,path2vpsc='.'):
    """
    Run vpsc in the destined folder and save the results
    to a tar file

    Arguments
    =========
    dst      ='/tmp'
    iwait    =True
    path2vpsc='.'

    Returns
    =======
    p, stdo, stde, elapsed_time
    """
    import subprocess
    t0 = time.time()
    path0=os.getcwd()
    os.chdir(dst)
    std_out = open('stdout.out','w')
    std_err = open('stderr.out','w')
    process = subprocess.Popen(
        [os.path.join(path2vpsc,'vpsc')],#shell=True,
        stderr=std_err,stdout=std_out,
        stdin=subprocess.PIPE)

    with open('pid_%i'%process.pid,'w') as f:
        f.write(time.asctime())

    if iwait: process.wait()
    os.chdir(path0)
    std_out.close()
    std_err.close()
    return process,std_out,std_err,time.time()-t0

def fld_vpsc_run2(dst=_tmp_):
    """
    Run vpsc in the destined folder and save the results
    to a tar file

    Arguments
    =========
    dst='/tmp'

    Returns
    =======
    p, stdo, stde, stdo_read,elapsed_time
    """
    import subprocess
    t0 = time.time()
    path0=os.getcwd()
    os.chdir(dst)
    std_out      = open(os.path.join(path0,'stdout.out'),'w')
    std_out_read = open(os.path.join(path0,'stdout.out'),'r')
    std_err = open(os.path.join(path0,'stderr.out'),'w')
    process = subprocess.Popen(
        ['./vpsc'],shell=True,
        stderr=std_err,stdout=std_out,
        stdin=subprocess.PIPE)
    # process.wait()
    os.chdir(path0)
    std_out.close()
    std_err.close()
    return process,std_out,std_err,std_out_read,\
        time.time()-t0

def fld_pre_probe(
        ialph=False,
        rho=0.,
        cvm=1e-3,
        evm_limit=2.0,
        deij=5e-05,
        dst=_tmp_,
        fn_ss0=None,
        isnapshot1=False):
    """
    Prep for region A simulation and move relevant files to
    the destined folder (dst)

    Arguments
    =========
    rho
    cvm
    evm_limit
    deij
    dst
    fn_ss0=None,
    isnapshot1=False

    Return
    =======
    fn_ss1
    """
    if ialph: ipro=35
    else:     ipro=30
    ## create the input file
    fn_ss1 = fld_fin(ipro=ipro,rho=rho,cvm=cvm,
                     evm_limit=evm_limit,deij=deij,
                     fnout='vpsc7.in',dst=dst,fn_ss0=fn_ss0,
                     isnapshot1=isnapshot1)
    ## Make the tar and move it to a destined folder.
    ## Extract and, then, delete it.
    run_tmp.main(
        dst=dst,wild_cards=['FLD_nu.in'])
    return fn_ss1

def fld_post_probe(f0,psi0,wc=[],dst=_tmp_,ipro=32,
                   fn_ss0=None,isnapshot1=False):
    """
    Run for region B

    ## Create vpsc7.in file and
    ## Then, move input and all the other necessary
    ## files to detined folder

    Arguments
    =========
    f0
    psi0
    wc   = []
    dst  = '/tmp'
    ipro = 32 (NR method) / -32 (DA method)
    fn_ss0=None
    isnapshot1=False):

    Return
    ======
    fn_ss1
    """
    ## create the input file in [dst]
    fn_ss1 = fld_fin(ipro=ipro,f0=f0,psi0=psi0,
                     fnout='vpsc7.in',
                     dst=dst,fn_ss0=fn_ss0,
                     isnapshot1=isnapshot1)

    ## find rhos and execute 'run_tmp.main'
    ## run_tmp.main does:
    ##     1) create tar and move it to the destined folder
    ##     2) Extract in that folder and remove the tar.
    run_tmp.main(
        dst=dst,wild_cards=[],wc=wc
    ) # move the files to the destined folder
    return fn_ss1

def find_rhos(dst=_tmp_):
    """
    ## find rhos.
    ## from 'region.a' generated from linear path

    Arguments
    =========
    dst='/tmp'

    Return
    ======
    rhos
    """
    home = os.getcwd()
    os.chdir(dst)
    iflag=os.system('./find_rhos')
    if iflag!=0:
        print os.getcwd()
        raise IOError
    p = os.popen(
        './find_rhos')
    rhos = map(float,p.readlines()[-1].split())
    os.chdir(home)
    return rhos

def fld_post_tarx(FLDa_date='20140623',FLDa_hash=None,
                  i=0):
    """
    Extract a tar of a single path from a collected tar file.
    Then, extract the region_a.bin
    """
    # fn_a_tar = 'fld_a_rst_%s_%s.tar'%(FLDa_date,FLDa_hash)
    fn_a_tar, _theta_ = most_recent_tar(
        iopt=1,FLDa_hash=FLDa_hash,
        FLDa_date=FLDa_date)
    fn_a_i   = 'fld_a_rst_%2.2i.tar'%i
    cmd = '%s -xf %s %s'%(_tar_,fn_a_tar,fn_a_i)
    os.system(cmd)
    cmd = '%s '%_tar_ + \
        ' -xf %s region_a.bin '%fn_a_i
    iflag=os.system(cmd)
    if iflag!=0:
        print 'cwd:', os.getcwd()
        print 'iflag:', iflag
        print 'cmd:', cmd
        raise IOError, 'region_a.bin was not extracted.'
    os.remove(fn_a_i)

## VPSC Process parsers
def __read_snapshot__(fn):
    return '-5  : Read snapshot \n%s\n'%fn
def __write_snapshot__(fn):
    return '5   : Save snapshot (snapshot file name) \n%s\n'%fn
def __pro30__(rho,cvm,deij,evm_limit):
    cnt = '30  ! FLD-region A single path (rho-control)\n '
    return cnt + '%5.3f  %7.3e  %7.3e  %7.3e \n'%(
        rho,cvm,deij,evm_limit)
def __pro35__(alpha,cvm,deij,evm_limit):
    cnt = '35  ! FLD-region A single path (alpha-control)\n '
    return cnt + '%5.3f  %7.3e  %7.3e  %7.3e \n'%(
        alpha,cvm,deij,evm_limit)
def __pro32__(f0,psi0):
    cnt = '32  ! FLD-region B single path (N-R method)\n'
    return cnt + '%11.8f  %5.3f \n'%(f0,psi0)
def __pron32__(f0,psi0):
    cnt = '-32 ! FLD-region B single path (D-A method)\n'
    return cnt + '%11.8f  %5.3f \n'%(f0,psi0)

def fld_fin(rho=0,deij=1e-3,cvm=1e-3,evm_limit=1.5,
            ipro=30,f0=0.99,psi0=0.0,fnout='test',
            dst=None,echo=False,fn_ss0=None,
            isnapshot1=False):
    """
    Create vpsc7.in for FLD simulations
    in [dst] directory if given.
    Otherwise, create it in the current working directory.

    Arguments
    =========
    rho        = 0
    deij       = 1e-3
    cvm        = 1e-3
    evm_limit  = 1.5
    ipro       = 30
    f0         = 0.99
    psi0       = 0.0
    fnout      = 'test'
    dst        = None
    echo       = False
    fn_ss0     = None
    isnapshot1 = False

    Return
    =======
    fn_ss1 (optional) - only when isnapshot1 is True
    """
    from tempfile import NamedTemporaryFile as ntf
    ## base vpsc7.in
    cwd      = os.getcwd()
    fn_input = os.path.join(cwd,'vpsc7.in')
    lines    = open(fn_input,'r').readlines()
    ## new vpsc7.in
    if type(dst)==type(None): dst=cwd
    fnout = os.path.join(dst,fnout)
    f = open(fnout,'w')
    npro = 1
    isnapshot0=False
    if type(fn_ss0)!=type(None):
        isnapshot0=True
        npro = npro + 1
    if isnapshot1==True:
        npro = npro + 1
        fn_ss1 = ntf(
            suffix='.ss',prefix='vpsc_snapshot_',
            delete=True,dir=_tmp_).name
    cnt = "%i     ## Number of processes\n"%npro
    cnt = cnt + "## List of process generated "
    cnt = cnt + "by fld.fld_fin\n"
    for i in xrange(29):
        f.write(lines[i])
    if isnapshot0:
        cnt=cnt+__read_snapshot__(fn_ss0)
    if ipro==30:
        cnt=cnt+__pro30__(rho,cvm,deij,evm_limit)
    if ipro==35:
        cnt=cnt+__pro35__(rho,cvm,deij,evm_limit)
    if ipro==32:
        cnt=cnt+__pro32__(f0,psi0)
    if ipro==-32:
        cnt=cnt+__pron32__(f0,psi0)
    if isnapshot1:
        cnt=cnt+__write_snapshot__(fn_ss1)
    f.write(cnt)
    f.close()
    if echo:
        print os.popen('cat %s'%f.name).read()
    if isnapshot1==True:
        return fn_ss1

def most_recent_tar(iopt=0,FLDa_hash=None,FLDa_date=None):
    """
    Find relevant FLD-a-region tar file.

    iopt=0  : out-dated version
    iopt=1  : new version that reads HASH code and the date
              of FLD-a tar file.
    """
    # 1. Neither FLDa_has and FLDa_date is given.
    if FLDa_hash==None and FLDa_date==None:
        wc='fld_a_rst_*.tar'
    # 2. FLDa_hash is given, while FLDa_date is not.
    if FLDa_hash!=None and FLDa_date==None:
        wc='fld_a_rst_*_%s.tar'%FLDa_hash
    # 3. FLDa_date is given, while FLDa_hash is not.
    if FLDa_date!=None and FLDa_hash==None:
        wc='fld_a_rst_%s_*.tar'%FLDa_date
    # 4. both FLDa_date and FLDa_hash is given
    if FLDa_date!=None and FLDa_hash!=None:
        wc='fld_a_rst_%s*%s.tar'%(FLDa_date, FLDa_hash)

    if iopt==0:
        from glob import glob
        fns=glob(wc)
        if len(fns)==0:
            raise IOError, 'Could not find relevant'+\
                ' tar files for region A'
        fns.sort()
        return fns[-1]
    elif iopt==1:
        import os
        # print 'wc:', wc
        lines = os.popen('ls -lshkl %s'%wc).read().split('\n')
        if len(lines)>2:
            raise IOError, 'Possibly multiple options in FLDa-file'
        fn_FLDa_tar = lines[0].split()[-1]

        th_el = fn_FLDa_tar.split('theta')
        if len(th_el)==1:
            theta = 0.
        elif len(th_el)==2:
            theta = float(th_el[1].split('_')[1])
        return fn_FLDa_tar, theta

def rho_transform(rho):
    """
    Rho transformation (rho<=1 or rho>1)

    Argument
    ========
    rho
    """
    if rho<=1.: return rho
    if rho>1: return -1 *(rho -1.) + 1

def write_FLDnu(
        dst=_tmp_,limit_factor=10,
        njacob_cal=1,dx_jac=1e-04,
        err_fi=1e-02,err_vi=1e-07,
        max_iter=20,
        head='##      VPSC-FLD numerical condition'):
    """
    Write 'FLD_nu.in' in the form of below:

    ##      VPSC-FLD numerical condition
    10      limit_factor: critical value of D33(B)/D33(A)
    1       njacob_cal  : frequency of jacobian matrix calculation
    1e-04   dx_jac      : step size (dv) for forward jacobian calc
    1e-02   err_fi      : error tolerance for sum of objv func Fi
    1e-07   err_vi      : saturating vi limit
    20      max_iter    : max iteration for jacobian calc.
    """
    fout=open(os.path.join(dst,'FLD_nu.in'),'w')
    fout.write(head+'\n')
    fout.write('%5.5i   limit_factor: critical value of D33(B)/D33(A)'%limit_factor+\
               '\n')
    fout.write('%5.5i   njacob_cal  : frequency of jacobian matrix calculation'%njacob_cal+\
               '\n')
    fout.write('%5.0e   dx_jac      : step size (dv) for forward jacobian calc'%dx_jac+'\n')
    fout.write('%5.0e   err_fi      : error tolerance for sum of objv func Fi'%err_fi+'\n')
    fout.write('%5.0e   err_vi      : saturating vi limit'%err_vi+'\n')
    fout.write('%5.5i   max_iter    : max iteration for jacobian calc.'%max_iter+'\n')
    fout.close()

def gen_hash_code(nchar=6):
    import hashlib
    ## -------------------------------------------------------
    ## Gen HASH code
    m = hashlib.md5()
    m.update(tar_date)
    m.update(time.asctime())
    m.update(time.time())
    return m.hexdigest()[:nchar]

def rho_from_xy(xy):
    xy = map(float, xy.split())
    ## if xy is given, that can be shared commonly for different paths, conduct (x,y)
    ## simulation first, take a snapshot and share it.
    print 'xy:', xy
    x,y = xy
    # _r_ = float(y)/float(x)
    ## rho transform
    if   x>=y:
        _r_ = float(y)/float(x)
    elif x<y:
        _r_ = float(x)/float(y)
        _r_ = 1 + (1 - _r_)

    z = -(x+y)
    _evml_ = np.sqrt(2./3.*(x**2+y**2+z**2))
    return _r_, _evml_

def test_pp():
    prestrain_planestrain(args=None,tar_date='30000000')

def prestrain_planestrain(args,tar_date):
    """
    Conduct plane-strain along 'TD' (axis 2)
    Take a snapshot after and return the name.

    Arguments
    =========
    args
    tar_date

    Return
    ======
    fn_ss1 (snapshot file name)
    """
    import vpscTools.parsers.vpsc_parsers as vp_parsers
    import vpscTools.parsers.vpsc_hist_parsers as hs_parsers
    import os, time
    from multiprocessing import Pool

    vp_in = vp_parsers.in_parser

    path_home = os.getcwd()

    rho0      = args.i
    rho1      = args.f
    cvm       = args.r
    deij      = args.d
    theta     = args.t
    ialph     = args.a
    prePS     = args.prePS

    nstp = prePS/deij

    ## Use fld.main to prep folders.
    rst = main(
        rho0=rho0,rho1=rho1,n_probe=1,
        cvm=cvm,deij=deij,
        evm_limit=prePS,theta=theta,iopt=0,isnapshot1=True)
    path_hub, paths_run, main_fn_tar, \
        ntotgr, fn_ss1 = rst
    fn_ss1=fn_ss1[0]

    vpsc_0    = os.path.join(path_home,'vpsc')
    vpsc_1    = os.path.join(path_hub,'vpsc')
    vpsc_in_0 = os.path.join(path_home,'vpsc7.in')
    vpsc_in_1 = os.path.join(path_hub,'vpsc7.in_bk')
    vpsc_in_new = os.path.join(path_hub,'vpsc7.in')

    ## plane-strain compression
    # vpsc_hist = hs_parsers.plane_strain_compression(
    #     nstp=nstp, de22=deij, cvm=cvm, fn_hist=None)
    ## uniaxial compression
    vpsc_hist = hs_parsers.unix_compression3(evm=prePS,cvm=cvm,deij=deij)


    shutil.copy(vpsc_0,vpsc_1)
    shutil.copy(vpsc_in_0,vpsc_in_1)

    # print 'path to hub:', path_hub
    # print 'path to run:', paths_run[0]
    # print 'vpsc_in_new:', vpsc_in_new

    ## modify vpsc7.in in path_hub and run.
    fo = open(vpsc_in_1,'r')
    lines_to_preserve = fo.readlines()[0:29]
    fo.close()

    with open(vpsc_in_new,'w') as fw:
        for i in xrange(len(lines_to_preserve)):
            fw.write(lines_to_preserve[i])

        ## lines_to_modify = fo.read().split('\n')[37:]
        print '#'*20
        pro0 = '2   : Number of processes\n## List of Processes generated in fld.py\n'
        pro1 = '0   : FLD-A prePS \n%s\n'%vpsc_hist
        pro2 = __write_snapshot__(fn_ss1)
        print pro0,
        print pro1,
        print pro2
        print '#'*20
        fw.write(pro0)
        fw.write(pro1)
        fw.write(pro2)
        ## Save new lines to vpsc_in_new.
        fw.close()

    pool = Pool(processes=1)

    results = pool.apply_async(
        func=fld_vpsc_run, args=(
            path_hub,True,path_hub))

    t0_start   = time.time()
    pool.close(); pool.join(); pool.terminate()
    elapsed = time.time()-t0_start
    print 'elapsed time:', elapsed
    return fn_ss1


## ------------------------------------------------------------
## Command line usage
if __name__=='__main__':
    """
    ## command line running for region A
    ## region A simulation example

    $> python fld.py --i -0.5 --f 1.0 --n 8 --r 0.001
                     --d 0.001 --l 1.0 --t 0. --xy '0.1 0.1'
    """
    import multiprocessing, saveout, argparse
    import run_tmp, fld_monit
    import vpscTools.parsers.vpsc_parsers as vp_parsers
    vp_in = vp_parsers.in_parser
    # vp_sx = vp_parsers.sx_parser
    from tempfile import mkdtemp
    from os import getcwd, chdir
    from tempfile import NamedTemporaryFile as ntf
    from multiprocessing import Pool
    from glob import glob

    path_home = getcwd()

    ## Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--i', type=float, help='rho0',default=-0.5)
    parser.add_argument(
        '--f', type=float, help='rho1',default=1.0)
    parser.add_argument(
        '--n', type=int,   help='Number of probings', default=16)
    parser.add_argument(
        '--r', type=float, help='Strain Rate (cvm)',  default=1e-3)
    parser.add_argument(
        '--d', type=float, help='Step size (deij)', default=1e-3)
    parser.add_argument(
        '--l', type=float, help='EVM_limit', default = 1.2)
    parser.add_argument(
        '--u', type=int,   help='Number of CPUs', default = multiprocessing.cpu_count())
    parser.add_argument(
        '--t', type=float,   help='theta: in-plane rotation of the given texture', default = 0.)
    parser.add_argument(
        '--a', action='store_true', help='Use alpha control (stress ratio)')
    parser.add_argument(
        '--xy', type=str,  help='Exx,Eyy -- prestrain for path change', default=None)
    parser.add_argument(
        '--prePS', type=float,  help='Prestrain along TD', default=None)
    parser.add_argument(
        '--hash', type=str, help='HASH code', default=None)
    parser.add_argument(
        '--pm', type=str,default=False,
        help ="POST MORTEM run based on the given 'FLD_*.csv'"+\
        " file. One should provide FLD-B hash")

    args      = parser.parse_args()
    rho0      = args.i
    rho1      = args.f
    n_probe   = args.n
    cvm       = args.r
    deij      = args.d
    evm_limit = args.l
    ncpu      = args.u
    theta     = args.t
    ialph     = args.a
    xy        = args.xy
    prePS     = args.prePS
    hash_a    = args.hash
    pm        = args.pm ## hash

    tar_date  = run_tmp.get_stime()[0]

    ## Check Conflicts between arguments
    if xy!=None and prePS!=None:
        raise IOError, 'Cannot prescribe both xy and prePS'

    if pm!=False and (xy!=None or prePS!=None):
        print 'pm:', pm
        print 'xy:', xy
        print 'prePS:', prePS
        raise IOError, 'Cannot use xy or prePS '+\
            'when pm (POSTMORTEM FLD-A calculation) is given'

    fn_ss0=None
    if prePS!=None:
        ## obtain snapshot after the pre-strain
        fn_ss0 = prestrain_planestrain(args,tar_date)

    ## --------------------------------------------------
    ## OVERWRITE arguments if pm is not False.
    ## Find PM with FLD-B hash...
    if pm==False:
        _evm_ = evm_limit
    else:
        ## Find evm_limits based on 'FLD_csv'...
        _p_='./archive/FLD_A/*/FLD_*/FLD_plot_%s.csv'%pm
        fn_csv = glob(_p_)
        if len(fn_csv)!=1:
            print fn_csv
            print _p_
            raise IOError, 'Must have only one file.'
        fn_csv = fn_csv[0]
        FLDA_dat = np.loadtxt(fn_csv,skiprows=1,delimiter=',',dtype='float').T
        Eyy = FLDA_dat[2]; Exx = FLDA_dat[3]
        Exy = FLDA_dat[4]; Ezz = FLDA_dat[5]

        ## Find argument file
        path_2_archive = os.path.split(fn_csv)[0]
        hash_a = os.path.split(path_2_archive)[-1][4:10]
        print 'path_2_archive:',path_2_archive
        # raise IOError, 'debug.'

        fn_arg = glob(path_2_archive+os.sep+'FLDA_args_*')
        if len(fn_arg)!=1:
            raise IOError, 'fn_arg is multiple...'

        fn_arg=fn_arg[0]
        with open(fn_arg,'r') as fo:
            lines   = fo.read().split('\n')
            rho0    = float(lines[0].split()[2])
            rho1    = float(lines[1].split()[2])
            n_probe = int(lines[2].split()[2])
            cvm     = float(lines[3].split()[2])
            deij    = float(lines[4].split()[2])
            ## evm_limit
            ## ncpu
            theta   = float(lines[7].split()[2])
            ialph   = lines[8].split()[2].strip()
            if ialph=='False': ialph=False
            if ialph=='True': ialph=True

            try:
                xy   = map(float,lines[9].split('[')[-1].split(']')[0].split(','))
                xy = '%f %f'%(xy[0],xy[1])
            except:
                xy = None
            #print 'xy:', xy

        ## Calculate EVM_limits
        _evm_ = []
        for i in xrange(n_probe):
            e11, e22, e12 = Exx[i], Eyy[i], Exy[i]
            e33 = -e11-e22
            _e_ = e11**2+e22**2+e33**2+2*(e12**2)
            _e_ = 2./3. * _e_
            _e_ = np.sqrt(_e_)
            _evm_.append(_e_)

    pool = Pool(processes=ncpu)
    if ialph:
        print 'Stress ratio control (FLD-alpha)'
    else:
        print 'Strain ratio control (FLD-rho)'

    ## --------------------------------------------------
    ## Prestrain.... (xy) or (ps)

    if xy!=None:
        _r_, _evml_ = rho_from_xy(xy)
        _rst_ = main(
            ialph=ialph,rho0=_r_,rho1=_r_, n_probe=1,
            cvm=cvm,deij=deij,evm_limit=_evml_,
            theta=theta,iopt=0,FLDa_date=tar_date,
            isnapshot1=True)
        _path_hub_, _paths_run_, _main_fn_tar_, \
            _ntotgr_dum_, fn_ss0 = _rst_
        ## Copy the executable to _path_hub
        shutil.copy(os.path.join(path_home,'vpsc'),_path_hub_)
        print _paths_run_[0]
        print _path_hub_
        print _main_fn_tar_[0]
        print _ntotgr_dum_  ## not important
        print 'len(fn_ss0):',len(fn_ss0)
        fn_ss0 = fn_ss0[0]
        results = pool.apply_async(
            func=fld_vpsc_run,args=(
                _paths_run_[0],True,_path_hub_))
        queue   = multiprocessing.Queue()
        p_monit = multiprocessing.Process(
            target=fld_monit.Monitor_worker,
            args=(queue,))
        p_monit.start()
        pr = np.array(_paths_run_[::])
        queue.put(fld_monit.RealTimeMonitor(pr))
        queue.close(); queue.join_thread()

        t0_start   = time.time()
        pool.close(); pool.join(); pool.terminate()
        wall_clock = time.time() - t0_start
        p_monit.terminate()

        ## Save the 'region_a.bin' file.
        ## this would be used later: 'cat fn1 fn2 > fn2'
        fn1 = os.path.join(_paths_run_[0],'region_a.bin')
    ## --------------------------------------------------

    # raise IOError, 'Debugged until this line'

    pool = Pool(processes=ncpu)

    # isnapshot1=False
    # if pm!=False: isnapshot1=True
    isnapshot1=True

    rst = main(
        ialph=ialph,rho0=rho0,rho1=rho1,n_probe=n_probe,
        cvm=cvm,deij=deij,evm_limit=_evm_,
        theta=theta,iopt=0,FLDa_date=tar_date,fn_ss0=fn_ss0,
        isnapshot1=isnapshot1)

    path_hub, paths_run, main_fn_tar, ntotgr, fn_ss1 = rst
    fn_vpsc_in_ref = os.path.join(path_home,'vpsc7.in')

    # ----------------------------------------------------------------
    # -- Archiving begins before pp-runs start
    ## 0. Collecting information on model-run conditions
    if abs(theta)>0.: th_str='theta_%+3.1f_'%theta
    else: th_str=''

    ## In case hash_a is given, use it.
    if hash_a==None:
        fn_FLDA_tar = ntf(
            suffix='.tar',delete=True,
            prefix='fld_a_rst_%s_%s'%(
                run_tmp.get_stime()[0],th_str))
        fn_FLDA_tar = fn_FLDA_tar.name
        FLDA_hash_code = os.path.split(fn_FLDA_tar)[-1].\
                         split('.tar')[0][::-1][:6][::-1]
    else:
        fn_FLDA_tar = 'fld_a_rst_%s_%s%s.tar'%(
            run_tmp.get_stime()[0],th_str,hash_a)
        FLDA_hash_code = hash_a

    print '-'*60
    print 'FLDA_hash_code:', FLDA_hash_code
    print 'FLDA filename :', fn_FLDA_tar
    print '-'*60

    fmaster, pro_dict = vp_in(fn_vpsc_in_ref)

    ## Create directory if missing.
    if pm==False:
        path2arch = os.path.join(
            path_home,'archive','FLD_A',
            tar_date,'FLD_%s'%(FLDA_hash_code))
        if not(os.path.isdir(path2arch)):
            os.makedirs(path2arch)
            pass
        ## Archiving.
        for _fn_ in [fn_vpsc_in_ref]+fmaster:
            shutil.copy(_fn_,path2arch)
        ## Saving the arguments as well.
        with open(os.path.join(
                path2arch,'FLDA_args_%s'%FLDA_hash_code),'w') as fo:
            fo.write('** %10s %.3f \n'%('rho0',rho0))
            fo.write('** %10s %.3f \n'%('rho1',rho1))
            fo.write('** %10s %i \n'%('n_probe',n_probe))
            fo.write('** %10s %.3e \n'%('cvm',cvm))
            fo.write('** %10s %.3e \n'%('deij',deij))
            fo.write('** %10s %.3e \n'%('evm_limit',evm_limit))
            fo.write('** %10s %i \n'%('ncpu',ncpu))
            fo.write('** %10s %.3e \n'%('theta',theta))
            fo.write('** %10s %s \n'%('ialph',ialph))
            fo.write('** %10s %s \n'%('xy',xy))
            if prePS==None:
                fo.write('** %10s None \n'%'prePS')
            else:
                fo.write('** %10s %.4f \n'%('prePS',prePS))

    # ----------------------------------------------------------------
    ## 1. run vpsc in paths_run
    results = []

    ## Copy the executable to _path_hub
    shutil.copy(os.path.join(path_home,'vpsc'),path_hub)

    for irho in xrange(len(paths_run)):
        rst = pool.apply_async(
            func=fld_vpsc_run,
            args=(paths_run[irho],True,path_hub))
        results.append(rst)

    # ----------------------------------------------------------------
    ## 2. Close the pool and wait
    imonit=True # False ## still need debugging
    if imonit:
        queue = multiprocessing.Queue()
        p_monit=multiprocessing.Process(
            target=fld_monit.Monitor_worker,args=(queue,))
        p_monit.start()
        pr=np.array(paths_run[::])
        queue.put(fld_monit.RealTimeMonitor(pr))
        queue.close();queue.join_thread()

    print 'Close pool'
    t0_start = time.time()
    pool.close(); pool.join(); pool.terminate()
    wall_clock=time.time() - t0_start
    print 'Pool closed/joined'
    print 'path_hub:', path_hub
    if imonit: p_monit.terminate()

    ## if pm was True:
    if pm!=False:
        print '-'*30
        print 'FN_SS1'
        for i in xrange(len(fn_ss1)):
            print fn_ss1[i]
        print '-'*30

    # ----------------------------------------------------------------
    ## 3. Analyze the time spent for each condition
    print '-'*90
    f_time_score = open(os.path.join(
        path_home,'%s_FLD_A_PP_timescore.txt'%tar_date), 'w')

    CPU_time = 0.

    run_times=[]
    for i in xrange(len(results)):
        p, stdo, stde, rt = results[i].get()
        run_times.append(rt)

    ## average time for each processes
    proc_avg_time = np.array(run_times).mean()

    for i in xrange(len(run_times)):
        run_time = run_times[i]
        CPU_time = CPU_time+run_time
        isgn = np.sign(run_time - proc_avg_time)
        if isgn>=1: sgn ='+'
        else:       sgn ='-'
        time_dev = progress_bar.convert_sec_to_string(
            abs(run_time - proc_avg_time))
        run_time = progress_bar.convert_sec_to_string(
            run_time)
        cnt = 'Task-%3.3i: Run time %s in %s // time diff from avg: %s %s'%(
            i+1,run_time,paths_run[i],sgn,time_dev)
        print cnt
        f_time_score.write('%s\n'%cnt)

    cnt ='--'*80+'\n'
    cnt = cnt + 'Total CPU running time        : %40s, %i\n'%(
        progress_bar.convert_sec_to_string(CPU_time), int(CPU_time))
    cnt = cnt + 'WallClock   time              : %40s, %i\n'%(
        progress_bar.convert_sec_to_string(wall_clock), int(wall_clock))
    cnt = cnt + 'NCPU cores                    : %i\n'%ncpu
    cnt = cnt + 'Speed-up (CPU run time/Wall)  : %.3f\n'%(
        CPU_time/wall_clock)
    cnt = cnt + 'Efficiency (Speed-up)/NCPU    : %.1f [pct] \n'%(
        CPU_time/wall_clock/ncpu*100)
    print cnt
    f_time_score.write(cnt)
    f_time_score.close()
    # print '%s has been saved'%f_time_score.name
    print '-'*80

    # ----------------------------------------------------------------
    ## 4. tar the results
    if xy!=None:
        for irho in xrange(len(paths_run)):
            ## Concatenation if xy was given.
            fn2 = os.path.join(paths_run[irho], 'region_a.bin')
            fn_dum = ntf(suffix='_fn_',delete=True).name
            cmd = 'cat %s %s > %s'%(fn1,fn2,fn_dum)
            iflag = os.system(cmd)
            # print 'cmd: ', cmd
            shutil.move(fn_dum,fn2)
            ## raise IOError, 'debug the command...'
    fns=[]
    for irho in xrange(len(paths_run)):
        chdir(paths_run[irho])
        cmd, fn_tar = saveout.main_tar(
            fn_tar=main_fn_tar[irho],
            wild_cards=['*.out','*.OUT','*.bin'],
            iopt=0)
        os.system(cmd)
        shutil.move(fn_tar, path_hub)
        fns.append(fn_tar)
        if irho==0:
            shutil.copy('vpsc7.in',path_hub)

    if pm==False: ## Not postmortem run.
        os.chdir(path_hub)
        ## Collect all tars
        cmd = '%s -czf %s '%(_tar_, fn_FLDA_tar)
        for i in xrange(len(fns)):
            cmd = cmd + '%s '%fns[i].split(os.sep)[-1]
        cmd = cmd +'%s '%'vpsc7.in'

        ## add fn_ss0 to tar file in case that's not None
        if fn_ss0!=None:
            print 'fn_ss0:', fn_ss0
            print 'path_hub:', path_hub
            shutil.copy(fn_ss0,path_hub)
            if prePS!=None:
                cmd = cmd + '%s '%os.path.split(fn_ss0)[-1]
        ## ------
        os.system(cmd)
        ## Collect all tars
        shutil.move(fn_FLDA_tar, path_home)
    chdir(path_home)

    # ----------------------------------------------------------------
    ## 5. Archiving
    ## Pass everthing there.

    #fn_csv = glob('./archive/FLD_A/*/FLD_*/FLD_plot_%s.csv')
    # FLDA_hash_code
    tmp = glob('./archive/FLD_A/*/FLD_%s/vpsc7.in'%FLDA_hash_code)
    if len(tmp)!=1:
        raise IOError, 'Could not find the relevant path'
    else:
        ## may overwrite path_2_archive...
        path_2_archive = os.path.split(tmp[0])[0]

    if pm==False:
        fn_FLDA_tar = os.path.join(
            path_home,os.path.split(fn_FLDA_tar)[-1])
        shutil.copy(fn_FLDA_tar,path2arch)
        shutil.copy(f_time_score.name,path2arch)

        print '-'*50
        fn_ss_temp = 'FLDA-ss-collection_%s.tgz'%(FLDA_hash_code)

        _p_ = os.path.split(fn_ss1[0])[0]
        cmd = 'tar -C %s -czf %s'%(_p_,fn_ss_temp)
        for i in xrange(len(fn_ss1)):
            _fn_ = os.path.split(fn_ss1[i])[-1]
            cmd = '%s %s'%(cmd,_fn_ )
        print cmd
        iflag=os.system(cmd)
        if iflag==0:
            shutil.copy(fn_ss_temp,path_2_archive)
            print 'VPSCFLD-A_ss is saved to: ', \
                os.path.join(path_2_archive,fn_ss_temp)
        else:
            print 'iflag:',iflag
            print '** Error: Could not create fn_ss_temp **'
        print 'VPSC-FLD calculation results are Archived to %s'%path_2_archive

    elif pm!=False:
        print '-'*50
        fn_ss_temp = 'VPSCFLD-A_ss_%s_%s.tgz'%(FLDA_hash_code,pm)
        _p_ = os.path.split(fn_ss1[0])[0]
        cmd = 'tar -C %s -czf %s'%(_p_,fn_ss_temp)
        for i in xrange(len(fn_ss1)):
            _fn_ = os.path.split(fn_ss1[i])[-1]
            cmd = '%s %s'%(cmd,_fn_ )
        print cmd
        iflag=os.system(cmd)
        if iflag==0:
            shutil.copy(fn_ss_temp,path_2_archive)
            print 'VPSCFLD-A_ss is saved to: ', \
                os.path.join(path_2_archive,fn_ss_temp)
        else:
            print 'iflag:',iflag
            print '** Error: Could not create fn_ss_temp **'

        #print 'VPSC-FLD calculation results are Archived to %s'%path_2_archive

    print '-'*50
