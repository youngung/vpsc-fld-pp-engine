"""
VPSC testing module

- Compile / -O0 and -O3
- Run a bunch a examples
"""

"""
Flow charts:
============
1. Compile the code
2. mktemp folders.
3. Prep/Run VPSC examples
4. Run VPSC-FLD examples
  4-1. Run FLD-A 
  4-2. Run FLD-B
"""
import saveout, shutil, os, subprocess, multiprocessing
from os import getcwd, chdir
from tempfile import mkdtemp, mkstemp
from tempfile import NamedTemporaryFile as ntf
pjoin = os.path.join
_tar_ = saveout._tar_

def main(jobs_id=[1,2,3,4,5],exfn='vpsc_ex',ncpu=3):
    """
    Arguments
    =========
    jobs_id = [1,2,3,4,5]
    exfn    = 'vpsc_ex'
    ncpu    = 3
    """
    from fld_pp import rw_FLDb_stream as rw
    from fld_pp import plot_FLDb
    ## compile for optimization 3
    path_home = getcwd()
    ## Try compile
    f         = os.popen('which python')
    py_bin    = f.read().split('\n')[0]; f.close()
    _i_, fstd = mkstemp(dir='/tmp', prefix='stdout_cmp')
    fstd = open(fstd, 'w')
    ## compile usually takes a long time - a few seconds

    print '# Compile -O3 starts'
    proc_cmp = subprocess.Popen(
        ['%s compile.py -o 3'%py_bin],
        stdout=fstd, shell=True)

    print '# Compile -O0 starts'
    if cmp(exfn, o=0)!=0:
        proc_cmp.terminate()
        raise IOError, 'Compilation of the code failed.'
    print '# Compiled successfully for -O0'
    ## prepare the path_run/tar_fn
    paths_run = []; tar_fns = []
    for k in xrange(len(jobs_id)):
        job_id = jobs_id[k]; dd = datetime()
        _path_ = mkdtemp(
            prefix='VPSC-testing-suite_%s_%2.2i_'%(
                dd, job_id), dir='/tmp')
        iflag, tar_fn = examples(job_id, _path_, exfn)
        if iflag==0:
            paths_run.append(_path_)
            tar_fns.append(os.path.split(
                tar_fn)[-1])

    ## extract tar
    for i in xrange(len(paths_run)):
        chdir(paths_run[i])
        iflag = os.system('tar -xf %s'%tar_fns[i])
        if iflag!=0:
            chdir(path_home)
            raise IOError, 'something wrong'
    chdir(path_home)
    proc_cmp.wait(); fstd.close()
    print '# Compilation with -O3 finished'
    for i in xrange(len(paths_run)):
        shutil.copy('vpsc',paths_run[i])

    ## run a batch to submit various jobs in parallel
    if ncpu==None:
        ncpu = multiprocessing.cpu_count()
    from multiprocessing import Pool
    pool = Pool(processes=ncpu)

    results=[]
    for i in xrange(len(paths_run)):
        rst = pool.apply_async(
            func = run_vpsc,
            args = (paths_run[i],))
        results.append(rst)
    print '# Close pool - running %i jobs in parallel'%len(jobs_id)
    pool.close(); pool.join()
    print '# Exit'
    print '# print paths and exit code'
    iex=[]
    for i in xrange(len(results)):
        p, stdo, stde = results[i].get()
        iexit = p.poll();iex.append(iexit)
        print '   --', paths_run[i], '  --  exit code: ', iexit,
        if iexit==0: print ''
        else: print ' Failed'
    chdir(path_home)

    ## --------------------------------------------------
    ## Run FLD-A and FLD-B
    # for i in xrange(len(jobs_id)):
    #     k = jobs_id[i]
    #     if k==5 and iex[i]==0:
    #         job_id = 6 #post-process
    #         print '# successful pre-FLD process available in ',
    #         print paths_run[i]
    #         shutil.copy(paths_run[i]+os.sep+'region_a.bin',
    #                     path_home)
    #         _path_ = mkdtemp(
    #             prefix='VPSC-testing-suite_%s_%2.2i_'%(
    #                 dd, job_id), dir='/tmp')
    #         iflag, tar_fn = examples(job_id, _path_, exfn)
    #         if iflag==0:
    #             tar_fn = os.path.split(tar_fn)[-1]
    #             chdir(_path_)
    #             iflag = os.system('tar -xf %s'%tar_fn)
    #             if iflag!=0:
    #                 chdir(path_home)
    #                 raise IOError, \
    #                     'Could not extract correctly'

    #             print '# running FLDb in' ,_path_
    #             shutil.copy(path_home+os.sep+'vpsc',_path_)
    #             p, stdo, stde = run_vpsc(_path_)
    #             p.wait(); iexit = p.poll()
    #             print '# exit code:', iexit

    #             fout = open(path_home+os.sep+'FLDb.out','w')
    #             rw(fn_stdo=os.path.join(_path_,'stdout.out'),
    #                fn_stde=os.path.join(_path_,'stderr.out'),
    #                f_crit=10, fout=fout)
    #             fout = open(fout.name, 'r')
    #             plot_FLDb(path_home+os.sep+'FLDb.out')
    #         chdir(path_home)
    # ## --------------------------------------------------
    # chdir(path_home)

    # ## Plot Forming Limit Diagrams
    # inp_bk = ntf(prefix='vpsc7_',suffix='_.in_backup',
    #              delete=True,dir = getcwd()).name
    # shutil.move('vpsc7.in',inp_bk)
    # shutil.copy('examples/ex15_FLD/vpsc7.in_proc30','vpsc7.in')

    ## Forming limit for region A
    cmd="python fld.py --d 0.0010 --f 2.5 --i -0.5 --l 1.20"
    cmd = cmd + " --n 7 --r 0.0010 --u 3 --xy '0.2 0.2' --hash ttestt"
    proc_FLDa = subprocess.Popen(
        [cmd],shell=True)
    print '# running FLDa in parallel'
    print '##################################################'
    proc_FLDa.wait()
    print '##################################################'
    print '# FLDa completed'
    dt=datetime().split('_')[0]

    ## Forming limit for region B
    chdir(path_home)
    proc_FLDb = subprocess.Popen(
        ["python FL.py --h0s '0.990' --psi0s '0' "+
         " --ahash ttestt --nu 1 --ncpus 3"],shell=True)
    print '# running FLDb in parallel'
    print '##################################################'
    proc_FLDb.wait()
    print '##################################################'
    print '# FLDb completed'
    chdir(path_home)

def run_vpsc(path_run):
    path_home = getcwd()
    chdir(path_run)
    std_out = open('stdout.out','w')
    std_err = open('stderr.out','w')
    process = subprocess.Popen(
        ['./vpsc'],shell=True,
        stderr=std_err, stdout=std_out,
        stdin=subprocess.PIPE)
    process.wait()
    chdir(path_home)
    return process, std_out, std_err

def examples(job_id, path_run=None, *fns):
    """
    Arguments
    =========
    job_id
    path_run  : if None, create a temp folder
    *fns      : Additional file names if necessary
    """
    path_home = getcwd()
    if path_run==None:
        path_run = mkdtemp(
            prefix='VPSC-testing-suite_%s_JOB%2.2i_'%(
                datetime(),99),
            dir = '/tmp')
    inp_bk = ntf(prefix='vpsc7_',suffix='_.in_backup',
                 delete=True,dir = getcwd()).name
    FLD_nu_bk = ntf(prefix='FLD_nu_',suffix='.in_backup',
                    delete=True,dir = getcwd()).name
    shutil.move('vpsc7.in',inp_bk)
    shutil.move('FLD_nu.in',FLD_nu_bk)

    """
    ## each example function returns a list of files
    needed for that example to be executed correctly.
    """
    if   job_id==1: fn_master = ex02_FCCa()
    elif job_id==2: fn_master = ex02_FCCb()
    elif job_id==3: fn_master = ex03_BCCa()
    elif job_id==4: fn_master = ex03_BCCb()
    elif job_id==5: fn_master = ex15_FLD1()
    elif job_id==6: fn_master = ex15_FLD2()
    else: raise IOError, 'Not ready for the job_id %i'%job_id
    fn_master.append('vpsc7.in')

    for fn in fns: fn_master.append(fn)
    for i in xrange(len(fn_master)):
        if not(os.path.isfile(fn_master[i])):
            print 'File %s has not been found.'%fn_master[i]
            shutil.move(inp_bk,'vpsc7.in')
            shutil.move(FLD_nu_bk,'FLD_nu.in')
            chdir(path_home)
            return -1, None

    ## create a list of files necessary
    ## for the given example job_id
    tar_fn = ntf(
        prefix='ex%2.2i_'%job_id,suffix='.tar',
        delete=True,dir = getcwd()).name
    cmd = 'tar -cf %s'%tar_fn
    for i in xrange(len(fn_master)):
        cmd = '%s %s'%(cmd,fn_master[i])
    iflag = os.system(cmd)
    if iflag!=0:
        print 'Failed to create the tar file'
        shutil.move(inp_bk,'vpsc7.in')
        shutil.move(FLD_nu_bk,'FLD_nu.in')
        chdir(path_home)
        return -1, None
    elif iflag==0:
        shutil.move(tar_fn,path_run)
        shutil.move(inp_bk,'vpsc7.in')
        shutil.move(FLD_nu_bk,'FLD_nu.in')
        chdir(path_home)
        return 0, tar_fn
    return -1, None

def convert_sep(fn,a='\\',b=os.sep):
    f=open(fn,'r');fstr = f.read();f.close()
    fstr = fstr.replace(a,b)
    f=open(fn,'w');f.write(fstr);f.close()

def ex02_FCCa():
    """
    Find/return relevant files necessary
    for the given example (ex02_FCC)
    """
    shutil.copy(pjoin('examples','ex02_FCC','vpsc7.ina'),
                'vpsc7.in')
    shutil.copy(pjoin('examples','ex02_FCC','CUBCOMP.IN'),
                './')
    ## Replace 'path seperator' with os.sep
    convert_sep('vpsc7.in','/',os.sep)
    convert_sep('vpsc7.in','\\',os.sep)
    fn_master=saveout.find_files(fn='vpsc7.in')
    fn_master.append(pjoin('examples','ex02_FCC','rolling'))
    fn_master.append('CUBCOMP.IN')
    return fn_master

def ex02_FCCb():
    shutil.copy(pjoin('examples','ex02_FCC','vpsc7.inb'),'vpsc7.in')
    shutil.copy(pjoin('examples','ex02_FCC','CUBCOMP.IN'),'./')
    convert_sep('vpsc7.in','/',os.sep)
    convert_sep('vpsc7.in','\\',os.sep)
    fn_master=saveout.find_files(fn='vpsc7.in')
    fn_master.append(pjoin('examples','ex02_FCC','lij_hist.dat'))
    fn_master.append('CUBCOMP.IN')
    return fn_master

def ex03_BCCa():
    shutil.copy(pjoin('examples','ex03_BCC','vpsc7.ina'),'vpsc7.in')
    convert_sep('vpsc7.in','/',os.sep)
    convert_sep('vpsc7.in','\\',os.sep)
    fn_master=saveout.find_files(fn='vpsc7.in')
    fn_master.append(pjoin('examples','ex03_BCC','rolling'))
    return fn_master

def ex03_BCCb():
    shutil.copy(pjoin('examples','ex03_BCC','vpsc7.inb'),'vpsc7.in')
    shutil.copy(pjoin('examples','ex03_BCC','CUBCOMP.IN'),'./')
    convert_sep('vpsc7.in','/',os.sep)
    convert_sep('vpsc7.in','\\',os.sep)
    fn_master=saveout.find_files(fn='vpsc7.in')
    fn_master.append(pjoin('examples','ex03_BCC','rolling'))
    fn_master.append('CUBCOMP.IN')
    return fn_master

def ex15_FLD1():
    shutil.copy(pjoin('examples','ex15_FLD','vpsc7.in_proc30'),'vpsc7.in')
    shutil.copy(pjoin('examples','ex15_FLD','FLD_nu.in'),'FLD_nu.in')
    convert_sep('vpsc7.in','/',os.sep)
    convert_sep('vpsc7.in','\\',os.sep)
    fn_master=saveout.find_files(fn='vpsc7.in')
    fn_master.append('FLD_nu.in')
    return fn_master

def ex15_FLD2():
    shutil.copy(pjoin('examples','ex15_FLD','vpsc7.in_proc32'),'vpsc7.in')
    shutil.copy(pjoin('examples','ex15_FLD','FLD_nu.in'),'FLD_nu.in')
    convert_sep('vpsc7.in','/',os.sep)
    convert_sep('vpsc7.in','\\',os.sep)
    fn_master=saveout.find_files(fn='vpsc7.in')
    fn_master.append('FLD_nu.in')
    fn_master.append('region_a.bin')
    return fn_master

def datetime():
    import time
    t = time.asctime()
    tlocal = time.localtime()
    date = time.strftime('%Y%M%d')
    dat, month, dum, time, year = t.split()
    hr, mn, sec = time.split(':')
    month = '%2.2i'%tlocal.tm_mon
    mday  = '%2.2i'%tlocal.tm_mday
    return year+month+mday+'_'+hr+mn+sec

def cmp(exfn='vpsc_ex',o=0):
    """
    Compile the source code and generate
    the executable named as <exfn>

    Arguments
    =========
    exfn = 'vpsc_ex'
    """
    import compile
    iflag = compile.main(
        exefn=exfn,optimization=o,verbose=False)
    if iflag!=0:
        print 'Could not compile vpsc'
        return -1
    return iflag

if __name__=='__main__':
    main(jobs_id=[1,2,3,4],exfn='vpsc_ex',ncpu=3)
