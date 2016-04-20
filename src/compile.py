"""
Compilation script for VPSC

Command line usage:

>>> python compile.py -o 0
>>> python compile.py -o 3
>>> python compile.py -o 3 -r

Arguments
=========
r : flag for run
o : Optimization flag
"""
## VPSC compiling batch using 'gfortran'
import os

def compile_find_rhos():
    filename='bin2out.f'
    cmd = 'gfortran bin2out.f -O3 -o find_rhos'
    iflag = os.system(cmd)
    return iflag

def main(aargs=[],optimization=3,exefn = 'vpsc',
          filename=['src/vpsc_src/*.for','src/vpsc_src/*.f'],
          verbose=True):
    args = ['-finit-local-zero', '-fno-automatic',
            '-fno-align-commons','-finit-integer=zero',
            '-fbackslash','-fbounds-check','-finit-real=zero',
            '-fdefault-double-8','-fdefault-real-8'#]
            ## debug
            ,'-g'
            #,'-Wall'
            # ,'-ffpe-trap=zero,overflow,underflow'
            ,'-fbacktrace']

    args.append('-O%i '%optimization)
    for i in xrange(len(aargs)):
        args.append(aargs[i])
    # cmp_path = os.popen('which gfortran').\
    #            readline().split('\n')[0]
    cmp_path = 'gfortran'
    cmd = '%s '%cmp_path
    if verbose:
        nc = len(cmd)
        header='Compiler path:'
        nc = nc + len(header)
        print '-'*nc,'\n%s%s'%(header, cmd),'\n','-'*nc
        print 'List of arguments to the compiler:'
        for i in xrange(len(args)):
            print '%2.2i %s'%(i+1,args[i])

    for s in args: cmd = cmd+'%s '%s
    for i in xrange(len(filename)):
        cmd = cmd+'%s '%filename[i]

    cmd = cmd+' -o %s'%exefn
    print '-'*20,'\ncmd:\n',cmd,'\n','-'*20
    iflag = os.system('%s'%cmd)
    if iflag==0:
        if verbose: print \
           'Compilation was successful.'
        return 0
    print iflag
    print 'Compilation error, got %i \n\n\n'%iflag
    print 'compile command was:\n>>> %s  \n\n\n'%cmd
    return -1

if __name__=='__main__':
    import subprocess, multiprocessing, time,\
        getopt, sys, os, tempfile
    from MP import progress_bar
    uet = progress_bar.update_elapsed_time
    t0  = time.time()

    try: opts, args = getopt.getopt(sys.argv[1:],'rbo:')
    except getopt.GetoptError, err: print str(err); sys.exit(2)
    run = False; aargs = []; opt = 3; exefn = 'vpsc'

    for o, a in opts:
        if o=='-o': opt=int(a)
        if o=='-r': run = True

    ## Pool
    from multiprocessing import Pool
    pool = Pool(processes = 3)
    ##
    results = []
    rst0 = pool.apply_async(compile_find_rhos)
    results.append(rst0)
    if opt==0:
        rst1 = pool.apply_async(main, args=(aargs,0,exefn,))
        results.append(rst1)
    elif opt!=0:
        #tfn = tempfile.mkstemp(dir='/tmp',prefix='vpsc-exe-')[1]
        tfn='vpsc0'
        rst1 = pool.apply_async(
            main, args=(aargs,0,tfn))

        time.sleep(0.05)
        rst2 = pool.apply_async(main, args=(aargs,opt,exefn,))
        results.append(rst1);results.append(rst2)

    pool.close();rst1.wait()
    pool.terminate()
    uet(time.time()-t0,head='Compilation elapsed time')
    if opt!=0: dats="'%s' has been created"%tfn
    if opt!=0: print '\n'+'='*len(dats),'\nO0 completed'
    if opt!=0: print dats
    if rst1.get()!=0:
        raise IOError, 'Unsuccessful compilation'
    # if opt!=0: os.remove(tfn)

    if opt!=0: print '='*len(dats)
    pool.join()
    uet(time.time()-t0,head='Compilation elapsed time')
    print
    pool.terminate()

    # if iflag!=0: raise IOError, 'Not successful compilation.'
    if run: os.system('./%s'%exefn)
