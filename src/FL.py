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

## check if NIST-VPSC-tool is properly set.
import matplotlib as mpl
mpl.use('Agg') ## In case X-window is not available.

import fld,time,os,multiprocessing,fld_monit,fld_pp
import numpy as np
from glob import glob
from fld_pp import principal_strain_2d as calc_prcn_2d
from tempfile import NamedTemporaryFile as ntf
import os
func  = fld.fld_vpsc_run

def find_fn_ss0(FLDa_hash):
    """
    See if there is the snapshot inclucded in fn_a_tar.
    If so, return its name
    Otherwise, return 'None'

    Argument
    --------
    FLDa_hash
    """
    fn_a_tar, _theta_ = fld.most_recent_tar(
        iopt=1,FLDa_hash=FLDa_hash)

    # ## See if a snapshot file is contained in fn_a_tar
    cmd = 'tar -tf %s'%fn_a_tar
    elements = os.popen(cmd).read().split('\n')

    n_ss=0;_fn_=[]
    for i in xrange(len(elements)):
        if elements[i][::-1][:3][::-1]=='.ss':
            n_ss = n_ss + 1
            _fn_.append(elements[i])
    fn_ss0=None
    if n_ss==0: pass
    elif n_ss==1:
        print 'There is a snapshot: %s'%_fn_[0]
        cmd ='tar -xf %s %s'%(fn_a_tar,_fn_[0])
        os.system(cmd)
        fn_ss0  = ntf(prefix='VPSC-SNAPSHOT-FLDA-PREP-',
                      suffix='.ss',delete=True,dir=fld._tmp_).name
        shutil.move(_fn_[0],fn_ss0)
    elif n_ss>1:
        raise IOError, 'Too many snapshots in fn_a_tar'
    return fn_ss0 ## either 'None' or file name of the snapshot

### To replace/supplement "fld.py"
if __name__=='__main__':
    """
    ## command line running using main_psi_f0_b in fld.py
    ## Applicable only to region B calculations
    ## various psi0 and f0 conditions
    psi0, n_rhos, f0
    """
    print __doc__
    import vpscTools.parsers.vpsc_parsers as vp_parsers
    import vpscTools.wts_rotate as wts_rotate
    vp_in = vp_parsers.in_parser

    import fld, shutil, saveout, time, tempfile

    from os import getcwd, chdir
    from MP import progress_bar
    import argparse, fld_lib

    uet = progress_bar.update_elapsed_time
    path_home = getcwd()
    main = fld.main
    # hash_code = fld.gen_hash_code(nchar=6)

    ## Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--h0s',type=str,help='a list of f0')
    parser.add_argument(
        '--psi0',type=float,help='psi0')
    parser.add_argument(
        '--psi1',type=float,help='psi1')
    parser.add_argument(
        '--npsi',type=int,help='npsi')
    parser.add_argument(
        '--psi0s',type=str,help='a list of psi0',
        default=None)
    parser.add_argument(
        '--ncpus', type=int,
        help='Number of CPU cores',
        default=multiprocessing.cpu_count() )
    parser.add_argument(
        '--name', type=str,
        help='appending string to the tar file name',
        default='')
    parser.add_argument(
        '--d',type=int,
        help='VPSC-FLD region A date format in YYYYMMDD',
        default=None)
    parser.add_argument(
        '--ahash',type=str,
        help='VPSC-FLD region A HASH tag',
        default=None)
    parser.add_argument(
        '--bhash',type=str,
        help='VPSC-FLD region B HASH tag',
        default=None)
    parser.add_argument(
        '--nu',type=int,
        help='Numerical method (0: N-R; 1: DA)',
        default=0)
    parser.add_argument(
        '--monit',action='store_true',
        help="Turn on Real-Time-Monitor to kill 'unnecessary' processes")

    args        = parser.parse_args()
    f0s         = args.h0s
    print f0s
    print type(f0s)
    print f0s.split()
    print type(f0s.split())
    append_name = args.name
    FLDa_date   = str(args.d)
    f0s         = map(float,f0s.split())
    p           = args.psi0s
    psi0        = args.psi0
    psi1        = args.psi1
    npsi        = args.npsi

    print p, type(p)
    if type(p)==type(None):
        psi0s = np.linspace(psi0,psi1,npsi)
    else:
        psi0s       = map(float,p.split())

    print psi0s

    ncpu        = args.ncpus
    FLDa_hash   = args.ahash
    FLDb_hash   = args.bhash
    NU_opt      = args.nu
    imonit      = args.monit

    if FLDb_hash!=None and len(f0s)>1:
        raise IOError, 'If len(f0s)>1 then FLDb_hash'+\
            ' cannot be manually assigned.'

    if   NU_opt==0: ipro_opt= 32
    elif NU_opt==1: ipro_opt=-32

    # max_active_jobs = args.u
    if FLDa_hash==None:
        raise IOError, 'One should now specify HASH correctly'

    ## determine which FLDa tar file to be used.
    t_fn, FLDa_theta = fld.most_recent_tar(iopt=1,FLDa_hash=FLDa_hash)
    n_probe          = fld_lib.count_paths(FLDA_tar_fn=t_fn)
    print '* INFO: targetted FLDa-tar file:', t_fn
    func      = fld.fld_vpsc_run
    FLDa_date = t_fn.split('fld_a_rst_')[1][:8]

    print 'FLDa_date ', FLDa_date
    print 'FLDa_hash ', FLDa_hash
    print 'FLDa_theta', FLDa_theta

    ## ----------------------------------------
    ## !! use the path_hub from the first *main* run
    path_hub = None
    PATHS    = []; FNAME = []
    t0_start_copy = time.time()
    #FN_SS1 = np.chararray((len(f0s),len(psi0s),n_probe))
    FN_SS1=[];

    # ## See if a snapshot file is contained in fn_a_tar
    fn_ss0 = find_fn_ss0(FLDa_hash)

    for i in xrange(len(f0s)):
        PATHS.append([])
        FNAME.append([])
        FN_SS1.append([])
        for j in xrange(len(psi0s)):
            PATHS[i].append([])
            FNAME[i].append([])
            FN_SS1[i].append([])
            rst = main(
                f0=f0s[i], psi0=psi0s[j],
                FLDa_date=FLDa_date,
                FLDa_hash=FLDa_hash,
                iopt=1,ipro=ipro_opt,
                path_hub=path_hub,
                n_probe=n_probe,fn_ss0=fn_ss0,
                isnapshot1=True) ## isnapshot1 is default

            path_hub, paths_run, main_fn_tar, \
                ngrs, fn_ss1s = rst
            for k in xrange(n_probe):
                # print fn_ss1s[k]
                PATHS[i][j].append(paths_run[k])
                FNAME[i][j].append(main_fn_tar[k])
                FN_SS1[i][j].append(fn_ss1s[k])

    FN_SS1=np.array(FN_SS1,dtype='str')
    # print FN_SS1
    # raise IOError, 'debug'
    ## copy vpsc to path_hub
    shutil.copy(os.path.join(path_home,'vpsc'),path_hub)

    t_elp_copy = time.time() - t0_start_copy
    pool = multiprocessing.Pool(processes=ncpu)
    results=[]
    R = []
    for i in xrange(len(f0s)):
        R.append([])
        for k in xrange(n_probe):
            R[i].append([])
            for j in xrange(len(psi0s)):
                R[i][k].append(
                    pool.apply_async(
                        func=func,
                        args=(PATHS[i][j][k],True,path_hub)))

    ## Start the real-time monitor ------------------------
    if imonit:
        queue=multiprocessing.Queue()
        p_monit = multiprocessing.Process(
            target=fld_monit.Monitor_worker,args=(queue,))
        p_monit.start()
        PATHS_swapped = np.array(PATHS).swapaxes(1,2)
        queue.put(fld_monit.RealTimeMonitor(PATHS_swapped))
        queue.close();queue.join_thread()

    ## Start the main job
    t0_start = time.time()
    pool.close(); pool.join(); pool.terminate()
    wall_clock=time.time() - t0_start
    print 'Pool closed/joined'

    ## Kill real-time monitor -----------------------------
    if imonit: p_monit.terminate()

    ## ----------------------------------------
    ## Print-out time elapse.
    CPU_time = 0.

    print '\n'
    print '-'*84
    print 'F0s/Psi0s/N_probe'
    rts=np.zeros((len(f0s),n_probe,len(psi0s)))
    for i in xrange(len(f0s)):
        for k in xrange(n_probe):
            for j in xrange(len(psi0s)):
                p, stdo, stde, run_time = R[i][k][j].get()
                rts[i,k,j] = run_time
                CPU_time   = CPU_time+run_time




    ## average time for each process:
    proc_avg_time = np.array(rts).flatten().mean()
    cnt = ''
    f_scores=[]
    for i in xrange(len(f0s)):
        f_time_score = open(os.path.join(
            os.getcwd(),'%s_time_scores_%5.3f.txt'%(
                FLDa_date,f0s[i])),'w')
        f_scores.append(f_time_score)
        for k in xrange(n_probe):
            for j in xrange(len(psi0s)):
                run_time = rts[i,k,j]
                isgn = np.sign(run_time - proc_avg_time)
                if isgn>=1: sgn ='+'
                else:       sgn ='-'
                time_dev = progress_bar.convert_sec_to_string(
                    abs(run_time - proc_avg_time))
                run_time = progress_bar.convert_sec_to_string(run_time)
                cnt = cnt + 'Task-%3.3i/%2.2i/%2.2i:'%(i+1,k+1,j+1)+\
                      ' Run time %s in %s // time diff from avg: %s %s\n'%(
                          run_time,PATHS[i][j][k], sgn,time_dev)

        cnt = cnt + '--'*10+'\n'
        cnt = cnt + 'f0s:                          : %s\n'%f0s[i]
        cnt = cnt + 'Number of psi:                : %s\n'%len(psi0s)
        cnt = cnt + 'Number of probed paths:       : %s\n'%n_probe
        cnt = cnt + 'Number of grains:             : %s\n'%ngrs
        cnt = cnt + 'Real-Time-Monitor             : %s\n'%imonit
        cnt = cnt + 'Numerical method (0:NR; 1:DA) : %i\n'%NU_opt
        cnt = cnt + '--'*10+'\n'
        cnt = cnt + "Total time for 'copying'      : %40s , %i\n"%(
            progress_bar.convert_sec_to_string(t_elp_copy),
            int(t_elp_copy))
        cnt = cnt + 'Total CPU running time        : %40s , %i\n'%(
            progress_bar.convert_sec_to_string(CPU_time),
            int(CPU_time))
        cnt = cnt + 'WallClock   time              : %40s , %i\n'%(
            progress_bar.convert_sec_to_string(wall_clock),
            int(wall_clock))
        cnt = cnt + 'NCPU cores                    : %i\n'%ncpu
        cnt = cnt + 'Speed-up (CPU run time/Wall)  : %.3f\n'%(CPU_time/wall_clock)
        cnt = cnt + 'Efficiency (Speed-up)/NCPU    : %.1f [pct] \n'%(
            CPU_time/wall_clock/ncpu*100)

        print cnt
        f_time_score.write(cnt)
        f_time_score.close()
        print '-'*84

    ## gather results!
    print '\n---------------------------------'
    print 'VPSCFLD parallel run in path_hub:'
    print  path_hub
    print 'has been completed'
    ## ----------------------------------------
    FN_TARS=[]

    for i in xrange(len(f0s)):
        f0 = f0s[i]
        if FLDb_hash==None:
            f  = ntf(prefix='FLD_B_%s_f%5.3f_%s_AHASH_%s_'%(
                FLDa_date,f0,append_name,FLDa_hash),
                     suffix='.tgz',delete=True)
            FNTAR = f.name.split(os.sep)[-1]
            FLDb_hash = FNTAR.split('.tgz')[0][::-1][:6][::-1]
        else:
            FNTAR = 'FLD_B_%s_f%5.3f_%s_AHASH_%s_%s.tgz'%(
                FLDa_date,f0,append_name,FLDa_hash,FLDb_hash)

        MEMBERS = ''
        for j in xrange(len(psi0s)):
            psi0 = psi0s[j]
            ## tar filename for the given psi0 and f0.
            fn_tar = 'fld_b_rst_%s_psi%2.2i_f%5.3f.tar'%(
                FLDa_date, psi0, f0)
            members= ''
            for k in xrange(n_probe):
                fn = FNAME[i][j][k].split(os.sep)[-1]
                pa = PATHS[i][j][k]
                chdir(pa)
                ## make a tar that contains both stdout.out and stdout.err
                cmd = '%s -cf %s *.out'%(saveout._tar_,fn)
                members = '%s %s'%(members, fn)
                os.system(cmd)
                ## move the created tar member
                shutil.move(fn, path_hub)

            chdir(path_hub)
            cmd = '%s -cf %s %s'%(saveout._tar_, fn_tar,members)
            os.system(cmd)
            MEMBERS = '%s %s'%(MEMBERS, fn_tar)

            ## delete the members.
            members = members.split()
            for k in xrange(len(members)):
                os.remove(members[k])

        cmd = '%s -czf %s %s'%(saveout._tar_, FNTAR, MEMBERS)
        os.system(cmd)
        shutil.move(FNTAR, path_home)
        print '%s has been moved to home'%FNTAR
        FN_TARS.append(FNTAR)

        print '---------------------------------\n'
        # if len(f0s)>1: raise IOError, 'Have not considered len(f0s)>1 case yet'
        FLDb_hash = FNTAR.split('.tgz')[0][::-1][:6][::-1]

        ## f_time_score.name
        f_time_score=f_scores[i].name
        fn_score = '%s_%s_time_score.txt'%(FLDa_date,FLDb_hash)
        print f_time_score
        print fn_score
        try: shutil.move(f_time_score, os.path.join(path_home,fn_score))
        except: pass
        print '%s has been saved'%fn_score

        ## consider archiving all files here.
        d_arc = os.path.join(path_home,'archive','FLD_A',FLDa_date)
        path2arch = os.path.join(d_arc,'FLD_%s'%(FLDa_hash))
        print 'Archived to:', path2arch
        try: shutil.copy(os.path.join(path_home,FNTAR),       path2arch)
        except: pass
        try: shutil.copy(os.path.join(path_home,fn_score),    path2arch)
        except: pass

        ## FLD_B_args
        fn_arg = os.path.join(path2arch,'FLDB_args_%s'%FLDb_hash)
        with open(fn_arg,'w') as fo:
            fo.write('** %10s %.3f\n'%('f0',f0))
            fo.write('** %10s %.3f\n'%('FLDA_theta',FLDa_theta))
            fo.write('** %10s %s\n'%('git_hash',FLDa_hash))

        ## ## pp and copy it to path2arch
        chdir(path_home)
        fn_csv,figs,ind_psi = fld_pp.pp(bhash=FLDb_hash)
        fn_pdf_fig = 'FLD_plot_%s.pdf'%FLDb_hash
        figs.savefig(fn_pdf_fig,bbox_inches='tight')
        fn_csv_new = 'FLD_plot_%s.csv'%FLDb_hash
        try: shutil.move(fn_csv,fn_csv_new)
        except: pass
        print 'FN_CSV: %s has been created/moved to %s'%(fn_csv_new, path2arch)
        print 'FN_pdf: %s has been created/moved to %s'%(fn_pdf_fig, path2arch)
        try: shutil.copy(os.path.join(path_home,fn_csv_new),path2arch)
        except: pass
        try: shutil.copy(os.path.join(path_home,fn_pdf_fig),path2arch)
        except: pass

        ## Create temp folder.
        print '--'*30
        print 'Generate VPSCFLD-B_ss tar file that contains snapshots pertaining to B region polycrystals'
        path_tf = tempfile.mkdtemp(prefix='VPSCFLD-B_ss_')
        print 'path_tf:', path_tf
        FLDb_ss_fn = 'VPSCFLD-B_ss_%s_%s.tgz'%(FLDa_hash,FLDb_hash)
        print 'FLDb_ss_fn %s'%FLDb_ss_fn
        cmd = 'tar -cf %s'%FLDb_ss_fn

        for k in xrange(n_probe):
        #for k in xrange(len(ind_psi)):
            _fn_=FN_SS1[i,ind_psi[k],k]
            if os.path.isfile(_fn_):
                cmd ='%s %s'%(cmd,os.path.split(_fn_)[-1])
                shutil.copy(_fn_,path_tf)
            else:
                print '(i,j,k): (%i,%i,%i)'%(i,j,k)
                print "Could not find %s file. for ind_psi: %i"%(_fn_,k)

        os.chdir(path_tf)
        print 'cmd for FLDb_ss_fn below:'
        print cmd
        print '--'*30
        iflag=os.system(cmd)
        shutil.copy(FLDb_ss_fn, path2arch)
        chdir(path_home)
        print 'FLDb_ss_fn has been moved to %s'%path2arch

    chdir(path_home)
    print '\n'
    print '-'*40
    print 'Consider using FLD_Postmortem ipynb to further study'
    print 'Yield surface evolution'
    print '-'*40
