"""
This module is for assisting real-time monitoring of
processes of VPSC running for forming limit strain calculations
based on Marciniak-Kuczynski model.

"""
import fld,time,os,subprocess
import numpy as np
from glob import glob
from fld_pp import principal_strain_2d as calc_prcn_2d
from tempfile import NamedTemporaryFile as ntf
from MP import progress_bar

iflag = os.system('which tail')
if iflag!=0:
    _itail_=False

def Monitor_worker(q):
    obj=q.get()
    obj.run()

def test_gnuplot():
    """
    See if gnuplot is available
    """
    cmd = 'gnuplot --version'
    pipe = os.popen(cmd)
    txt = pipe.read()
    if len(txt)==0:
        return False
    elif 'gnuplot' in txt:
        return True
    else:
        return False

_is_gnuplot_available_ = test_gnuplot()

## Real-time monitoring capability to kill unncessary processes
## in order to improve computational speed for VPSC-FLD runs.
class RealTimeMonitor(object):
    def __init__(self,paths):
        self.paths=np.array(paths)
        self.n_killed=0

        ## Monitoring conditions
        self.dt = 5.           # frequency
        self.timeout=3600.*24. # max timeout (1day as a default)

    def run(self):
        ## determine which is 'run'
        self.t_start = time.time()
        if len(self.paths.shape)==1:
            print 'Running for region A'
            self.run_a()
        if len(self.paths.shape)==3:
            print 'Running for region B'
            self.run_b()

    def run_a(self):
        """
        """
        ## running for Region A
        self.fn_monit = ntf(
            prefix='FLD_A_stats_',
            suffix='.txt',delete=True,dir=os.getcwd())
        self.fn_monit = self.fn_monit.name
        print 'Monitoring file name:',self.fn_monit
        ## running for Region A

        ## initialize flags
        self.n_rho = len(self.paths)
        self.flags = np.zeros(self.n_rho)
        self.eps_a  = np.zeros((self.n_rho,2))

        ## flag= 0: not started
        ## flag= 1: process active
        ## flag=-1: processes that should be terminated
        ## flag=-2: processes that 'are' already terminated.
        ## initial forming limit criterion

        self.pids = np.zeros(self.n_rho)
        self.pids[::] = -1

        t0 = time.time()
        while True:
            time.sleep(self.dt)
            self.check_pids_a()
            self.check_strain_a()
            self.current_status_a()
            if time.time()-t0>self.timeout:
                print 'Timeout'
                return -1  ## timeout
            ## Create/updates folders that are 'active'

    def run_b(self):
        """
        """
        ## running for Region B
        ## Current status updates
        self.fn_monit = ntf(
            prefix='FLD_B_stats_',
            suffix='.txt',delete=True,dir=os.getcwd())
        self.fn_monit = self.fn_monit.name
        print 'Monitoring file name:',self.fn_monit

        self.fn_dat = ntf(
            prefix='FLD_B_DATA_',suffix='.csv',
            delete=True)
        self.fn_dat = self.fn_dat.name
        print 'Data file name:',self.fn_dat

        self.n_fs,self.n_prob,self.n_psi\
            = self.paths.shape
        ## initialize flags
        self.flags  = np.zeros((self.n_fs,self.n_prob,\
                                self.n_psi))
        self.crt_e1 = np.zeros((self.n_fs,self.n_prob))
        ## flag= 0: not started
        ## flag= 1: process active
        ## flag=-1: processes that should be terminated
        ## flag=-2: processes that 'are' already terminated.
        ## initial forming limit criterion
        self.crt_e1[::] = np.inf
        self.pids = np.zeros((self.n_fs,self.n_prob,self.n_psi))
        self.pids[::] = -1

        self.eps_a = np.zeros((self.n_fs,self.n_prob,\
                               self.n_psi,2))
        ## thickness strain ratio.
        self.et_rat = np.ones((
            self.n_fs,self.n_prob,self.n_psi))

        t0 = time.time()
        igraph_start=False
        while True:
            time.sleep(self.dt)
            self.check_pids()
            self.check_strain()
            self.update_crit()
            self.kill_unnecessary()
            self.current_status()

            ##
            if igraph_start==False:
                if (self.flags==-2).any():
                    igraph_start=True
            ## data write and graph?
            if igraph_start and _is_gnuplot_available_:
                self.fld_data_write()
                self.gnu_plot()

            if time.time()-t0>self.timeout:
                print 'Timeout'
                return -1  ## timeout
            ## Create/updates folders that are 'active'
        return time.time()-t0

    def check_pids_a(self):
        """
        """
        # flt_inactive  = self.flags[self.flags==0]
        # flt_no_pids   = self.pids[self.pids==-1]
        # paths_no_pids = self.paths[flt_inactive]

        for i in xrange(len(self.paths)):
            _path_ = self.paths[i]
            wc = os.path.join(_path_,'pid*')
            lines=glob(wc)
            if len(lines)==0:
                ## process is not active yet.
                pass
            elif len(lines)==1 and self.flags[i]==0:
                # print 'lines:', lines,
                pid_filename = os.path.split(
                    lines[0].split('\n')[0])[-1]
                pid = int(pid_filename.split('pid_')[-1])
                self.pids[i]=pid
                self.flags[i]=1
                print '** New Process found; pid:', \
                    pid, ' at %s'%_path_
            elif len(lines)>1:
                raise IOError, \
                    'Multiple number of pid file exist'

            if self.flags[i]==1:
                ## check if the process is active..
                p=subprocess.Popen(
                    ['kill  -s 0 %i'%self.pids[i],],shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
                t0_=time.time()
                while p.poll()==None:
                    time.sleep(0.001)
                    if time.time()-t0_> 3.0:
                        break
                if p.poll()==0: pass
                elif p.poll()==1: ## the pid not exisiting
                    self.flags[i]=-2

    def check_pids(self):
        """
        Pass [i,j,k] of those paths where pid is not found yet.
        """
        for i in xrange(self.n_fs):
            for j in xrange(self.n_prob):
                for k in xrange(self.n_psi):
                    if self.flags[i][j][k]==0:
                        p = self.paths[i][j][k]
                        wc = os.path.join(p,'pid*')
                        lines=glob(wc)
                        if len(lines)==0:
                            ## process is not active yet.
                            pass
                        elif len(lines)==1 and \
                             self.flags[i][j][k]==0:
                            # print 'lines:', lines,
                            pid_filename = os.path.split(
                                lines[0].split('\n')[0])[-1]
                            pid = int(pid_filename.split(
                                'pid_')[-1])
                            self.pids[i][j][k]=pid
                            self.flags[i][j][k]=1
                            print '** New Process found; pid:',\
                                pid,' at %s'%p
                        elif len(lines)>1:
                            raise IOError, 'Multiple number'+\
                                ' of pid file exist'

                    if self.flags[i][j][k]==1:
                        ## check if the process is active..
                        p=subprocess.Popen(
                            ['kill -s 0 %i'%self.pids[i,j,k],],
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
                        t0_=time.time()
                        while p.poll()==None:
                            time.sleep(0.001)
                            if time.time()-t0_> 0.5:
                                break
                        if p.poll()==0: pass
                        elif p.poll()==1: ## the pid not existing
                            self.flags[i][j][k]=-2

    def check_strain(self):
        """
        Check/updates latest strain levels for those paths,
        where process is active.
        """
        for i in xrange(self.n_fs):
            for j in xrange(self.n_prob):
                for k in xrange(self.n_psi):
                    if self.flags[i,j,k]==0: ## inactive process
                        pass
                    elif self.flags[i,j,k]==1: ## active process
                        self.__check_strain__(i,j,k)
                    ## to be terminated process
                    elif self.flags[i,j,k]==-1:
                        pass
                    ## terminated process
                    elif self.flags[i,j,k]==-2:
                        pass
                    else:
                        raise IOError, 'Unexpected case'

    def check_strain_a(self):
        for i in xrange(self.n_rho):
            if self.flags[i]==1:
                self.__check_strain_a__(i)

    def __check_strain__(self,i,j,k):
        """
        """
        fn = os.path.join(self.paths[i][j][k],'stdout.out')
        if not(os.path.isfile(fn)): return
        cmd = 'tail -n 6 %s'%(fn)
        lines = os.popen(cmd).readlines()
        # print cmd
        #print lines
        if len(lines)==0: pass
        elif len(lines)>0:
            for m in xrange(len(lines)):
                l = lines[m]
                try:
                    ## region A
                    elems = map(float,l.split('||')[0].split())
                    nstp, exx, eyy = elems[:3]
                    exy = elems[4]
                    ## region B
                    elems = l.split('||')[-1].split()
                    tsrr = float(elems[9])
                except: pass
                else:
                    e1,e2=calc_prcn_2d(exx,eyy,exy)
                    self.eps_a[i,j,k,0] = e1
                    self.eps_a[i,j,k,1] = e2
                    self.et_rat[i,j,k]  = tsrr

    def __check_strain_a__(self,i):
        """
        """
        fn = os.path.join(self.paths[i],'stdout.out')
        if not(os.path.isfile(fn)): return
        cmd = 'tail -n 6 %s'%(fn)
        lines = os.popen(cmd).readlines()
        # print cmd
        #print lines
        if len(lines)==0: pass
        elif len(lines)>0:
            for m in xrange(len(lines)):
                l = lines[m]
                try:
                    # ## region A
                    elems = map(float,l.split('||')[0].split())
                    exx, eyy, ezz, eyz, exz, exy = elems[:6]
                except: pass
                else:
                    e1,e2=calc_prcn_2d(exx,eyy,exy)
                    self.eps_a[i,0] = e1
                    self.eps_a[i,1] = e2

    def update_crit(self):
        """
        Scan through processes that are finished 'correctly'.
        Change their flags accordingly.

        And see if current forming limit should be updated.
        And if it is necessary, 
        update the forming limit 'on the fly'.
        """
        for i in xrange(self.n_fs):
            for j in xrange(self.n_prob):
                for k in xrange(self.n_psi):
                    if self.flags[i,j,k]==1: ## active process
                        ## check strain again
                        self.__check_strain__(i,j,k)
                        if self.et_rat[i,j,k]>10.:
                            self.flags[i,j,k]=-1
                            if self.eps_a[i,j,k,0] < \
                               self.crt_e1[i,j] \
                               and self.eps_a[i,j,k,0]>0.1:
                                print '-- Update critical '+\
                                    'forming limit at (n_fs,'+\
                                    'n_prob,n_psi)', \
                                    '(',i+1,j+1,k+1,')',
                                print '%5.3f'%self.crt_e1[i,j],\
                                    ' -> ',\
                                    '%5.3f'%self.eps_a[i,j,k,0]
                                self.crt_e1[i,j]=self.eps_a[
                                    i,j,k,0]

    def kill_unnecessary(self):
        """
        Kill a processes if
            1) the process is active but its principal strain
               is beyond the self.crt_e1
            2) ... to be determined - timeout
                increasing thickness?
        """
        for i in xrange(self.n_fs):
            for j in xrange(self.n_prob):
                for k in xrange(self.n_psi):
                    self.__check_strain__(i,j,k)
                    if self.flags[i,j,k]==1 \
                       and self.eps_a[i,j,k,0]>\
                       self.crt_e1[i,j]+0.01:
                        ## kill the process!
                        # print 'Killing a process'
                        p=subprocess.Popen(
                            ['kill','%i'%self.pids[i,j,k]],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

                        self.flags[i,j,k]=-2
                        self.n_killed=self.n_killed+1
                        print 'Killed process %i'%\
                            self.pids[i,j,k],\
                            ' (',i+1,j+1,k+1,')',\
                            'at %s'%self.paths[i,j,k]
                        print ('Its E1 is %5.3f while'+\
                            ' current critical '+\
                               'strain is: %5.3f')%(
                                   self.eps_a[i,j,k,0],
                                   self.crt_e1[i,j])
                        print "Total number of 'killed'"+\
                            " processes: %i"%self.n_killed

    def current_status_a(self):
        """
        Save summary of current status
        """
        ## write infor to self.fn_monit
        flt_to_be_killed = self.flags==-1
        flt_inactive     = self.flags== 0
        flt_terminated   = self.flags==-2

        n_active = len(self.flags[self.flags==1].flatten())
        n_trm    = len(self.flags[self.flags==-2].flatten())
        nx = self.paths.shape[0]
        ntot = nx

        ## remove contents in the file.
        # os.system('> %s'%self.fn_monit)
        with open(self.fn_monit,'w') as f:
            for i in xrange(len(self.flags)):
                if self.flags[i]==1:
                    f.write('%s  %6.3f %6.3f , %7i\n'%(
                        self.paths[i], self.eps_a[i,0],
                        self.eps_a[i,1], self.pids[i]))
            f.write('-'*64+'\n')
            ## write
            f.write(('%3s  '*3+'\n')%('TRM','ACT','TOT'))
            f.write('%3.3i  %3.3i  %3.3i \n'%(
                n_trm,n_active, ntot))
            f.write('%3.1f%% %3.1f%% \n'%(
                float(n_trm)/ntot*100.,
                float(n_active)/ntot*100.))

            f.write('-'*64+'\n')
            elp = progress_bar.convert_sec_to_string(
                time.time() - self.t_start)
            f.write(elp)
            f.write('\n\n\n')

    def current_status(self):
        """
        Save summary of current status

        self.eps_a=np.zeros((self.n_fs,self.n_prob,self.n_psi,2))
        self.flags=np.zeros((self.n_fs,self.n_prob,self.n_psi))
        """
        ## write infor to self.fn_monit
        n_active = len(self.flags[self.flags==1].flatten())
        n_trm    = len(self.flags[self.flags==-2].flatten())
        nx, ny, nz = self.paths.shape
        ntot = nx * ny * nz

        ## remove contents in the file.
        # os.system('> %s'%self.fn_monit)
        with open(self.fn_monit,'w') as f:
            ## Folders, in which processes are 'active'
            for i in xrange(len(self.flags)):
                for j in xrange(len(self.flags[i])):
                    for k in xrange(len(self.flags[i][j])):
                        if self.flags[i][j][k]==1:
                            f.write('%s, %6.3f, %6.3f, %7i \n'%(
                                self.paths[i][j][k],
                                self.eps_a[i,j,k,0],
                                self.eps_a[i,j,k,1],
                                self.pids[i][j][k]))
            f.write('-'*64+'\n')
            ## write
            f.write(('%3s  '*3+'\n')%('TRM','ACT','TOT'))
            f.write('%3.3i  %3.3i  %3.3i \n'%(
                n_trm, n_active, ntot))
            f.write('%3.1f%% %3.1f%% \n'%(
                float(n_trm)/ntot*100.,float(n_active)/ntot*100.))

            f.write('-'*64+'\n')
            elp = progress_bar.convert_sec_to_string(
                time.time() - self.t_start)
            f.write(elp)
            f.write('\n\n\n')

    def fld_data_write(self):
        """
        Write data from 'terminated' processes
        """
        ## data plot
        with open(self.fn_dat,'w') as f:
            # f.write('%5s, %5s\n'%('E2','E1'))
            for i in xrange(len(self.flags)):
                for j in xrange(len(self.flags[i])):
                    for k in xrange(len(self.flags[i][j])):
                        if self.flags[i][j][k]==-2:
                            f.write('%5.2f, %5.2f \n'%(
                                self.eps_a[i,j,k,1],
                                self.eps_a[i,j,k,0]))

    def gnu_plot(self):
        """
        gnuplot -e "FLD='20151005_f0.993_fld.csv'" foo.plg
        """
        f_flg="""## set terminal dumb 
        ## to have command line text-based graph
        set terminal dumb
        set yrange[0:]
        set xrange[-0.5:0.5]
        set xlabel 'E2'
        set ylabel 'E1'
        set size 0.8, 1.0

        plot FLD title 'Forming Limits'
        """
        fn_gnu_plg='fld_gnuplot.plg'
        if not(os.path.isfile(fn_gnu_plg)):
            with open(fn_gnu_plg,'w') as fo:
                fo.write(f_flg)
        cmd = """gnuplot -e "FLD='%s'" %s"""%(
            self.fn_dat,fn_gnu_plg)
        pipe = os.popen(cmd)
        # while p.poll()==None:
        #     time.sleep(0.001)
        graph = pipe.read()
        with open(self.fn_monit,'a') as f:
            f.write(graph)
