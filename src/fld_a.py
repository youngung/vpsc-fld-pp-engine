## Single FLD probing engine that allows real-time monitoring..
## dependencies

import os
from os import getcwd,chdir

path_vpsc_home = '/Users/yj/repo/vpsc/vpsc-dev-fld'

chdir(path_vpsc_home)
from fld import write_FLDnu
from MP import progress_bar
from MP.mat import mech
import tempfile, timeit, fld, subprocess, threading,time,shutil, multiprocessing
from tempfile import NamedTemporaryFile as ntf
from tempfile import mkdtemp
from multiprocessing import Process

upr = progress_bar.update_progress
uet = progress_bar.update_elapsed_time
FlowCurve = mech.FlowCurve
path_home = getcwd()

class workerFLDa():
    def __init__(self):
        ## Save the path where the currnet worker is created.
        self.path_home=os.getcwd()
        ## Create the flow history class
        self.fc = FlowCurve()
    def create_working_folder(
            self,rho=0,cvm=1e-3,evm_limit=0.03,
            deij=1e-3,path_work=None):
        """
        Assign worker's duty
        The worker should create 'vpsc7.in' input
        according to the linear path (rho), strain rate (cvm),
        with an increment of <deij>. The worker should keep
        up the duty until <evm_limit> is reached.
        
        Arguments
        =========
        rho
        cvm
        evm_limit
        deij
        path_work
        """
        from fld import fld_pre_probe
        if path_work==None:
            self.path_work= tempfile.mkdtemp(dir='/tmp', 
                                             prefix='VPSC-FLDa-worker-')
        else: self.path_work = path_work
        print self.path_work, ' has been created'
        fld_pre_probe(False,rho, cvm, evm_limit, deij,
                      self.path_work)
    def run(self,bufsize=1):
        """
        Execute VPSC and piping stdout
        
        Argments
        ========
        wait=False
        bufsize=1
        """
        os.chdir(self.path_work)

        ## piping to file object
        ##
        path_stdo = mkdtemp(dir='/tmp',prefix='VPSC-FLDa-stdouts-')
        fn = ntf(dir=path_stdo, suffix='.txt',prefix='VPSC-FLDa-stdo-',delete=True)
        fn = fn.name
        self.stdout   = open(fn,'w')
        self.stdout_r = open(fn,'r')
        ##
        self.fc.epsilon_vm=[]
        self.p = subprocess.Popen(
            ['./vpsc'], shell=True,
            stdout=self.stdout,bufsize=bufsize)
        self.pid=self.p.pid
        self.sigma = []
        self.epsilon = [] 
    def remove(self):
        """
        Caution when removing directories...
        """
        shutil.rmtree(self.path_work)
        print 'path_work: %s has been removed'%self.path_work
        os.chdir(self.path_home)
        print 'Homing to %s'%self.path_home
    def readlines(self,nbuf=10,verbose=False):
        """
        Read <nbuf> number of lines from self.stdout
        
        Arguments
        =========
        nbuf=10
        verbose=False
        """
        for i in range(nbuf):
            line = self.stdout_r.readline().split('\n')[0]
            try:
                d    = line.split('||')
                eps  = np.array(map(float,d[0].split()))
                sig  = np.array(map(float,d[1].split()))
                udot = np.array(map(float,d[2].split()))
            except: pass
            else:
                if verbose: print d
                self.sigma.append(sig)
                self.epsilon.append(eps)
                self.update_fc()
                self.fc.get_eqv()
        
    def set_monitor(self,term_evm=0.1,dt=5,nbuf=10):
        """
        Monitoring stress-strain, if p terminates
        prior to the given term_evm, save/return that value.
        
        Arguments
        =========
        term_evm=0.1
        dt=5
        nbuf=10
        """
        t0=time.time()
        iexit=False
        while not(iexit):
            t1=time.time(); elps = t1-t0
            uet(elps,False,'Total elapsed time')
            self.readlines(nbuf=nbuf)
            try: 
                dum=self.fc.nstp
            except:
                dum=0
            else:
                if self.fc.epsilon_vm[-1]>=term_evm:
                    #self.p.kill()
                    self.p.terminate()
                    print '\nprocess killed due to exceeding term_evm'
                    return self.fc.epsilon_vm[-1]
                iexit = True
            if self.p.poll()!=None:
                print 'process terminated'
                print self.p.poll();iexit=True
                if self.fc.epsilon_vm[-1]<term_evm:
                    print 'VM strain found from pid:%i '%self.p.pid,
                    print 'is lower than the given value!'
                    return self.fc.epsilon_vm[-1]
            if not(iexit): time.sleep(dt)

    def update_fc(self):
        """
        Update flow stress/strain data
        """
        eps,sig = np.array(self.epsilon).T, np.array(self.sigma).T
        self.fc.get_6strain(eps)
        self.fc.get_6stress(sig)
