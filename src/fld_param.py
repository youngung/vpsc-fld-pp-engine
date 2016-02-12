## parametric study on VPSC-FLD
# from fld_pp import untar_wc

import os
import numpy as np
from tempfile import NamedTemporaryFile as ntf

def FLDA_theta(ref_fn='FLa.sh',th0=0,th1=90,nth=6):
    """
    Prepare a set of FLa bash scripts
    based on the refrence FLa.sh script <ref_fn>
    """
    thetas=np.linspace(th0,th1,nth)
    fns = []
    for i in xrange(len(thetas)):
        fn = FLDA_arg(ref_fn,'t',str(thetas[i]))
        fns.append(fn)
    return fns, thetas


def FLDA_BB_pre(ref_fn='FLa.sh',e0=0.0, e1=0.05,ne=6)
    """
    Prepare a set of FLa bash scripts
    based on the refrence FLa.sh script <ref_fn>
    """
    thetas=np.linspace(th0,th1,nth)
    fns = []
    for i in xrange(len(thetas)):
        fn = FLDA_arg(ref_fn,'t',str(thetas[i]))
        fns.append(fn)
    return fns, thetas



def FLDB_hash_A(ref_fn='FLb.sh'):
    """
    Prepare a set of FLb bash scripts
    based on the refrence FLb.sh with changing
    """
    thetas=np.linspace(th0,th1,nth)
    fns = []
    for i in xrange(len(thetas)):
        fn = FLDA_arg(ref_fn,'t',str(thetas[i]))
        fns.append(fn)
    return fns, thetas


def FLDA_rate(ref_fn='FLa.sh',r0=-3,r1=1,nr=3):
    """
    Prepare a set of FLa bash scripts
    based on the refrence FLa.sh script <ref_fn>
    """
    rates=np.linspace(r0,r1,nr)
    rates = 10**rates
    fns = []
    for i in xrange(len(rates)):
        fn = FLDA_arg(ref_fn,'r','%.2e'%rates[i])
        fns.append(fn)
    return fns

def FLDA_arg(ref_fn='FLDa.sh',KEY='t',ARG='0'):
    """
    Prepare a set of FLa bash script
    based on the reference FLa.sh script <ref_fn>

    Arguments
    ---------
    ref_fn = 'FLDa.sh'
    KEY ='t'
    ARG ='0'

    Return
    ------
    fns
    """
    with open(ref_fn,'r') as fo:
        lines = fo.read().split('\n')

    n_cmd_line = None
    for i in xrange(len(lines)):
        l=lines[i]
        if l[:6]=='python':
            n_cmd_line = i
            cmd = l.split('#')[0]
            cmd_arg = cmd.split(
                'fld.py')[1].split('--')[1:]
            keys=[]; args=[]
            for j in xrange(len(cmd_arg)):
                k, a = cmd_arg[j].split()
                if k!=KEY:
                    keys.append(k); args.append(a)

    temp = ntf(delete=True,
               prefix='FLDa_%s_'%KEY,suffix='.sh',
               dir=os.getcwd()).name

    cmd = ''
    with open(temp,'w') as temp_fo:
        for j in xrange(n_cmd_line+1):
            if j!=n_cmd_line:
                temp_fo.write('%s\n'%lines[j])
            elif j==n_cmd_line:
                cmd = '%s %s'%(cmd, 'python fld.py')
                for k in xrange(len(keys)):
                    cmd = '%s %s'%(cmd, ' --%s %s'%(
                        keys[k],args[k]))
                cmd = '%s --%s %s '%(cmd, KEY,ARG)
                print cmd
                temp_fo.write(cmd)
    return temp


def qsub_parse(line):
    """
    Return line from qsub and return jobid (in string)
    """
    el = line.split()
    jobid = el[2]
    return jobid

def main(sleep=60.):
    """
    Run a parametric study for various thetas.

    Your job 2334604 ("FLb.sh") has been submitted
    """
    import time
    import numpy as np

    th0=0; th1=180; nth=13

    FLA_fns, ths = FLDA_theta(
        ref_fn='FLa.sh',th0=th0,th1=th1,nth=nth)

    ## Submit the jobs through FLA_fns
    JobIDs = []
    for i in xrange(nth):
        cmd = 'qsub %s'%FLA_fns[i]
        JobIDs.append(qsub_parse(os.popen(cmd).read()))
        print JobIDs[i], ths[i]
        time.sleep(sleep)

    ## Find FLDa hashes.
    FLDA_hash_codes=np.zeros(nth)
    for i in xrange(nth):
        if FLDA_hash_codes[i]!=0:
            cmd = 'head -n 50 *.sh.o%s'%JobIDs[i]
            lines = os.popen(cmd).read().split('\n')

            for j in xrange(len(lines)):
                if len(lines[j].split())!=2:
                    pass
                elif lines[j].split()[0]=='FLDA_hash_code:':
                    FLDA_hash_codes[i]=lines[j].split()[1]
    print 'Collection of FLDA hash codes:'
    print '-'*30
    for i in xrange(nth):
        print '%5.1f'%ths[i], FLDA_hash_codes[i]
    print '-'*30

    return FLDA_hash_codes

def q_job_status(jobid):
    """
    Arguments
    ---------
    jobid

    Return
    ------
    """
    lines = os.popen('qstat').read().split('\n')
    lines = lines[2:]

    for i in xrange(len(lines)):
        elems = lines[i].split()
        try:
            int(elems[0])
        except: pass
        else:
            _jobid_ = elems[0]
            if jobid==_jobid_:
                return elems[4] ## Return state
    return None
