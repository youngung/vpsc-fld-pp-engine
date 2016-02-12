"""
Combine FLD_B_YYYYMMDD_f0.9xx__*.tar to FLD_B

>>> python fldb.py -d YYYYMMDD -f 0.9xx -r
"""


from glob import glob
import os, time, saveout
tarc = saveout._tar_
from os import remove as rm
from MP import progress_bar
uet = progress_bar.update_elapsed_time
up = progress_bar.update_progress
def main(date='20150226',f0=0.990,remove=False,ahash=None):
    """
    Combine multiple FLD_B_date_f0.990*.tar
    into one.
    """
    from tempfile import NamedTemporaryFile as ntf

    if ahash==None:
        line = 'FLD_B_%s_f%5.3f'%(date,f0)
    if ahash!=None:
        line = 'FLD_B_%s_f%5.3f__AHASH_%s'%(date,f0,ahash)
    fns=glob('%s*.tar'%line)

    if len(fns)==0: return

    for i in xrange(len(fns)):
        print fns[i]

    fn_tars=[]
    t0 = time.time()
    for i in xrange(len(fns)):
        cmd = '%s -xvf %s '%(tarc,fns[i])
        lines = os.popen(cmd).readlines()
        for j in xrange(len(lines)):
            fn= lines[j].split('\n')[0]
            # print fn
            fn_tars.append(fn)
        uet(time.time()-t0)
    print
    fn_tars.sort()
    print 'total number of files:',len(fn_tars)
    fn_all = ''
    for i in xrange(len(fn_tars)):
        fn_all = '%s %s'%(fn_all,fn_tars[i])
        # print fn_tars[i]

    fn=ntf(delete=True,dir=os.getcwd(),
          prefix='FLD_B_%s_f%5.3f_'%(date,f0),
          suffix='.tar').name

    os.system('%s -cvf %s %s'%(tarc,fn,fn_all))
    print fn, 'has been created'

    if remove:
        for i in xrange(len(fns)):
            rm(fns[i])
    uet(time.time()-t0)
    print
    pass

def reduce_size(fn='fld_b_rst_00.tar'):
    """
    Delete everything else 'stdout.out'
    """

    lines = os.popen('%s -tf %s'%(tarc,fn)).readlines()
    for line in lines:
        member = line.split('\n')[0]
        if member!='stdout.out':
            cmd ='%s -f %s --delete  %s'%(tarc,fn,line)

def reduce_size_members(fn='fld_b_rst_20150227_psi00_f0.990.tar'):
    """
    Recursively use reduce_size to remove all members other tahn 'stdout.out'
    """
    lines = os.popen('%s -tf %s'%(tarc,fn)).readlines()
    members=[]
    for line in lines:
        members.append(line.split('\n')[0])
    for member in members:


        os.system('%s -xf %s %s'%(tarc,fn,member))
        os.system('%s -f %s --delete %s'%(tarc,fn,member))
        reduce_size(member)
        os.system('%s -f %s -r %s'%(tarc,fn,member))

rsz = reduce_size_members

if __name__=='__main__':
    print 'Reduce the size of tar files'
    import getopt, sys
    try: opts, args = getopt.getopt(
        sys.argv[1:],
        'd:f:r:a')
    except getopt.GetoptError, err: print str(err); sys.exit(2)

    remove=False
    ahash =None
    for o, a in opts:
        if o=='-d': date      = a
        if o=='-f': f         = float(a)
        if o=='-r': remove    = True
        if o=='-a': ahash     = a

    print 'given date and f:',date, f

    main(date=date,f0=f,remove=remove,ahash=ahash)
