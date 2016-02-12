## check the hardening of region A along critical path

## what about region B???

from fld_pp import untar_f_from_tars as untar
from glob import glob
from os import popen, system, sep
import numpy as np


def fns_fld_bcc():
    fns = [
        'archive/FLD_gamma/fld_gam015_6k_f0.990.tar',
        'archive/FLD_sigma/fld_b_rst_20140821_SIGMA2k.tar',
        'archive/FLD_eta/fld_b_rst_20140728_ETA.tar',
        'archive/FLD_alpha/fld_b_rst_20140728_alpha.tar',
        'archive/FLD_iso/FLD_iso06000/FLD_iso6k_f0.990.tar',
        'archive/FLD_Bst/FLD_Bst_02000/old/fld_b_rst_20140717_IFsteel_f0.990.tar'
        ]
    labs=[r'$\gamma$',r'$\sigma$',r'$\eta$',r'$\alpha$','R','F']
    return fns, labs



def tar_list(tarf):
    f=popen('tar -tf %s'%tarf)
    return f.readlines()

def untar_wc(tarf,wc):
    cmd = "tar --wildcards -xvf %s '%s'"%(tarf,wc)
    f=popen(cmd)
    return f.read()

def read_flow(
    fn='archive/FLD_gamma/fld_b_rst_20140728_GAMMA.tar',
    rho=0):
    """
    Return 'Von Mises' flow stress/strain for the
    given tar filename (including full path), and rho`

    Arguments
    =========
    fn  = 'archive/FLD_gamma/fld_b_rst_20140728_GAMMA.tar'
    rho = 0
    """
    from MP.mat.mech import FlowCurve as FC
    fn_remove = []

    ## tar *.fld to check the forming limits
    fn_fld = untar_wc(fn,'*.fld')
    fn_fld = fn_fld.split('\n')[0]
    print '-- ', fn_fld; fn_remove.append(fn_fld)

    ## find critical psi0 for the path (i_rho)
    psi0,i_rho = find_critical_psi0(fn_fld,rho=rho)
    psi0 = '%2.2i'%psi0

    ## find the critical path
    fn_list = tar_list(fn)
    fn_tarb=None
    for i in xrange(len(fn_list)):
        fn_b=fn_list[i].split('\n')[0]
        if fn_b[::-1][:3][::-1]=='out':
            pass
        elif fn_b[:5]=='fld_b':
            if fn_b.split('psi')[1][:2]==psi0:
                fn_tarb = fn_b[::]

    cmd = 'tar -xvf %s %s'%(fn,fn_tarb)
    system(cmd); ## print '--', fn_tarb;fn_remove.append(fn_tarb)

    f=tar_list(fn_tarb)
    f_rho = f[i_rho].split('\n')[0]
    ## print '--', f_rho; fn_remove.append(f_rho)


    cmd = 'tar -xvf %s %s'%(fn_tarb,f_rho)
    system(cmd)
    ## cmd = 'tar -tf %s '%f_rho
    ## system(cmd)
    cmd = 'tar --wildcards -xvf %s */region_a2.out'%f_rho
    f=popen(cmd);
    l= f.readlines()
    region_a2_fn=l[0].split('\n')[0]
    f.close()

    cmd = 'mv %s ./'%region_a2_fn
    system(cmd)

    flow=FC()
    dat=np.loadtxt('region_a2.out').T

    udots=dat[:6];eps=dat[6:12];sig=dat[12:18]

    if dat.ndim==1:
        flow.get_6stress([sig]);flow.get_6strain([eps])
    else:
        flow.get_6stress(sig);flow.get_6strain(eps)

    fn_remove.append('region_a2.out')

    ## delete
    for i in xrange(len(fn_remove)):
        system("rm '%s'"%fn_remove[i])
    system('rm -fr ./tmp')
    flow.get_eqv()
    return flow

def find_critical_psi0(fn,rho):
    exx,eyy,sxx,syy,psi0,psi1=np.loadtxt(fn,skiprows=1).T
    ind = None
    for i in xrange(len(exx)):
        ##print exx[i]
        if rho==1 and exx[i]==eyy[i]:  ## biaxial strain
            ind = i
        elif rho==0 and eyy[i]==0.:  ## PSRD
            ind = i
        elif rho==-0.5:
            if exx[i]!=0:
                if abs(eyy[i]/exx[i]+0.5)<0.001:  ## uniaxial along RD
                    ind = i
            else: pass
        elif rho==2 and exx[i]==0.: ## PSTD
            ind = i
        elif rho==2.5:
            if eyy[i]!=0:
                if abs(exx[i]/eyy[i]+0.5)<0.005:  ## uniaxial along TD
                    ind = i
            else: pass

    if ind==None:
        raise IOError, "Couldn't find the value"

    return psi0[ind],ind

def main(fnout='FC.out',rho=1):
    import matplotlib.pyplot as plt
    from MP.lib.axes_label import __vm__ as vm
    from MP.lib.mpl_lib import wide_fig as wf

    fns=[]

    ## various texture comparison (For the first VPSC-FLD paper prepared for BCC textures)
    """
    Order of appearance:
    gamma, sigma, eta, alpha, epsilon, R, F
    """

    ## Various BCC FLD
    #fns, labs = fns_fld_bcc()


    ## Various intensity comparison for gamma fiber
    # fns.append('archive/FLD_iso/fld_b_rst_20140729_ISO.tar')
    # fns.append('archive/FLD_var_gam/FLD_gam_045/fld_b_rst_20140807_gam_045.tar')
    # fns.append('archive/FLD_var_gam/FLD_gam_030/fld_b_rst_20140806_gam_030.tar')
    # fns.append('archive/FLD_var_gam/FLD_gam_015/fld_b_rst_20140806_gam_015.tar')
    # fns.append('archive/FLD_var_gam/FLD_gam_010/fld_b_rst_20140807_gam_010.tar')
    labs=['R',r'$w_{0}=45^\circ$',r'$w_{0}=30^\circ$',r'$w_{0}=15^\circ$',r'$w_{0}=10^\circ$']

    fig=wf(nh=1,nw=1)
    ax=fig.axes[0]
    # fig=plt.figure(figsize=(5,4))
    # ax=fig.add_subplot(111)

    FCs=[]
    if rho<1: sym='x'
    if rho>1: sym='s'
    if rho==1: sym='x'


    cs=['b','g','r','c','m','y']
    for i in xrange(len(fns)):
        fc=read_flow(fns[i],rho=rho)
        FCs.append(fc)
        l, = plt.plot(fc.epsilon_vm,fc.sigma_vm,
                      color=cs[i],label=labs[i])
        ax.plot(fc.epsilon_vm[-1],fc.sigma_vm[-1],
                sym,mec=l.get_color(),mfc='None')

    ax.legend(loc='best',ncol=3,
              fontsize=8,
              fancybox=True).get_frame().set_alpha(0.5)
    ax.set_title(r'$\rho = $ %3.1f'%rho)
    vm(ax=ax,ft=9)

    return FCs, labs




def ex01():
    from MP.lib.mpl_lib import wide_fig as wf
    from MP.lib.axes_label import __vm__ as vm
    fig = wf(nh=1,nw=3)
    ax=fig.axes[0]
    ax1=fig.axes[1]
    ax2=fig.axes[2]

    ## Read hardening from PSRD
    psrd,labs=main(rho=0)
    ## Read hardening from BB
    bb,labs=main(rho=1)
    ## Read hardening from PSTD
    pstd,labs=main(rho=2)
    ## Read hardening from URD
    urd,labs=main(rho=-0.5)
    ## Read hardening from UTD
    utd,labs=main(rho=2.5)
    #cs=['b','g','r','c','m','y']
    cs=['b','g','r','c','m','y']


    for i in xrange(len(psrd)):
        c = cs[i]
        ## PSRD
        x=psrd[i].epsilon_vm
        y=psrd[i].sigma_vm
        xd = np.diff(x); yd = np.diff(y); h = yd/xd

        ax.plot(x,y,'-',color=c, label=labs[i])
        ax.plot(x[-1],y[-1],'x',color=c)
        ax1.plot(x,y,'-',color=c,label=labs[i])
        ax1.plot(x[-1],y[-1],'x',color=c)

        ax2.plot(x[1:],h,'-',color=c,label=labs[i]);
        ax2.plot(x[-1],h[-1],'x',color=c)

        ## BB
        x=bb[i].epsilon_vm
        y=bb[i].sigma_vm
        xd = np.diff(x); yd = np.diff(y); h = yd/xd

        ax.plot(x,y,'--',color=c)
        ax.plot(x[-1],y[-1],'x',color=c)
        ax1.plot(x,y,'--',color=c)
        ax1.plot(x[-1],y[-1],'x',color=c)

        ax2.plot(x[1:],h,'--',color=c);
        ax2.plot(x[-1],h[-1],'x',color=c)


        ## PSTD
        x=pstd[i].epsilon_vm
        y=pstd[i].sigma_vm
        xd = np.diff(x); yd = np.diff(y); h = yd/xd

        ax.plot(x,y,'-',color=c, label=labs[i])
        ax.plot(x[-1],y[-1],'s',mec=c,mfc='None')
        ax1.plot(x,y,'-',color=c,label=labs[i])
        ax1.plot(x[-1],y[-1],'s',mec=c,mfc='None')

        ax2.plot(x[1:],h,'-',color=c,label=labs[i]);
        ax2.plot(x[-1],h[-1],'s',mec=c,mfc='None')

    x0 = 0.01
    x1 = x0 + 0.15
    x2 = x1 + 0.05
    y0 = 0.95
    y1 = y0 - 0.03
    ax.plot([x0,x1],[y0,y0],'k-',transform=ax.transAxes)
    ax.text(x2,y1,r'Plane Strain ($\rho=0$)',
            fontsize=8,transform=ax.transAxes)

    y0 = 0.85
    y1 = y0 - 0.03
    ax.plot([x0,x1],[y0,y0],'k--',transform=ax.transAxes)
    ax.text(x2,y1,r'Balanced Biaxial ($\rho=1$)',
            fontsize=8,transform=ax.transAxes)

    vm(ax=ax,ft=9)
    vm(ax=ax1,ft=9)
    ax.legend(loc='lower center',fontsize=7,ncol=2,fancybox=True).get_frame().set_alpha(0.5)
    ax1.legend(loc='lower center',fontsize=7,ncol=2,fancybox=True).get_frame().set_alpha(0.5)
    vm(ax=ax2,ft=9)
    ax2.set_ylabel(
        r'$\partial\bar{\Sigma}^{\mathrm{VM}} / \partial\bar{E}^{\mathrm{VM}}$',
        dict(fontsize=9))



