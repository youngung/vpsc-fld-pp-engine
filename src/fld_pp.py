from glob import glob
import numpy,saveout, os, shutil
from numpy import cos, sin, pi
np = numpy
_tar_ = saveout._tar_
pjoin=os.path.join

import fld

try:
    from MP.lib import mpl_lib
    from MP.lib import axes_label
    deco_fld = axes_label.__deco_fld__
    wf=mpl_lib.wide_fig
    colorline = mpl_lib.colorline
except:
    print 'Warning: some of the libraries were not loaded in fld_pp.py'

def main(figs=None,iplot=True):
    if figs==None and iplot: figs=wf(
        nw=4,nh=3,iarange=True,
        w0=0.2,ws=0.60,w1=0.2)
    if iplot: axs=figs.axes

    d = np.loadtxt('udots.out').T
    a = np.loadtxt('region_a2.out').T
    b = np.loadtxt('region_b.out').T

    eps_fl_a = [a[7][-1], a[6][-1]]
    eps_fl_b = [b[7][-1], b[6][-1]]
    sig_fl_a = [a[13][-1] - a[14][-1],a[12][-1] - a[14][-1]]
    sig_fl_b = [b[13][-1] - b[14][-1],b[12][-1] - b[14][-1]]

    psi=d[1]*180./np.pi  ## angle
    psi_f = psi[-1]

    if iplot:
        ## udot
        ax=axs[0]
        ax.plot(d[0],abs(d[10]),'r-',label='region A')
        ax.plot(d[0][-1],abs(d[10][-1]),'rx')
        ax.plot(d[0],abs(d[19]),'k-',label='region B')
        ax.plot(d[0][-1],abs(d[19][-1]),'kx',label='region B')
        ax.set_yscale('log')
        ax.set_xlabel('time')
        ax.set_ylabel('strain rate')

        ## udot_33(B)/udot_33(A)
        ax = axs[1]
        ax.plot(d[0],abs(d[19])/abs(d[10]),'k--')
        ax.plot(d[0][-1],abs(d[19][-1]/d[10][-1]),'kx')

        ax.set_yscale('log')
        ax.set_xlabel('time')
        ax.set_ylabel(r'$|\dot{\bar{E}}_{33}^{B}/\dot{\bar{E}_{33}}^{A}|$')

        ## shear stress/shear strain levels??
        ax = axs[2]

        e11=a[6]; e22=a[7] ; e33=a[8] ;
        e23=a[9]; e13=a[10]; e12=a[11];
        ax.plot(d[0],e12,label='e12')
        ax.plot(d[0],e23,label='e23')
        ax.plot(d[0],e13,label='e13')
        ax.set_xlabel('time'); ax.set_ylabel('shear strains')

        ax = axs[3]
        s11=a[12]; s22=a[13]; s33=a[14];
        s23=a[15]; s13=a[16]; s12=a[17];
        ax.plot(d[0],s12,label='s12')
        ax.plot(d[0],s23,label='s23')
        ax.plot(d[0],s13,label='s13')
        ax.set_xlabel('time'); ax.set_ylabel('shear stresses')

        ## Strain / Stress space
        ## limit strain of A
        ax=axs[4]
        ax.plot(a[7],a[6],'r-')
        ax.plot(a[7][-1],a[6][-1],'rx',label='region A')

        axes_label.__deco_fld__(ax,iopt=2)

        ## limit strain of B
        ax=axs[5]

        ax.plot(b[7],b[6],'k-')
        ax.plot(b[7][-1],b[6][-1],'kx',label='region B')

        axes_label.__deco_fld__(ax,iopt=2)

        ## Limit strain comparison between regions A&B
        ax=axs[6]
        ax.plot(eps_fl_a[0],eps_fl_a[1],'rx',label='region A')
        ax.plot(eps_fl_b[0],eps_fl_b[1],'kx',label='region B')
        axes_label.__deco_fld__(ax,iopt=2)

        ## Limit stress of A
        ax=axs[8]
        ax.plot(a[13]-a[14],a[12]-a[14],'r-',label='region A')
        ax.plot(a[13][-1]-a[14][-1],
                a[12][-1]-a[14][-1],
                'rx',label='region A')

        axes_label.__deco_fld__(ax,iopt=3)

        ## Limit stress of B
        ax=axs[9]
        ax.plot(b[13]-b[14],b[12]-b[14],'k-',label='region B')
        ax.plot(b[13][-1]-b[14][-1],
                b[12][-1]-b[14][-1],
                'kx',label='region A')


        axes_label.__deco_fld__(ax,iopt=3)

        ## Limit stress of both region A&B
        ax=axs[10]
        ax.plot(sig_fl_a[0],sig_fl_a[1],'rx',label='region A')
        ax.plot(sig_fl_b[0],sig_fl_b[1],'kx',label='region B')
        axes_label.__deco_fld__(ax,iopt=3)
        pass

    return figs, eps_fl_a, eps_fl_b, sig_fl_a, sig_fl_b, psi_f

def pre_path(figs=None):
    if figs==None: figs = wf(nw=2)
    axs = figs.axes
    ax=axs[0]
    a = np.loadtxt('region_a.out').T
    ax.plot(a[7],a[6],'k-',label='region A')
    ax=axs[1]
    ax.plot(a[13]-a[14],a[12]-a[14],'k-',label='region A')
    return figs

def fld_():
    figs=wf(nw=2,nh=2)
    axs=figs.axes

    wc=['fld_a_???.out','fld_b_???.out']
    for i in xrange(len(wc)):
        fns=glob(wc[i])

        if i==0: ax0,ax1=axs[2],axs[3]
        if i==1: ax0,ax1=axs[0],axs[1]
        for j in xrange(len(fns)):
            a = np.loadtxt(fns[j]).T
            # if i==0:
            #     e1=a[0]
            #     e2=a[1]
            #     s1=a[6]
            #     s2=a[7]
            # elif i==1:

            try:
                e1=a[6]
                e2=a[7]
                s1=a[12]
                s2=a[13]

                ax0.plot(e2,e1, 'k-')
                ax0.plot(e2[-1],e1[-1], 'kx')
                ax1.plot(s2,s1,'k-')
                ax1.plot(s2[-1],s1[-1],'kx')
            except: pass

        ax1.set_ylim(0.,)
        ax0.set_xlabel(r'$E_2$')
        ax0.set_ylabel(r'$E_1$')
        ax1.set_xlabel(r'$\bar{\Sigma}_2$')
        ax1.set_ylabel(r'$\bar{\Sigma}_1$')

        y = np.linspace(0.,1.)
        x = y*(-0.5)
        ax0.plot(x,y,'--')

        y = np.linspace(0.,1.)
        x = y*(1.0)
        ax0.plot(x,y,'--')

def path_31():
    from shutil import move
    fns = glob('region_a.bin_*')
    for i in xrange(len(fns)):
        move(fns[i],'region_a.bin')
        print fns[i]
        os.system('./bin2out') # to region_a.out
        move('region_a.out', 'region_a_%3.3i.out'%i)
        move('region_a.bin',fns[i])

"""
## plotting diagrams
"""
def plot_flds():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from MP.lib import mpl_lib
    deco = deco_fld
    figs=wf(nw=3,nh=2,left=0.04,right=0.21,
            uw=3.6,uh=3.6,iarange=True,
            w1=0.15,ws=0.65,hs=0.65)
    axes = figs.axes
    rhos = np.arange(-0.5,2.5,0.1)
    fns=glob('*.fld')
    fns.sort()
    for i in xrange(len(fns)):
        fn = fns[i]
        lab = fn.split('.')[0]
        if lab not in ['IF steel','Random']:
            lab=r'$\%s$'%lab.lower()
        dat = np.loadtxt(fn,skiprows=1).T
        x,y = dat[0],dat[1] ## eps_TD and eps_RD
        axes[0].plot(y,x,label=lab)


        i0 = 16;

        for k in xrange(len(x)):
            th = np.arctan2(x[k],y[k])*180./np.pi
            if abs(th-45.)<1e-4:
                i0=k+1
                break

        axes[3].plot(y[:i0],x[:i0],label=lab)
        axes[4].plot(x[i0-1:],y[i0-1:],label=lab)

        x,y=dat[2],dat[3]
        axes[1].plot(y,x)

        # psi0,psi1 = dat[4][:i0], dat[5][:i0]
        # l,=axes[2].plot(rhos[:i0],psi1,'--',alpha=0.5)
        # axes[2].plot(rhos[:i0],psi0,'o',color=l.get_color())



    axes[2].set_ylim(-5.,95)
    axes[2].set_yticks(np.arange(0,90.1,30))


    deco(axes[0],iopt=2,ft=12)
    deco(axes[1],iopt=3,ft=12)
    deco(axes[3],iopt=0,ft=12)
    deco(axes[4],iopt=0,ft=12)

    axes[0].legend(loc='upper left',ncol=2,fontsize=8,
                   fancybox=True).\
                   get_frame().set_alpha(0.5)
    axes[3].legend(loc='upper right',ncol=2,fontsize=8,
                   fancybox=True).\
                   get_frame().set_alpha(0.5)
    # axes[1].legend(loc='best',fontsize=7,
    #                fancybox=True).\
    #                get_frame().set_alpha(0.5)

    draw_guide(axes[0],r_line=[-0.5,0.0,1.0,2.0,2.5],
               max_r=1.5)
    draw_guide(axes[3],r_line=[-0.5,0.0,1.0],
               max_r=1.5)
    draw_guide(axes[4],r_line=[-0.5,0.0,1.0],
               max_r=1.5)
    draw_guide(axes[1],r_line=[0,0.5,1,1.5,2.],
               max_r=1000)

    axes[1].set_xlim(0,800);axes[1].set_ylim(0,800);
    axes[3].set_ylim(0.0,1.0)
    axes[4].set_ylim(0.0,1.0)

def FLD_fs(date='20140717',fs=[0.970,0.980,0.993,0.998]):
    """
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from MP.lib import mpl_lib
    deco = deco_fld
    cmap, m = mpl_lib.norm_cmap(mn=0.,mx  = 90.,cm_name='brg')
    figs=wf(nw=2,nh=1,left=0.08,right=0.13,uw=2.8,uh=2.3,
            down=0.02,up=0.01,h0=0.10,h1=0.10,hs=0.80,
            w0=0.13,w1=0.30,ws=0.57,iarange=True)
    for i in xrange(len(fs)):
        Ea,Pf,figs,ind=FLD(date=date,f=fs[i])
        least_plot(ref_dat=Ea,obj_dat=Ea,psi=Pf,ax=figs.axes[0],iopt=1,m=m)

    deco(figs.axes[0],iopt=0,ft=12)
    draw_guide(figs.axes[0],r_line=[-0.5,0.0,1.0],max_r=1.5)
    figs.savefig('Total_FLD.pdf')
    plt.close('all')


def FLD_thinning(date='20140304',f0=0.990):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from MP.lib import mpl_lib
    deco = deco_fld


    figs=wf(nw=1,nh=1,left=0.08,right=0.13,uw=2.8,uh=2.3,
            down=0.02,up=0.01,h0=0.10,h1=0.10,hs=0.80,
            w0=0.13,w1=0.30,ws=0.57,iarange=True)

    axes = figs.axes

    ## update raw data files
    update_tar_out(date=date,f=f0)
    ## read updated raw data files
    temp = 'fld_b_rst_%s_psi??_f%5.3f.out'%(date,f0)
    fns=glob(temp);fns.sort()

    if len(fns)==0: raise IOError, 'None exist'
    ## initialize forming limit calc sheet
    ft = open('%s_f%5.3f.fld'%(date,f0),'w')
    ft.write('%7s %7s %7s %7s %4s %4s\n'%(
            'Exx','Eyy','Sxx','Syy','psi0','psi1'))
    Ea=[]; Eb=[]; Sa=[]; Sb=[]; Pf=[]; P0=[]
    for i in xrange(len(fns)): ## psi0 / f0
        psi0 = float(fns[i].split('_psi')[1].split('_')[0])
        dat=np.loadtxt(fns[i],skiprows=1)
        eps_a = []; eps_b = []; sig_a = []; sig_b = [];
        psif  = []; psii = []
        for j in xrange(len(dat)): ## along a monotonic path
            ex_a, ey_a, sx_a, sy_a, \
                ex_b, ey_b, sx_b, sy_b, psi = dat[j]
            eps_a.append([ex_a,ey_a]); eps_b.append([ex_b,ey_b])
            sig_a.append([sx_a,sy_a]); sig_b.append([sx_b,sy_b])
            psif.append(psi);psii.append(psi0)

        Ea.append(eps_a); Eb.append(eps_b); Sa.append(sig_a)
        Sb.append(sig_b); Pf.append(psif); P0.append(psii)

    Ea=np.array(Ea); Eb=np.array(Eb); Sa=np.array(Sa);
    Sb=np.array(Sb); Pf=np.array(Pf); P0=np.array(P0);
    Ea=Ea.swapaxes(0,1); Eb=Eb.swapaxes(0,1); Sa=Sa.swapaxes(0,1)
    Sb=Sb.swapaxes(0,1); Pf=Pf.swapaxes(0,1); P0=P0.swapaxes(0,1)

    Pf = 90.-Pf
    P0 = 90. - P0

    exx,eyy,psif_sorted,dum = least_limit(ref_dat=Ea,psi=Pf);
    x_b,y_b,p_b,dum = least_limit(ref_dat=Ea,psi=Pf,obj_dat=Eb)

    fs=[]
    for i in xrange(len(exx)):
        eps_xx_A, eps_yy_A = exx[i], eyy[i]
        eps_xx_B, eps_yy_B = x_b[i], y_b[i]

        e33_B = -eps_xx_B - eps_yy_B
        e33_A = -eps_xx_A - eps_yy_A
        f = f0*np.exp(e33_B - e33_A)
        fs.append(f)

    norm = mpl.colors.Normalize(vmin=min(fs), vmax=max(fs))
    cmap, m = mpl_lib.norm_cmap(
        mn=min(fs),mx  = max(fs),cm_name='brg')
        ## 'winter','jet','copper','summer','gist_rainbow'
    for i in xrange(len(exx)):
        axes[0].plot(
            eyy[i],exx[i],'x',
            c=m.to_rgba(fs[i]))

    b    = axes[0].get_position()
    h    = b.y1-b.y0 ;h  = h * 0.7
    h0   = b.y0      ;h0 = h0 + 0.03
    axcb = figs.add_axes([0.80,h0,0.03,h])
    cb = mpl_lib.add_cb(axcb,cmap=cmap,
                        filled=True,norm=norm,
                        ylab=r'$f$',
                        format=r'$%5.3f$')
    deco_fld(axes[0],iopt=2,ft=12)
    axes[0].set_xlim(-0.5,1.0);
    axes[0].set_ylim(-0.5,1.0)

    draw_guide(
        axes[0],
        r_line=[-0.5,0.,1,2.0,2.5],
        max_r=2.)

def principal_strain_2d(exx,eyy,exy):
    """
    Calculates principal strains.

    Arguments
    ---------
    exx,eyy,exy

    Return
    -------
    e1,e2
    """
    if exy==0:
        if exx>=eyy:
            e1 = exx
            e2 = eyy
        else:
            e1 = eyy
            e2 = exx
        return e1,e2
    else:
        A  = (exx+eyy)/2.
        B  = np.sqrt(((exx-eyy)/2)**2.+exy**2)
        e1 = A+B
        e2 = A-B
        return e1,e2

def pp(bhash='NlJ6yA'):
    """
    Minimalistic FLD-pp

    ## extract the tar/tgz file in /tmp/
    ## then do 'FLD' analysis there.

    Arguemnt
    ---------
    bhash - HASH code for FLD_B tar/tgz file

    Return
    ------
    csv_fn, figs, ind_psi
    """
    import tempfile
    path_home = os.getcwd()
    wc='FLD_B_*%s.t*'%bhash
    fns=glob(wc)
    if len(fns)!=1:
        raise IOError
    fn=fns[0]
    date=fn.split('_')[2]
    f=float(fn.split('_')[3].split('f')[-1])

    path_run = tempfile.mkdtemp(dir='/tmp')
    print 'fn      :',fn
    print 'path_run:',path_run
    shutil.copy(pjoin(path_home,fn),path_run)

    os.chdir(path_run)
    cmd = 'tar -xf %s'%fn
    print 'cmd: ', cmd
    os.system(cmd)
    Ea, Pf, figs, ind_psi = FLD(date=date,f=f,iplot=True,verbose=False)
    ## copy csv file
    csv_fn = pjoin(path_run,'%s_f%5.3f_fld.csv'%(date,f))
    shutil.copy(csv_fn,path_home)
    print '-'*60
    print 'csv File created:', csv_fn
    os.chdir(path_home)
    return csv_fn, figs, ind_psi


def FLD(date='20140714',f=0.990,rot=False,dir=None,
        iplot=False,verbose=True):
    """
    Collectively plot all results possible generated
    from FLDa/FLb simulations using VPSC-FLD

    Arguments
    =========
    date    = '20140714'
    f       = 0.990
    rot     = False
    dir     = None
    iplot   = False
    verbose = True

    Returns
    =======
    Ea, Pf, figs, ind_Ep_a_sorted
             (Strain pertaninig to region A,
             and final necking angle, mpl.fig, and
             index array for necking angle)
    """
    if rot:
        raise IOError, "Use of 'rot' is banned."

    from matplotlib.ticker import MaxNLocator
    path_home=os.getcwd()
    if dir!=None:
        os.chdir(dir)
        print 'Current working direction is: ', os.getcwd()

    if iplot:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from MP.lib import mpl_lib
        from matplotlib import gridspec
        GS   = gridspec.GridSpec
        deco = deco_fld
        norm = mpl.colors.Normalize(vmin=0., vmax=90.)
        cmap, m = mpl_lib.norm_cmap(mn=0.,mx  = 90.,cm_name='brg')
        uf=4.5
        figsize=(2.0*uf,1.5*uf)
        figs=plt.figure(figsize=figsize)
        u = 10;dx,dy = figsize[0]*u, figsize[1]*u
        gs=GS(40,36,top=0.95,bottom=0.05,wspace=0,hspace=0)
        for iy in xrange(4):
            for ix in xrange(2):
                x0,x1 = (u)*ix, (u)*(ix+1)-5
                y0,y1 = (u)*iy, (u)*(iy+1)-3
                # print 'x0,x1:',x0,x1, '     y0,y1:',y0,y1
                figs.add_subplot(gs[y0:y1,x0:x1])
        figs.add_subplot(gs[0:7,25:30])
        figs.add_subplot(gs[10:17,25:30])
        figs.add_subplot(gs[20:27,25:30])
        figs.add_subplot(gs[30:37,25:30])
        axes = figs.axes
        for ax in axes:
            ax.tick_params(axis='both', which='major',labelsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.minorticks_on()

    ## update raw data files
    _n_ = update_tar_out(date=date,f=f,verbose=verbose)
    ## read updated raw data files
    temp = 'fld_b_rst_%s_psi??_f%5.3f.out'%(date,f)
    fns = glob(temp);fns.sort()

    dat_master=[]; n_rhos=0

    ## Principal strains (E1/E2)
    Eprcn_a=[]; Eprcn_b=[]
    Ea=[]; Eb=[]; Sa=[]; Sb=[]; Pf=[]; P0=[]; RHOA=[];RHOB=[]
    for i in xrange(len(fns)): ## psi0 / f0
        psi0 = float(fns[i].split('_psi')[1].split('_')[0])
        dat = np.loadtxt(fns[i],skiprows=1)
        prcn_Ea=[]; prcn_Eb=[]; sig_a = []; sig_b = [];
        eps_a = []; eps_b = []; psif  = []; psii = []; rhos = []
        rhos_b = []
        for j in xrange(len(dat)): ## along a monotonic path
            if i==0: n_rhos=n_rhos+1
            ex_a, ey_a, exya, sx_a, sy_a, sxya,\
                ex_b, ey_b, exyb, sx_b, sy_b, sxyb, psi, rho, rho_b = dat[j]
            ## Option: swap x/y and psi (caution required)
            ## Calculates principal strains.
            e1_a, e2_a = principal_strain_2d(ex_a,ey_a,exya)
            e1_b, e2_b = principal_strain_2d(ex_b,ey_b,exyb)
            prcn_Ea.append([e1_a,e2_a])
            prcn_Eb.append([e1_b,e2_b])
            eps_a.append([ex_a,ey_a,exya]); eps_b.append([ex_b,ey_b,exyb])
            sig_a.append([sx_a,sy_a,sxya]); sig_b.append([sx_b,sy_b,sxyb])
            psif.append(psi);psii.append(psi0)
            rhos.append(rho)
            rhos_b.append(rho_b)

        Ea.append(eps_a); Eb.append(eps_b); Sa.append(sig_a)
        Sb.append(sig_b); Pf.append(psif); P0.append(psii)
        RHOA.append(rhos)
        RHOB.append(rhos_b)
        ## Principal strain
        Eprcn_a.append(prcn_Ea)
        Eprcn_b.append(prcn_Eb)

        if iplot:
            c = m.to_rgba(psi0)
            axes[0].plot(np.array(eps_a).T[1],
                         np.array(eps_a).T[0],
                         '-',color=c)
            for j in xrange(len(np.array(eps_a).T[1])):
                axes[0].plot(np.array(eps_a).T[1][j],
                             np.array(eps_a).T[0][j],
                             '.',color=c,ms=3)
            axes[1].plot(np.array(sig_a).T[1],
                         np.array(sig_a).T[0],
                         '-',color=c)
            for j in xrange(len(np.array(sig_a).T[1])):
                axes[1].plot(np.array(sig_a).T[1][j],
                             np.array(sig_a).T[0][j],
                             '.',color=c,ms=3)

    Eprcn_a=np.array(Eprcn_a); Eprcn_b=np.array(Eprcn_b)
    Ea=np.array(Ea); Eb=np.array(Eb); Sa=np.array(Sa);
    Sb=np.array(Sb); Pf=np.array(Pf); P0=np.array(P0);
    RHOA=np.array(RHOA); RHOB=np.array(RHOB)


    ## rearrange shape of array
    Eprcn_a=Eprcn_a.swapaxes(0,1); Eprcn_b=Eprcn_b.swapaxes(0,1)
    Ea=Ea.swapaxes(0,1); Eb=Eb.swapaxes(0,1); Sa=Sa.swapaxes(0,1)
    Sb=Sb.swapaxes(0,1); Pf=Pf.swapaxes(0,1); P0=P0.swapaxes(0,1)
    RHOA=RHOA.swapaxes(0,1); RHOB=RHOB.swapaxes(0,1)

    ## find indices that give least forming limits
    if iplot==False:
        ## I should add Eprcn_a/Eprcn_b
        os.chdir(path_home)
        return Ea, Eb, Sa, Sb, Pf, P0 ## for 'write_stress_paths'

    ## Exx/Eyy
    least_plot(ref_dat=Ea,     psi=Pf,                  ax=axes[4], m=m, iopt=0)
    least_plot(ref_dat=Ea,     obj_dat=Sa,      psi=Pf, ax=axes[5], m=m, iopt=0)
    # least_plot(ref_dat=Ea,     obj_dat=Ea,      psi=Pf, ax=axes[6], m=m, iopt=1)
    least_plot(ref_dat=Ea,     obj_dat=Sa,      psi=Pf, ax=axes[7], m=m, iopt=1)
    ## E1/E2
    least_plot(ref_dat=Eprcn_a,obj_dat=Eprcn_a, psi=Pf, ax=axes[6], m=m, iopt=0)


    ## region A/B with E_xx/E_yy
    exx_a,eyy_a,psi0_sorted,ind_Ea_sorted = least_limit(ref_dat=Ea,psi=P0);
    exy_a = sort_ind(Ea[:,:,2], ind_Ea_sorted)
    psif_sorted = sort_ind(Pf,ind_Ea_sorted)
    exx_b = sort_ind(Eb[:,:,0], ind_Ea_sorted)
    eyy_b = sort_ind(Eb[:,:,1], ind_Ea_sorted)
    exy_b = sort_ind(Eb[:,:,2], ind_Ea_sorted)
    sxx_a = sort_ind(Sa[:,:,0], ind_Ea_sorted)
    syy_a = sort_ind(Sa[:,:,1], ind_Ea_sorted)
    sxy_a = sort_ind(Sa[:,:,2], ind_Ea_sorted)
    sxx_b = sort_ind(Sb[:,:,0], ind_Ea_sorted)
    syy_b = sort_ind(Sb[:,:,1], ind_Ea_sorted)
    sxy_b = sort_ind(Sb[:,:,2], ind_Ea_sorted)

    ## region A/B with E_1/E_2 (principal strains)
    e1_a,e2_a,psi0_sorted_by_Ep_a,ind_Ep_a_sorted = \
        least_limit(ref_dat=Eprcn_a,psi=P0);
    e1_b = sort_ind(Eprcn_b[:,:,0], ind_Ep_a_sorted)
    e2_b = sort_ind(Eprcn_b[:,:,1], ind_Ep_a_sorted)
    psif_sorted_by_Ep_a = sort_ind(Pf, ind_Ep_a_sorted)
    rho_sorted          = sort_ind(RHOA,ind_Ep_a_sorted)
    rho_sorted_b        = sort_ind(RHOB,ind_Ep_a_sorted)

    print ind_Ep_a_sorted

    axes[2].plot(eyy_a,exx_a,'kx',label = 'Region A')
    axes[2].plot(eyy_b,exx_b,'k.',label='Region B')
    axes[3].plot(syy_a,sxx_a,'kx')
    axes[3].plot(syy_b,sxx_b,'k.')
    axes[11].plot(e2_a,e1_a,'kx',label='Region A')
    axes[11].plot(e2_b,e1_b,'k.',label='Region B')

    ## initialize forming limit calc sheet
    fmt_string = '%7s, '*19+'%7s\n'
    fmt_dat    = '%7.3f, '*19+'%7.3f\n'
    with open('%s_f%5.3f_fld.csv'%(date,f),'w') as ft:
        ft.write(fmt_string%(
            'E2_a','E1_a','Eyy_a','Exx_a','Exy_a',
            'E2_b','E1_b','Eyy_b','Exx_b','Exy_b',
            'Syy_a','Sxx_a','Sxy_a',
            'Syy_b','Sxx_b','Sxy_b',
            'Psi0','Psif','rhoA','rhoB'))
        for i in xrange(len(e2_a)):
            ft.write(fmt_dat%(
                e2_a[i],e1_a[i],eyy_a[i],exx_a[i],exy_a[i],
                e2_b[i],e1_b[i],eyy_b[i],exx_b[i],exy_b[i],
                syy_a[i],sxx_a[i],sxy_a[i],
                syy_b[i],sxx_b[i],sxy_b[i],
                psi0_sorted_by_Ep_a[i],
                psif_sorted_by_Ep_a[i],
                rho_sorted[i],rho_sorted_b[i]))

    ## writing the stress paths
    ## use save_b_tar and track the Stress/strain line.
    try: write_stress_paths()
    except: pass

    axes[2].legend(
        loc='best',fancybox=True,numpoints=1,ncol=1,
        fontsize=7).get_frame().set_alpha(0.5)
    opts=[2,3,2,3,2,3,0,1,3,2,0];fs=9;iasp=False
    for i in xrange(len(figs.axes)-1):
        deco_fld(figs.axes[i],iopt=opts[i],ft=fs,iasp=iasp)

    # ## x/y limits
    axes[0].set_xlim(-0.5,1.0);axes[0].set_ylim(-0.5,1.0)
    axes[2].set_xlim(-0.5,1.0);axes[2].set_ylim(-0.5,1.0)
    axes[4].set_xlim(-0.5,1.0);axes[4].set_ylim(-0.5,1.0)
    axes[3].set_xlim(-300.,800);  axes[3].set_ylim(-300,800)
    axes[1].set_xlim(-300.,800);  axes[1].set_ylim(-300,800)
    axes[5].set_xlim(-300.,800);  axes[5].set_ylim(-300,800)
    axes[7].set_xlim(-300.,800);  axes[7].set_ylim(-300,800)
    axes[8].set_xlim(-300.,800);  axes[8].set_ylim(-300,800)
    axes[6].set_xlim(-0.5,0.5);axes[6].set_ylim(0.0,1.0)
    ix0,ix1=0,7;iy0,iy1=16,17;labf=[r'$\psi_0$',r'$\psi_f$']
    for i in xrange(2):
        axcb = figs.add_subplot(gs[ix0:ix1,iy0:iy1])
        cb = mpl_lib.add_cb(axcb,cmap=cmap,
                            filled=True,norm=norm,
                            ylab=labf[i],
                            format=r'$%2i^\circ$')
        cb.set_ticks(np.arange(0.,90.01,30.))
        axcb.set_frame_on(False);axcb.get_xaxis().set_visible(False)
        ix0,ix1=ix0+20,ix1+20

    for i in xrange(8):
        if np.mod(i,2)==0:
            r_line = [-0.5,0,1,0,2,2.5];max_r = 1.5
        else:
            r_line = [ 0.0,0.5,1.0,1.5,2.0];max_r = 1000
        draw_guide(axes[i],r_line=r_line,max_r=max_r)

    ## Plot stress/strain paths
    i_rhos=np.arange(n_rhos)
    #i_rhos=[0,1,2,3]
    fn1,fn2,fn3,fn4,fn5 = _wsp_main_(
        Ea,Eb,Sa,Sb,Pf,P0,os.getcwd(),i_rhos=i_rhos,
        date=date,f0=f,dir='.')
    x,y=_wp_reader_(fn1) ## stress path
    for i in xrange(len(x)):
        axes[8].plot(y[i],x[i],'-',color='gray')
    x_,y=_wp_reader_(fn2) ## strain path
    for i in xrange(len(x_)):
        axes[9].plot(y[i],x_[i],'-',color='gray')
    x,dum=_wp_reader_(fn3)
    for i in xrange(len(x)): ## E11 vs psif
        axes[10].plot(x_[i],x[i],'-',color='gray')
    axes[10].set_xlabel(r'$\bar{E}_{1}$')
    axes[10].set_ylabel(r'$\psi$')

    os.chdir(path_home)

    return Ea, Pf, figs, ind_Ep_a_sorted

def least_plot(ref_dat=None,obj_dat=None,
               psi=None,ax=None,m=None,iopt=None):
    """
    Arguments
    =========
    ref_dat : data array that is used as a reference
              to find a least limit. When obj_dat is 'None',
              the returned least limit is found in ref_dat
    obj_dat : When not a 'None', the least limit based on
              the sorted obj_dat is returned.
    psi
    ax
    m
    iopt (0: preserve RD/TD convention)
         (1: follow major/minor convention)
    """
    x, y, p, ind = least_limit(ref_dat,psi,obj_dat)
    for i in xrange(len(x)):
        c=m.to_rgba(p[i])
        X=x[i];Y=y[i] ## x//RD y//TD
        sym='+'
        if X==Y: sym ='o'
        if X>Y:  sym ='x'
        if Y>X:  sym ='s'

        if iopt==0:
            pass
        elif iopt==1 :
            if X<Y:
                dum = X
                X = Y
                Y = dum
                sym='s'
                pass
        ax.plot(Y,X,sym,mfc='none',ms=5,linewidth=5.,
                mec=c)#mec=c,mfc=0,color=c)

def sort_ind(a,ind):
    """
    """
    new_array = []
    n_path, n_ang = a.shape
    for i in xrange(n_path):
        i0 = ind[i]
        new_array.append(a[i][i0])
    return np.array(new_array)

def least_limit(ref_dat,psi,obj_dat=None):
    """
    Arguments
    ---------
    ref_dat : data array that is used as a reference
              to find a least limit. When obj_dat is 'None',
              the returned least limit is found in ref_dat
    obj_dat : When not a 'None', the least limit based on
              the sorted obj_dat is returned.
    psi
    """
    ## in case obj_dat is not given:
    if type(obj_dat)==type(None): obj_dat=ref_dat[::]

    from MP import ssort
    sort = ssort.shellSort
    X,Y,P=[],[],[]
    npaths=len(ref_dat)
    ind_min_paths = [] ## minum element indices collected for all paths.
    for i in xrange(npaths):
        R = []
        for j in xrange(len(ref_dat[i])): ## psi0/f0
            x = ref_dat[i][j][0]
            y = ref_dat[i][j][1]
            R.append(r(x,y))
        ## in order to discard nan, nan is replaced with
        ## an abnormally high value (1e10)
        for k in xrange(len(R)):
            if np.isnan(R[k]):
                R[k]=1e10
        val,ind = sort(R)
        i_mn = ind[0]
        ind_min_paths.append(i_mn)
        x=obj_dat[i][i_mn][0]
        y=obj_dat[i][i_mn][1]
        p=psi[i][i_mn]
        X.append(x)
        Y.append(y)
        P.append(p)
    return X,Y,P,ind_min_paths

def update_tar_out(date,f=0.980,verbose=True):
    """
    Find a tar archive generated by VPSC-FLD

    Arguments
    =========
    date = 'YYYYMMDD'
    f    = 0.980
    """
    fns=glob('fld_b_rst_%s_psi??_f%5.3f.tar'%(date,f))
    itgz=False
    if len(fns)==0:
        ## try with 'tgz'
        itgz=True
        fns=glob('fld_b_rst_%s_psi??_f%5.3f.tgz'%(date,f))
        if len(fns)==0:
            print 'Could not find any *.tar or *.tgz'
            return 0

    fns.sort()
    for i in xrange(len(fns)):
        fn = '%s%s'%(fns[i].split('.tar')[0],'.out')
        if os.path.isfile(fn):
            if verbose: print 'Existing %s'%fn
        else:
            import shutil, tempfile
            print 'Saving %s'%fn
            path_home = os.getcwd()
            _path_ = tempfile.mkdtemp(dir='/tmp')
            shutil.copy(fns[i],_path_)
            os.chdir(_path_)
            save_b_tar(fns[i],verbose=verbose)
            os.chdir(path_home)
            os.system('cp %s%s*.out ./'%(
                _path_,os.sep))
            ## are we removing the temp folder?

    return len(fns)

def rw_FLDb_stream(
        fn_stdo = 'stdout.out',
        fn_stde = 'stderr.out',
        f_crit  = 10,
        fout = None):
    """
    Read streams of information
    printed from VPSC-FLD to stdout.out
    from the simulations conducted for region B.

    Arguments
    =========
    fn_stdo = 'stdout.out'
    fn_stde = 'stderr.out'
    f_crit  = 10
    fout    = None: if None is given, create FLDb.out
                    and save all data there
    """
    if fout==None: fout=open('FLDb.out','w')

    fmt = ''
    for i in xrange(9): fmt = '%s %s'%(fmt, '%10s')
    header = fmt%('Ex','Ey','Sx','Sy','Ex','Ey',
                  'Sx','Sy','psi')
    ## print header
    fout.write(header+'\n')
    fmt = ''
    for i in xrange(9): fmt = '%s %s'%(fmt, '%10.5f')

    l  = os.popen('cat %s'%fn_stdo)
    el = os.popen('cat %s'%fn_stde)
    ed = el.readlines()
    d  = l.readlines() ## stdout
    d  = d[::]

    found = False
    il = 0
    ifail = False
    while not(ifail):
        found = True
        try:
            a = d[il].split('||')[0]; b = d[il].split('||')[1]
        except:
            found = False

        else:
            ## if the line contains '||'
            adat=a.split()
            bdat=b.split()
            f = float(bdat[9])
            # print il, f

            # (abs(f) - abs(f_crit*100) > 1e2 ):
            if f>f_crit and abs(f)>abs(f_crit*1.5):
                print "Final thickness strain rate",\
                    " seems unreliable. ",
                print "Found ratio was %3.1f,"%(f), \
                    " which exceeds the criterion",\
                    " (%3.1f) by far "%f_crit
                ifail = True

            if not(ifail):
                ex_a = float(adat[1]); ey_a = float(adat[2])
                sx_a = float(adat[5]); sy_a = float(adat[6])
                ex_b = float(bdat[0]); ey_b = float(bdat[1])
                sx_b = float(bdat[4]); sy_b = float(bdat[5])
                psi  = float(bdat[10])
                dat = fmt%(ex_a,ey_a,sx_a,sy_a,
                           ex_b,ey_b,sx_b,sy_b,psi)
                fout.write(dat)
                fout.write('\n')

        il = il +1
        if il>len(d):
            print 'End of line'
            ifail = True; found = True

    fout.close()

def nt_xycoords(psi,xy,r):
    """
    Calculates the normal and transverse lines

    Arguments
    =========
    psi, xy, r

    Returns
    =======
    n, t
    """
    n, t = nt(psi)
    ## Extends the length of the unit vectors
    t = t * r; n = n * r
    ## Translates t and n vectors to the x,y
    for i in xrange(2):
        t[i] = t[i] + xy[i]; n[i] = n[i] + xy[i]

    return n, t

def nt(psi):
    """
    Calculates the normal and the transverse
    directions of the band according to fld.f/ subr. nt_psi

    Arguments
    =========
    psi

    Return
    =========
    n, t
    """
    n = np.zeros((3,))
    t = np.zeros((3,))
    th = psi * pi/180.
    cth = cos(th)
    sth = sin(th)

    n[0] = cth
    n[1] = sth
    t[0] = sth
    t[1] = -cth
    return n, t

def plot_FLDb(fn='FLDb.out',fig=None):
    """
    Plot results in FLDb.out

    Arguments
    =========
    fn = 'FLDb.out'
    fig = None (if None, create one)
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from MP.lib import mpl_lib
    import matplotlib.gridspec as gridspec

    norm = mpl.colors.Normalize(vmin=0., vmax=90.)
    cmap, m = mpl_lib.norm_cmap(mn=0.,mx=90.,cm_name='brg')

    ## graphics -------------------------------
    if fig==None:
        fig = plt.figure(figsize=(5,3))
        gs  = gridspec.GridSpec(
            9,7,wspace=0.8,hspace=0.8,
            left=0.2,right=0.95,top=0.90,bottom=0.2)

        ax1 = fig.add_subplot(gs[3:,:3])
        ax2 = fig.add_subplot(gs[3:,4:7])
        axb = fig.add_subplot(gs[0,:])
        cb  = mpl.colorbar.ColorbarBase(
            axb,cmap=cmap,spacing='proprotional',
            orientation='horizontal',
            norm=norm,filled=True,format=r'$%2i^\circ$')
        axb.set_xlabel('$\psi_f$',dict(fontsize=15))
        cb.set_ticks(np.arange(0.,90.01,15.))
    else:
        ax1, ax2, axcb = fig.axes

    ## ----------------------------------------
    dat = np.loadtxt(fn,skiprows=1).T
    e1_a,e2_a,s1_a,s2_a,e1_b,e2_b,\
        s1_b,s2_b,psi = dat
    ## continous data trend lines
    ax1.plot(e1_a,e2_a,color='gray',ls='-',alpha=0.2)
    ax2.plot(s1_a,s2_a,color='gray',ls='-',alpha=0.2)
    ax1.locator_params(nbins=4);ax2.locator_params(nbins=4)
    ## select only a few for better visuability
    ntot = len(dat[0])
    nf   = ntot / 3
    dat  = dat.T[::nf].T
    e1_a,e2_a,s1_a,s2_a,e1_b,e2_b,\
        s1_b,s2_b,psi = dat

    for i in xrange(len(psi)):
        n, t = nt_xycoords(
            psi[i],xy=[e1_a[i],e2_a[i]],r=0.15)

        ax1.plot(e1_a[i], e2_a[i],'s',mfc='None',
                 mec=m.to_rgba(psi[i]))
        lab1=None; lab2=None
        if i==0: lab1='normal'; lab2='transverse'

        ## arrow
        ax1.arrow(e1_a[i],e2_a[i],n[0]-e1_a[i],n[1]-e2_a[i],
                  color='m',label=lab1,alpha=0.2)
        # ax1.arrow(e1_a[i],e2_a[i],t[0]-e1_a[i],t[1]-e2_a[i],
        #           color='m',label=lab1,alpha=0.2)
        ##
        ax2.plot(s1_a[i], s2_a[i],
                 's',mfc='None',
                 mec=m.to_rgba(psi[i]))
        ax1.plot(e1_b[i], e2_b[i],
                 'x',mfc='None',
                 mec=m.to_rgba(psi[i]))
        ax2.plot(s1_b[i], s2_b[i],
                 'x',mfc='None',
                 mec=m.to_rgba(psi[i]))
    # ax1.legend(loc='best',fancybox=True,
    #            framealpha=0.5,fontsize=7,numpoints=1)
    # ax1.set_xlim(-0.50,1.50); ax1.set_ylim(-0.50,1.50)
    # ax2.set_xlim(0.,700);ax2.set_ylim(0.,700)

    ax1.set_xlabel(r'$\bar{E}_{1}$')
    ax1.set_ylabel(r'$\bar{E}_{2}$')
    ax2.set_xlabel(r'$\bar{\Sigma}_{1}$')
    ax2.set_ylabel(r'$\bar{\Sigma}_{2}$')
    fig.savefig('FLDb.pdf')
    return fig

def _wsp_main_(Ea,Eb,Sa,Sb,Pf,P0,path_home,i_rhos,
               date,f0,dir):
    """
    The main component of 'write_stress_paths'
    -- taken out of write_stress_paths
    -- in order to reuse of the core part
    -- as a part of FLD function
    """
    from os import getcwd, chdir
    exx,eyy,psi0_sorted,dum = least_limit(ref_dat=Ea,psi=P0);
    chdir(path_home)
    fout = open('fld_a_%s_stress_path.txt'%date,'w')
    fout1 = open('fld_a_%s_strain_path.txt'%date,'w')
    fout2 = open('fld_a_%s_psi_path.txt'%date,'w')
    fout3 = open('fld_b_%s_strain_path.txt'%date,'w')
    fout4 = open('fld_b_%s_stress_path.txt'%date,'w')

    for i in xrange(len(i_rhos)):
        fout.write('%3.3i '%i_rhos[i])
        fout1.write('%3.3i '%i_rhos[i])
        fout2.write('%3.3i '%i_rhos[i])
        fout3.write('%3.3i '%i_rhos[i])
        fout4.write('%3.3i '%i_rhos[i])

    fout.write('\n');fout1.write('\n');fout2.write('\n');
    fout3.write('\n');fout4.write('\n')
    for i in xrange(len(i_rhos)):
        fout.write('--\n');fout1.write('--\n');fout2.write('--\n')
        fout3.write('--\n');fout4.write('--\n')
        irho = i_rhos[i]
        psi0 = psi0_sorted[irho]

        fn=os.path.join(
            path_home,dir,
            'fld_b_rst_%s_psi%2.2i_f%5.3f.tar'%(
                date,psi0,f0))
        chdir(path_home)
        Ea,Eb,Sa,Sb,psif = read_b_tar(
            fn=fn,ihub=True,irho=irho)
        for j in xrange(len(Sa[0])):
            fout.write('%6.1f  %6.1f \n'%(
                    Sa[0][j],Sa[1][j]))
        for j in xrange(len(Ea[0])):
            fout1.write('%6.4f  %6.4f \n'%(
                    Ea[0][j],Ea[1][j]))
        for j in xrange(len(Ea[0])):
            fout3.write('%6.4f  %6.4f \n'%(
                    Eb[0][j],Eb[1][j]))
        for j in xrange(len(Ea[0])):
            fout4.write('%6.4f  %6.4f \n'%(
                    Sb[0][j],Sb[1][j]))
        for j in xrange(len(psif)):
            fout2.write('%6.4f  nan \n'%psif[j])

    fout.close();fout1.close();fout2.close()
    fout3.close();fout4.close()
    print fout.name, ',',fout1.name,'and ',\
        fout2.name, 'have been created'
    return fout.name, fout1.name, fout2.name, \
        fout3.name, fout4.name

def _wp_reader_(fn='fld_a_20150226_stress_path.txt'):
    dat_string=open(fn,'r').read()
    d = dat_string.split('--\n')
    rhos=map(int, d[0].split('\n')[0].split())
    X = []
    Y = []
    for i in xrange(len(rhos)):
        dat = d[i+1].split('\n')
        x, y = [], []
        for j in xrange(len(dat)-1):
            xx, yy = map(float,dat[j].split())
            x.append(xx)
            y.append(yy)
        X.append(x)
        Y.append(y)
    return X, Y


def write_stress_paths(
    i_rhos=[0,5,15,-6,-1],
    date='20150303',f0=0.990,
    dir='.'):
    """
    Arguments
    =========
    i_rhos = [0,5,15,-6,-1]
    date   = '20150303'
    f0     = 0.990
    dir    = '.'
    """
    from os import getcwd, chdir
    path_home = getcwd()
    Ea, Eb, Sa, Sb, Pf, P0 = FLD(
        date=date,f=f0,dir=dir,iplot=False)
    ##
    _wsp_main_(Ea,Eb,Sa,Sb,Pf,P0,path_home,
               i_rhos,date,f0,dir)
    ##

def read_from_fld_std(fn='stdout.out'):
    """
    Read from stdo file generated by a VPCS-FLDB run

    Argument
    ========
    fn

    Returns
    =======
    Ea, Eb, Sa, Sb, psi
    """
    lines = open(fn,'r').read().split('\n')
    iexit=False
    Ea  = []
    Eb  = []
    Sa  = []
    Sb  = []
    psi = []
    il = 0
    for i in xrange(len(lines)):
        line = lines[i]
        try:
            a = line.split('||')[0]
            b = line.split('||')[1]
            adat = a.split()
            bdat = b.split()
            ex_a = float(adat[1])
            ey_a = float(adat[2])
            sx_a = float(adat[5])
            sy_a = float(adat[6])
            ex_b = float(bdat[0])
            ey_b = float(bdat[1])
            sx_b = float(bdat[4])
            sy_b = float(bdat[5])
            p    = float(bdat[10])

            Ea.append([ex_a,ey_a])
            Eb.append([ex_b,ey_b])
            Sa.append([sx_a,sy_a])
            Sb.append([sx_b,sy_b])
            psi.append(p)
        except: pass
        else:
            il = il+ 1
    return np.array(Ea).T, np.array(Eb).T,\
        np.array(Sa).T, np.array(Sb).T, psi

def read_b_tar(
    fn='fld_b_rst_20140304_psi00_f0.990.tar',
    ihub=True,
    irho=0
    ):
    """
    Read the full history of region A and B
    from stdout.out file.
    """
    if not(os.path.isfile(fn)):
        raise IOError, 'given file %s is not found'%fn

    member_names = tar_members(fn=fn)
    member_names.sort()
    # print irho
    member_name = member_names[irho]
    # print member_name
    if ihub:
        from os import chdir, getcwd
        from shutil import move, rmtree, copy
        from tempfile import mkdtemp
        path_hub = mkdtemp(dir='/tmp')
        path_home = getcwd()
        copy(fn, path_hub) # or copy?
        chdir(path_hub)

    ## untar member_name
    untar_fn(fn,member_name)
    untar_from_fld_tar(member_name, objf='stdout.out')
    os.remove(member_name)

    lines = open('stdout.out','r').read().split('\n')
    # print 'lines:', len(lines)
    # print os.path.join(os.getcwd(), 'stdout.out')

    iexit=False
    Ea  = []
    Eb  = []
    Sa  = []
    Sb  = []
    psi = []
    il = 0
    for i in xrange(len(lines)):
        line = lines[i]
        try:
            a = line.split('||')[0]
            b = line.split('||')[1]
            adat = a.split()
            bdat = b.split()
            ex_a = float(adat[1])
            ey_a = float(adat[2])
            sx_a = float(adat[5])
            sy_a = float(adat[6])
            ex_b = float(bdat[0])
            ey_b = float(bdat[1])
            sx_b = float(bdat[4])
            sy_b = float(bdat[5])
            p    = float(bdat[10])

            Ea.append([ex_a,ey_a])
            Eb.append([ex_b,ey_b])
            Sa.append([sx_a,sy_a])
            Sb.append([sx_b,sy_b])
            psi.append(p)
        except: pass
        else:
            il = il+ 1
    if ihub: chdir(path_home)
    return np.array(Ea).T, np.array(Eb).T,\
        np.array(Sa).T, np.array(Sb).T, psi


def save_b_tar(
    fn='fld_b_rst_20140714_psi00_f0.990.tar',
    f_crit=5,verbose=True):
    """
    From stdout.out save Ex, Ey, Sx, Sy, Ex, Ey, Sx, Sy, psi, rho, and rho_b

    Arguments
    =========
    fn
    f_crit = 10 ## tar file name of a region b member
    verbose = True
    """
    member_names = tar_members(fn=fn)
    ihub = False
    if len(member_names)>3:
        ## move it to temp file and extrac it there
        from os import chdir, getcwd
        from shutil import move, rmtree, copy
        from tempfile import mkdtemp
        path_hub = mkdtemp(dir='/tmp')
        path_home = getcwd()
        copy(fn, path_hub) # or copy?
        chdir(path_hub)
        ihub = True

    fout = '%s%s'%(fn.split('.tar')[0],'.out')
    fout = open(fout,'w')

    fmt ='%s '*15

    header = fmt%('Ex','Ey','Exy','Sx','Sy','Sxy',
                  'Ex','Ey','Exy','Sx','Sy','Sxy',
                  'psi','rho','rho_b')
    ## print header
    fout.write(header+'\n')
    fmt='%10.5f '*15

    for i in xrange(len(member_names)):
        untar_fn(fn,member_names[i])
        if ihub: copy(member_names[i],path_home)
        untar_from_fld_tar(member_names[i],objf='stdout.out')
        untar_from_fld_tar(member_names[i],objf='stderr.out')
        os.remove(member_names[i])

        ## look only at the last 10 lines
        mx_line = 10
        l = os.popen('tail -n %i stdout.out'%mx_line)
        el = os.popen('tail -n %i stderr.out'%mx_line)
        ed = el.readlines()
        d = l.readlines()
        d = d[::-1]
        if len(d)<mx_line: print 'Not enough',\
           ' lines were found in stdout.out'

        ##
        found = False
        il = 0
        ifail = False
        while not(found):
            found = True
            try:
                a = d[il].split('||')[0]
                b = d[il].split('||')[1]
            except:
                found = False
                il = il +1
                pass
            if il>len(d):
                ifail=True
                found = True
        adat=a.split()
        ##

        try:
            bdat=b.split()
            f = float(bdat[9])
        except: ifail = True

        if not(ifail):
            if f<f_crit:
                ifail = True
                if verbose:
                    print "Error 1) Final thickness strain",\
                        " rate didn't reach the critical value. ",
                    print "Found ratio was %3.1f but criterion"%f,\
                        " was set as %3.1f"%f_crit

                    print 'Line:', (len(d[il])-6)*'-'
                    print d[il].split('\n')[0]
                    print len(d[il])*'-'
            if f>f_crit and abs(f)>abs(f_crit*1e2):
                ifail = True
                if verbose:
                    print "Error 2) Final thickness strain",\
                        " rate seems unreliable. ",
                    print "Found ratio was %3.1f, which"%f,\
                        " exceeds the criterion (%f) by far "%(f_crit*1e3)


        if not(ifail):
            if verbose:
                print a, b.split('\n')[0]

            try:print ed[-1].split('\n')[0]
            except:IndexError, 'Nothing in stderr.out'
            ex_a = float(adat[1])
            ey_a = float(adat[2])
            exya = float(adat[4])
            sx_a = float(adat[5])
            sy_a = float(adat[6])
            sxya = float(adat[8])
            ex_b = float(bdat[0])
            ey_b = float(bdat[1])
            exyb = float(bdat[3])
            sx_b = float(bdat[4])
            sy_b = float(bdat[5])
            sxyb = float(bdat[7])
            psi  = float(bdat[10])
            try:    rho  = float(bdat[16])
            except: rho =np.nan
            try:    rho_b= float(bdat[17])
            except: rho_b=np.nan

        elif ifail:
            if verbose: print 'Unsuccessful probing'
            ex_a = np.nan
            ey_a = np.nan
            exya = np.nan
            sx_a = np.nan
            sy_a = np.nan
            sxya = np.nan
            ex_b = np.nan
            ey_b = np.nan
            exyb = np.nan
            sx_b = np.nan
            sy_b = np.nan
            sxyb = np.nan
            psi  = np.nan
            rho  = np.nan
            rho_b= np.nan

        dat = fmt%(ex_a,ey_a,exya,sx_a,sy_a,sxya,
                   ex_b,ey_b,exyb,sx_b,sy_b,sxyb,
                   psi,rho,rho_b)

        fout.write(dat)
        fout.write('\n')
    fout.close()
    ## in case local /tmp is much faster
    ## use a temporal folder (path_hub)
    if ihub:
        move(fout.name, path_home)
        chdir(path_home)
        rmtree(path_hub)

def strain_path_ab(a='fld_a_rst_20140623.tar',
                   b='fld_b_rst_20140623_psi??.tar',
                   irm_a=True,iplot=True):
    deco = axes_label.__deco_fld__
    psi0 = None; f0   = None
    if b!= None:
        taf, tbf = fld_subfiles(a,b)
        psi0 = float(b.split('psi')[-1].split('_')[0])
        f0 = float(b.split('f')[-1].split('.tar')[0])
    else:
        taf = fld_subfiles(a,b)
        taf.sort()
    if b!=None: tbf.sort()
    figs=None
    eps_fl_a = []; sig_fl_a = []; eps_fl_b = [];
    sig_fl_b = []; psi_fl = []
    for i in xrange(len(taf)):
        if b==None:
        # (region A)
            if os.path.isfile(taf[i]): pass
            else: untar_fn(a,taf[i]) # untar a path file

            #print taf[i]
            untar_from_fld_tar(taf[i],objf='region_a.out')
            os.system('cp region_a.out region_a_%2.2i.out'%i)

            if irm_a: os.remove(taf[i])
            figs = pre_path(figs=figs)

        if b!=None:
        # (region B)
            untar_fn(b,tbf[i]) # untar a path file
            untar_from_fld_tar(tbf[i],objf='region_b.out')
            untar_from_fld_tar(tbf[i],objf='region_a2.out')
            untar_from_fld_tar(tbf[i],objf='udots.out')
            os.remove(tbf[i])
            figs, eps_fa, eps_fb, sig_fa, sig_fb, psi_f = \
                main(figs=figs,iplot=iplot)
            ## over-ride psi_f from udots.out
            # psi_f = psi_f * np.tan(psi0*np.pi/180.) * 180./np.pi
            # print '%5.1f  %5.1f'%(psi0, psi_f)

            eps_fl_a.append(eps_fa)
            sig_fl_a.append(sig_fa)
            eps_fl_b.append(eps_fb)
            sig_fl_b.append(sig_fb)
            psi_fl.append(psi_f)

    if b!=None:
        os.remove('region_b.out')
        os.remove('region_a2.out')
        os.remove('udots.out')

    deco_fld(figs.axes[0],iopt=2)
    deco_fld(figs.axes[1],iopt=3)

    return figs, eps_fl_a, eps_fl_b, \
        sig_fl_a, sig_fl_b, psi0, psi_fl, f0

def tar_members(fn='fld_a_rst_20140623.tar'):
    """
    Find members of the given tar file and return

    Arguments
    =========
    fn (name of the tar file)

    Returns
    =======
    taf (list of tar file members)
    """
    tar_a=os.popen('%s --warning=none -tf %s'%(_tar_, fn))
    tar_a_fns = tar_a.readlines()
    taf=[]
    for i in xrange(len(tar_a_fns)):
        taf.append(tar_a_fns[i][:-1])
    return taf

def fld_subfiles(a='fld_a_rst_20140623.tar',
                 b='fld_b_rst_20140623.tar'):
    """
    Provide the list of files in a tar archive

    Arguments
    =========
    a, b
    """
    tar_a=os.popen('%s --warning=none -tf %s'%(_tar_,a))
    if b!=None: tar_b=os.popen('%s -tf %s'%(_tar_,b))
    tar_a_fns = tar_a.readlines()
    if b!=None: tar_b_fns = tar_b.readlines()

    taf=[]; tbf=[]
    for i in xrange(len(tar_a_fns)):
        taf.append(tar_a_fns[i][:-1])
    if b!=None:
        for i in xrange(len(tar_b_fns)):
            tbf.append(tar_b_fns[i][:-1])
    if b!=None:
        if (len(tar_a_fns)!=len(tar_b_fns)):
            print 'Number of results differ'
    if b!=None:
        return taf, tbf
    else: return taf

def diagram_various_psi(
    a='fld_a_rst_20140623.tar',iopt=0):
    """
    Move all files to temp folder...

    a: name of the tar file for region A
    iopt=0: reference psi: psi0
    iopt=1: reference psi: psi1
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from MP.lib import mpl_lib

    date = a.split('_')[-1].split('.tar')[0]
    fnb=glob('fld_b_rst_%s_psi*_f*.tar'%date)
    fnb.sort()

    eps_fl_a=[]; sig_fl_a=[]; psi0s=[]; f0s=[]; psi1s=[]

    for i in xrange(len(fnb)):
        fig,eps_f_a,eps_f_b_b, sig_f_a,sig_f_b,psi0,psi1,\
            f0 = strain_path_ab(a=a,b=fnb[i])
        fn_FLD_indv = PdfPages(
            'all_FLD_psi0_%5.2f_f0_%5.3f.pdf'%(psi0,f0))
        eps_fl_a.append(eps_f_a); sig_fl_a.append(sig_f_a)
        psi0s.append(psi0); f0s.append(f0);
        psi1s.append(psi1)
        fn_FLD_indv.savefig(fig)
        plt.close(fig)
        fn_FLD_indv.close()

    ## (32,3) (nrhos, npsi0)
    eps_fl_a = np.array(eps_fl_a).T
    sig_fl_a = np.array(sig_fl_a).T
    nrhos = len(eps_fl_a)
    npsi0 = len(eps_fl_a[0])
    print 'nrhos and npsi0:', nrhos, npsi0

    ex = eps_fl_a[0].T; ey = eps_fl_a[1].T
    sx = sig_fl_a[0].T; sy = sig_fl_a[1].T

    psi1s = np.array(psi1s)
    figs=wf(nw=2,right=0.21,uw=3.2,
            uh=3.2,left=0.04,w1=0.15,ws=0.65,hs=0.65)

    psi1s_flat = psi1s.flatten()

    minpsi = min([min(psi0s),min(psi1s_flat)])
    maxpsi = max([max(psi0s),max(psi1s_flat)])

    norm = mpl.colors.Normalize(vmin=0., vmax=90.)
    cmap, m = mpl_lib.norm_cmap(mn  =0.,   mx=90.)
    colors=[]

    ## reference psi (iopt=0 initial, iopt=1 final)
    if iopt==0: ref_psi = psi0s
    if iopt==1: ref_psi = psi1s_flat

    ax=figs.axes[0] # FLD in strain space
    if iopt==0:
        for i in xrange(len(ex)): ## (npsi0, nrhos)
            label=None
            cl = m.to_rgba(ref_psi[i])
            colors.append(cl)
            ax.plot(ex[i],ey[i],label=label,color=cl)

    if iopt==1:
        for i in xrange(len(ex)):  ## (npsi0, nrhos)
            for j in xrange(len(ex[i])):
                label=None
                cl = m.to_rgba(psi1s[i][j])
                colors.append(cl)
                ax.plot(ex[i][j],ey[i][j],'o',
                        label=label,mec=cl,
                        mfc=None,color=cl)

    deco_fld(ax,iopt=2,ft=12)
    draw_guide(ax,r_line=[-0.5,0.,1,2.0,2.5],max_r=2.)

    ax=figs.axes[1] # FLD in stress space
    if iopt==0:
        for i in xrange(len(sx)):
            ax.plot(sx[i],sy[i],
                    color=m.to_rgba(ref_psi[i]))
    if iopt==1:
        for i in xrange(len(sx)):
            for j in xrange(len(sx[i])):
                cl = m.to_rgba(psi1s[i][j])
                ax.plot(sx[i][j],sy[i][j],
                        'o',mec=cl,mfc=None,color=cl)

    deco_fld(ax,iopt=3,ft=12)

    draw_guide(ax,r_line=[0,0.5,1,1.5,2.],max_r=1000)
    b    = figs.axes[-1].get_position()
    axcb = figs.add_axes([0.80,b.y0,0.03,b.y1-b.y0])

    # axcb.set_ylim(minpsi,maxpsi)
    # axcb.set_xlim(minpsi,maxpsi)

    if iopt==0:ylab=r'$\psi_0$'
    if iopt==1:ylab=r'$\psi_f$'

    lw = 1.2
    #if iopt==0: lw=8
    #if iopt==1: lw=1.2

    mpl_lib.add_cb(axcb,cmap=cmap,filled=False,
                   norm=norm,ylab=ylab,levels=ref_psi,
                   colors=colors,lw=lw)

    axcb.set_frame_on(False)
    axcb.get_xaxis().set_visible(False)

    if iopt==0: figs.savefig('FLD_col_0.pdf')
    if iopt==1: figs.savefig('FLD_col_1.pdf')

    return ex,ey,sx,sy,psi0s,psi1s

def draw_guide(ax,r_line = [-0.5,0. ,1],max_r=2,
               ls='--',color='k',alpha=0.5):
    """
    Maximum should be a radius...
    """
    import fld
    # guide lines for probed paths
    xlim=ax.get_xlim(); ylim=ax.get_ylim()
    for i in xrange(len(r_line)):
        r = r_line[i]
        if r<=1:
            mx=max_r
            mx = mx/np.sqrt(1.+r**2)
            ys = np.linspace(0.,mx)
            xs = r * ys
        elif r>1:
            r = fld.rho_transform(r)
            my = mx/np.sqrt(1.+r**2)
            xs = np.linspace(0.,my)
            ys = r * xs

        ax.plot(xs,ys,ls=ls,color=color,alpha=alpha)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def fld_psi(a='fld_a_rst_20140623.tar',rho0=-0.7,rho1=2.7,n=32,iopt=0):
    """
    Find least amount of formability spanning all probed psi0 angle

    iopt (0: psi0)
    iopt (1: psi1)

    """
    import matplotlib as mpl
    from MP.lib import mpl_lib
    from MP import ssort
    sort = ssort.shellSort


    if iopt==0: figs=wf(nw=2,right=0.21,uw=3.2,uh=3.2,left=0.04,w1=0.15,ws=0.65,hs=0.65)
    if iopt==1: figs=wf(nw=2,nh=2,right=0.21,uw=3.2,uh=3.2,
                        left=0.04,w1=0.15,ws=0.65,hs=0.65,iarange=True)
    axs = figs.axes

    ## getting/plotting data collectively
    ex,ey,sx,sy,psi0s,psi1s = diagram_various_psi(a=a,iopt=iopt)

    ex=ex.T;ey=ey.T;sx=sx.T;sy=sy.T;# ex now is (nprob,npsi)

    ## Finding the 'true' forming limit that gives least deformation
    nprob = len(ex)    ## for different rhos
    npsi0 = len(ex[0]) ## for different psi0

    ## reference psi (iopt=0 initial, iopt=1 final)
    minpsi = min([min(psi0s),min(psi1s.flatten())])
    maxpsi = max([max(psi0s),max(psi1s.flatten())])
    if iopt==0: ref_psi = psi0s
    if iopt==1: ref_psi = psi1s

    ## colorbar creation
    norm = mpl.colors.Normalize(vmin=0., vmax=90.) ##vmin=minpsi,vmax=maxpsi)
    cmap, m = mpl_lib.norm_cmap(mn=0.,   mx  = 90.) ##mx=maxpsi,mn=minpsi)

    colors=[]
    p0s = [] ## critical angles
    minx=100
    miny=100
    Exs=[];Eys=[];zs =[]
    Sxs=[];Sys=[];
    for irho in xrange(nprob):
        r_mn = 0.
        i_mn = None
        epsx = ex[irho]; epsy = ey[irho]
        sigx = sx[irho]; sigy = sy[irho]
        R = []
        for ipsi in xrange(npsi0):
            R.append(r(epsx[ipsi],epsy[ipsi]))

        ## sort and find the minimum radius from zero
        val,ind=sort(R)
        i_mn = ind[0]
        if iopt==0: p0s.append(ref_psi[i_mn])

        x,y = epsx[i_mn],epsy[i_mn]
        if x<minx: minx = x
        if y<miny: miny = y

        if iopt==0:
            cl = m.to_rgba(ref_psi[i_mn])
            colors.append(cl)
            axs[0].plot(x,y,'o',mfc=None,mec=cl,color=cl)
            axs[1].plot(sigx[i_mn],sigy[i_mn],'o',mfc=None,mec=cl,color=cl)
        if iopt==1:
            cl = m.to_rgba(ref_psi[i_mn][irho])
            p0s.append(ref_psi[i_mn][irho])
            colors.append(cl)
            axs[0].plot(x,y,'o',mfc=None,mec=cl,color=cl)
            axs[1].plot(sigx[i_mn],sigy[i_mn],'o',mfc=None,mec=cl,color=cl)

            symb = '+'
            sxx = sigx[i_mn]
            syy = sigy[i_mn]
            if y<x:
                dum = y
                y = x
                x = dum
                symb = 'x'
                dum = syy
                syy = sxx
                sxx = dum

            axs[2].plot(x,y,symb,color=cl)
            axs[3].plot(sxx,syy,symb,color=cl)
            Exs.append(x);Eys.append(y);
            Sxs.append(sxx);Sys.append(syy);
            zs.append(ref_psi[i_mn][irho])

    ## if iopt==1: # colored line segments
    ##     colorline(axs[2],Exs,Eys,zs,cmap=cmap,norm=norm)
    ##     colorline(axs[3],Sxs,Sxy,zs,cmap=cmap,norm=norm)

    ## decorating the FLD axes
    deco_fld(axs[0],iopt=2,ft=12)
    deco_fld(axs[1],iopt=3,ft=12)

    if iopt==1:
        deco_fld(axs[2],iopt=0,ft=12)
        deco_fld(axs[3],iopt=1,ft=12)
        axs[2].set_ylim(0.,1.)
        axs[3].set_ylim(0.,)
        axs[2].text(-0.45,0.85,'Axis 1 being RD (+) or TD (x)', fontsize=7)

    draw_guide(axs[0],r_line=[-0.5,0.0,1.0,2.0,2.5],max_r=0.5)
    draw_guide(axs[1],r_line=[ 0.0,0.5,1.0,1.5,2.0],max_r=1000)
    if iopt==1:
        draw_guide(axs[2],r_line=[-0.5,0.0,1],max_r=0.5)
        draw_guide(axs[3],r_line=[ 0.0,0.5,1],max_r=1000)

    b    = figs.axes[1].get_position()
    axcb = figs.add_axes([0.80,b.y0,0.03,b.y1-b.y0])
    if iopt==0:ylab=r'$\psi_0$'
    if iopt==1:ylab=r'$\psi_f$'
    lw=1.2
    # if iopt==0: lw=8
    # if iopt==1: lw=1.2
    mpl_lib.add_cb(axcb,cmap=cmap,filled=False,norm=norm,
                   ylab=ylab,levels=p0s,colors=colors,lw=lw)
    axcb.set_frame_on(False)
    axcb.get_xaxis().set_visible(False)
    if iopt==0:figs.savefig('FLD_true_0.pdf')
    if iopt==1:figs.savefig('FLD_true_1.pdf')

def r(x,y): return np.sqrt(x**2+y**2)

def which_is_closer(xy0,xy1):
    r0 = r(xy0[0],xy0[1])
    r1 = r(xy1[0],xy1[1])
    if r0<r1: return 0
    if r1<r0: return 1
    if r0==r1: return 1

def untar_fn(
    tarfn='fld_a_rst_20140623.tar',
    fn='fld_a_rst_00.tar'):
    os.system('%s --warning=none -xf %s %s'%(_tar_,tarfn,fn))

def untar_from_fld_tar(fn='fld_a_rst_00.tar',
                       objf='region_a.out'):
    """
    This def is compliant with GNU-tar only as
    BSD tar doesn't support --wildcards nor --strip
    """
    tar_cmd = _tar_
    cmd = '%s '%(tar_cmd)+\
          ' -xf %s %s '%(fn,objf)
    p = os.system(cmd)
    if p!=0:
        cmd = '%s --warning=none --wildcards'%(tar_cmd)+\
              ' -xf %s %s'%(fn,objf)
        p = os.system(cmd)
        if p!=0:
            raise IOError, "Couldn't extract %s from %s"%(
                objf, fn)

def untar_f_from_tars(wc='fld_a_rst_??.tar',objf='vpsc7.in'):
    fns = glob(wc)
    for i in xrange(len(fns)):
        untar_from_fld_tar(fn=fns[i],objf=objf)
        os.system('cp %s %s_%2.2i'%(objf,objf,i))

def strip_prefix_tar(wc='fld_b_rst_20140623*.tar',verbose=False):
    fns = glob(wc)
    for i in xrange(len(fns)):
        strip_prefix_path_tar(fn=fns[i],
                              verbose=verbose)

def strip_prefix_path_tar(
        fn='fld_a_rst_00.tar',
        verbose=False):
    # # move file/cwd to './tmp' to speed up a bit
    path0 = os.getcwd()

    if os.path.exists('/tmp/%s'%fn):
        os.remove('/tmp/%s'%fn)

    shutil.move(fn,'/tmp/')
    os.chdir('/tmp/')

    fns=os.popen('%s --warning=none -tf %s'%(_tar_,fn)) # list of archived filenames

    fns=fns.readlines()
    FNS=[]
    new_fns=[]

    for i in xrange(len(fns)):
        f = fns[i][:-1]
        if verbose: print f
        n_prefix = len(f.split(os.sep))-1
        if n_prefix>1:
            cmd = '%s --warning=none -xf %s %s --strip=%i'%(
                _tar_, fn,f,n_prefix)
            if verbose: cmd = '%s --warning=none -xvf %s %s --strip=%i'%(
                    _tar_, fn,f,n_prefix)
            iflag=os.system(cmd)
            if iflag!=0: raise IOError
            new_fn = f.split(os.sep)[-1]
            new_fns.append(new_fn)

    #print new_fns

    if len(new_fns)>0:
        cmd = '%s --warning=none -cf temp.tar'%_tar_
        for i in xrange(len(new_fns)):
            cmd = '%s %s'%(cmd, new_fns[i])
        # print '\n\n'
        # print cmd
        # print '\n\n'
        iflag = os.system(cmd)
        if iflag!=0: raise IOError
        else:
            for i in xrange(len(new_fns)):
                os.remove(new_fns[i])

            shutil.move('temp.tar', 'temp0.tar')
            cmd2 = 'mv temp0.tar %s'%fn
            os.system(cmd2)

    if os.path.exists('%s%s%s'%(path0,os.sep,fn)):
        os.remove('%s%s%s'%(path0,os.sep,fn))
    shutil.move(fn,path0)
    os.chdir(path0)

def fit_c3():
    import matplotlib.pyplot as plt
    fig=plt.figure(1,figsize=(4,3));ax=fig.add_axes((0.25,0.25,0.7,0.7))
    exp_dat =np.loadtxt('EXP_BULGE_JINKIM.txt').T
    ax.plot(exp_dat[0][::10],exp_dat[1][::10],'gx',label='EXP bulge')
    vpsc_dat=np.loadtxt('STR_STR.OUT',skiprows=1).T
    ax.plot(-vpsc_dat[4],-vpsc_dat[10],'g-',label='VPSC C3')


def flow():
    import matplotlib.pyplot as plt
    # fig=plt.figure(1,figsize=(4,3));ax=fig.add_axes((0.25,0.25,0.7,0.7))
    # vpsc_dat=np.loadtxt('STR_STR.ten_nrs',skiprows=1).T
    # ax.plot(vpsc_dat[2],vpsc_dat[8],'r-',label='VPSC tension')
    # exp_dat =np.loadtxt('avgstr_000.txt').T
    # ax.plot(exp_dat[0][::2],exp_dat[1][::2],'rx',label='EXP tension')

    # vpsc_dat=np.loadtxt('STR_STR.EB_nrs',skiprows=1).T
    # ax.plot(-vpsc_dat[4],(vpsc_dat[8]+vpsc_dat[9])/2.,'b-',label='VPSC EB')
    # exp_dat =np.loadtxt('EXP_BULGE_JINKIM.txt').T
    # ax.plot(exp_dat[0][::10],exp_dat[1][::10],'bx',label='EXP bulge')

    # vpsc_dat=np.loadtxt('STR_STR.c3_nrs',skiprows=1).T
    # ax.plot(-vpsc_dat[4],-vpsc_dat[10],'g-',label='VPSC C3')
    # # exp_dat =np.loadtxt('EXP_BULGE_JINKIM.txt').T
    # # ax.plot(exp_dat[0][::10],exp_dat[1][::10],'gx',label='EXP bulge')

    # ax.legend(loc='best',fontsize=12)


    ## rate sensitive sx
    fig=plt.figure(2,figsize=(4,3));ax=fig.add_axes((0.25,0.25,0.7,0.7))
    vpsc_dat=np.loadtxt('STR_STR.ten_rs',skiprows=1).T
    #ax.plot(vpsc_dat[2],vpsc_dat[8],'r-',label='VPSC uniaxial tension')
    ax.plot(vpsc_dat[0],vpsc_dat[1],'r-',label='VPSC uniaxial tension')
    exp_dat =np.loadtxt('mat/Bst/avgstr_000.txt').T
    nstp = 1
    ax.plot(exp_dat[0][::nstp],exp_dat[1][::nstp],'rx',label='EXP uniaxial tension')

    # ## EB
    # vpsc_dat=np.loadtxt('STR_STR.EB_rs',skiprows=1).T
    # ax.plot(-vpsc_dat[4],(vpsc_dat[8]+vpsc_dat[9])/2.,'b-',label='VPSC EB')
    # exp_dat =np.loadtxt('EXP_BULGE_JINKIM.txt').T
    # ax.plot(exp_dat[0][::10],exp_dat[1][::10],'bx',label='EXP bulge')

    ## Compr.3
    vpsc_dat=np.loadtxt('STR_STR.c3_rs',skiprows=1).T
    #ax.plot(-vpsc_dat[4],-vpsc_dat[10],'b-',label='VPSC bulge')
    ax.plot(vpsc_dat[0],vpsc_dat[1],'b-',label='VPSC bulge')
    exp_dat =np.loadtxt('mat/Bst/EXP_BULGE_JINKIM_0.3.txt').T
    ax.plot(exp_dat[0][::10],exp_dat[1][::10],'b+',label='EXP bulge (up to 30%)')

    exp_dat =np.loadtxt('mat/Bst/TGH_MK_flow.txt').T
    e_vm = exp_dat[0][:-1]*2.
    s_vm = (exp_dat[1]+exp_dat[3])[:-1]/2.
    s_er = (exp_dat[2]+exp_dat[4])[:-1]/2.
    ax.errorbar(e_vm,s_vm,s_er,fmt='g.',label='EXP Marciniak X-ray')

    exp_dat =np.loadtxt('mat/Bst/EXP_BULGE_JINKIM.txt').T
    ax.plot(exp_dat[0][::10],exp_dat[1][::10],'b--',label='EXP bulge (total)')

    ax.legend(loc='best',fontsize=9,fancybox=True).get_frame().set_alpha(0.5)

    ax.set_xticks(np.linspace(0.,0.9,4))
    ax.set_yticks(np.linspace(0.,600,5))

    ax.set_ylabel(r'Von Mises Stress $\bar{\mathrm{\Sigma}}^{\mathrm{VM}}$ [MPa]',
                  dict(fontsize=12))
    ax.set_xlabel(r'Von Mises Strain $\bar{\mathrm{E}}^{\mathrm{VM}}$',
                  dict(fontsize=12))
    ax.grid()
    fig.savefig('Flow_opti.pdf')

def gam_var():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from MP.lib import mpl_lib
    deco = deco_fld
    import matplotlib.pyplot as plt
    norm    = mpl.colors.Normalize(vmin=45., vmax=90.)
    cmap, m = mpl_lib.norm_cmap(mn=45.,   mx  = 90.,
                                cm_name='brg')

    fig=wf(nw=3,nh=1,right=0.18,uw=2.75,uh=2.6,
           down=0.20,up=0.20,
           h0=0.05,hs=0.90,h1=0.05,
           left=0.06,w1=0.15,ws=0.65,iarange=True)


    ax=fig.axes[0]
    ax1=fig.axes[1]
    fns=glob('gamma_???.fld')
    fns.sort()
    for i in xrange(len(fns)):
        fn=fns[i]
        dat=np.loadtxt(fn,skiprows=1).T
        x=dat[0]
        y=dat[1]
        sx,sy = dat[2],dat[3]
        lab = r'w$_0$=%i'%int(fn.split('_')[1].split('.')[0])
        l, = ax.plot(y,x,label=lab)
        ax1.plot(sy,sx,color=l.get_color())

    deco(ax,iopt=0,ft=12)
    deco(ax1,iopt=1,ft=12)

    ax.set_xlim(-0.1,0.5)
    ax.set_ylim(0.2,0.8)

    ax1.set_xlim(0.0,800);ax1.set_ylim(0.,800)


    ## isotropic
    dat=np.loadtxt('ISO.fld',skiprows=1).T
    x=dat[0][:16]
    y=dat[1][:16]
    sx,sy=dat[2][:16],dat[3][:16]
    l, = ax.plot(y,x,label='isotropic')
    ax1.plot(sy,sx,label='isotropic',color=l.get_color())



    ax.legend(loc='best',fancybox=True,
              fontsize=6
              ).get_frame().set_alpha(0.5)

    ## isotropic
    dat=np.loadtxt('ISO.fld',skiprows=1).T
    x=dat[0][:16]
    y=dat[1][:16]
    psif=dat[5][:16]
    ax=fig.axes[2]
    ## ax.set_aspect('equal')
    for i in xrange(len(x)):
        ax.plot(y[i],x[i],'o',mfc='none',mec=m.to_rgba(psif[i]))

    ## gamma 45
    dat=np.loadtxt('gamma_045.fld',skiprows=1).T
    x=dat[0][:16]
    y=dat[1][:16]
    psif=dat[5][:16]
    ax.plot([0,y[0]],[0,x[0]])
    for i in xrange(len(x)):
        ax.plot(y[i],x[i],'^',mfc='none',mec=m.to_rgba(psif[i]))

    ## gamma 30
    dat=np.loadtxt('gamma_030.fld',skiprows=1).T
    x=dat[0]
    y=dat[1]
    psif=dat[5]
    for i in xrange(len(x)):
        ax.plot(y[i],x[i],'d',mfc='none',mec=m.to_rgba(psif[i]))

    ## gamma 15
    dat=np.loadtxt('gamma_015.fld',skiprows=1).T
    x=dat[0]
    y=dat[1]
    psif=dat[5]
    for i in xrange(len(x)):
        ax.plot(y[i],x[i],'2',mfc='none',mec=m.to_rgba(psif[i]))

    ## gamma 10
    dat=np.loadtxt('gamma_010.fld',skiprows=1).T
    x=dat[0]
    y=dat[1]
    psif=dat[5]
    for i in xrange(len(x)):
        ax.plot(y[i],x[i],'s',mfc='none',mec=m.to_rgba(psif[i]))

    b    = fig.axes[1].get_position()
    h    = b.y1-b.y0 ;h  = h * 0.6
    h0   = b.y0      ;h0 = h0 + 0.18
    axcb = fig.add_axes([0.80,h0,0.03,h])
    cb = mpl_lib.add_cb(axcb,cmap=cmap,filled=True,norm=norm,
                   ylab=r'$\psi_f$')
    cb.set_ticks(np.arange(0.,90.01,15.))

if __name__=='__main__':
    import getopt, sys
    try: opts, args = getopt.getopt(
        sys.argv[1:],
        'w:v')
    except getopt.GetoptError, err: print str(err); sys.exit(2)

    wc='fld_b_rst_20140623*.tar'
    verbose = False
    ## override the defaults
    for o, a in opts:
        if o=='-w': wc      = a
        if o=='-v':
            verbose = True
    strip_prefix_tar(wc=wc,verbose=verbose)

def color_bar():
    """
    Plot color bar
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from MP.lib import mpl_lib
    import matplotlib.gridspec as gridspec

    norm = mpl.colors.Normalize(vmin=0., vmax=90.)
    cmap, m = mpl_lib.norm_cmap(
        mn=0.,mx  = 90.,cm_name='brg')
    fig=plt.figure(figsize=(5,1.5))
    gs=gridspec.GridSpec(
        2,1,
        wspace=0.0,hspace=5,
        left=0.05,right=0.95,
        top=0.95,bottom=0.4)
    ax  = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cb  = mpl.colorbar.ColorbarBase(
        ax,cmap=cmap,spacing='proprotional',
        orientation='horizontal',
        norm=norm,filled=True,format=r'$%2i^\circ$')
    cb1 = mpl.colorbar.ColorbarBase(
        ax1,cmap=cmap,spacing='proprotional',
        orientation='horizontal',
        norm=norm,filled=True,format=r'$%2i^\circ$')

    ax.set_xlabel('$\psi_f$',dict(fontsize=15))
    ax1.set_xlabel('$\psi_0$',dict(fontsize=15))
    cb.set_ticks(np.arange(0.,90.01,15.))
    cb1.set_ticks(np.arange(0.,90.01,15.))
    fig.savefig('cb.pdf')


def combine_FLDB(date='20150226',f0=0.990):
    """
    Combine multiple FLD_B_date_f0.990*.tar
    into one.
    """
    from tempfile import NamedTemporaryFile as ntf
    line = 'FLD_B_%s_f%5.3f'%(date,f0)
    fns=glob('%s*.tar'%line)
    if len(fns)==0: return

    fns.sort()
    FNS=[]
    for i in xrange(len(fns)):
        cmd = 'tar -tf %s '%fns[i]
        # print cmd
        F = os.popen(cmd).readlines()
        for j in xrange(len(F)):
            FNS.append(F[j].split('\n')[0])


    ## list completed jobs
    FNS.sort()
    for i in xrange(len(FNS)):
        psi = FNS[i].split('psi')[1][:2]
        print '%2.2i  %2.2i  %s'%(int(psi)/5+1, int(psi), FNS[i])
