c     Forming Limit Diagram following the Marciniak&Kuzynski's
c     inhomogeneity-based approach

c     This subroutine controls, two independent simulations
c     that initially have the same state variables but soon
c     follows completely different deformation history
c
c------------------------------------------------------------c
c     iopt 0: run only for the region A (rho - control but including uniaxial stress)
c     iopt 1: run only for the region A (for a fixed rho)
c     iopt 2: run only for the region B based on precedent
c     iopt 3: run only for the region A (alpha control)
c     iopt 4: run only for the region A (omega control- use omega is passed as alpha)
c     iopt 5: run only for the region B using the mixed boundary condition...
c         **  information from iopt 1
c         **  rho and evm_limit are dummies in iopt 2

c------------------------------------------------------------c
      subroutine fld(rho,alpha,cvm,evm_limit,deij0, !: iopt0
     $     f0,psi0,             !: iopt1
     $     fld_xy,
     $     iopt)
      include 'vpsc7.dim'
      common/FLD_MK/limit_factor,njacob_calc,dx_jac,err_fi,err_vi,
     $     max_jacob_iter
      real*8 :: limit_factor,dx_jac,err_fi,err_vi,fld_xy(2,2),jacob,
     $     theta0,theta1,omega,err_f
      integer:: njacob_calc,max_jacob_iter
      real*8:: rho,rho0,cvm,evm_limit,f0,f,psi0,pi,psi,aux2(2),
     $     aux33(3,3),bux33(3,3),aux55(5,5),cux33(3,3),
     $     aux5(5),aux3333(3,3,3,3),vi(2),deij0,deij,tincr_a,
     $     udot_a(3,3),epstot_a(3,3),sbar_a(5),epstot_a_old(3,3),
     $     sbar_a_old(5),epstot_a_0(3,3),sbar_a_0(5),
     $     epstot_a_delt(3,3),sbar_a_delt(5),tincr_a_cur,
     $     epstot_a_cur(3,3),sbar_a_cur(5),tincr_a_consumed,
     $     time,tx_3(3),vi_3(2,3),vi_0(2,7),f_obj0,f_obj1,
     $     delt_theta
      real*8:: br(3,3), brt(3,3)
      character(72)::fn_snap
      integer i,j,k,m,istp,ijv(6,2),iopt,n,nn
      logical limit,ieq,ifail
      data ((ijv(k,m),m=1,2),k=1,6)/1,1,2,2,3,3,2,3,1,3,1,2/
      data ((vi_0(i,j),i=1,2),j=1,7)
     $     / 0, 0,   1, 0,   0, 1,  -1, 0,
     $       0,-1,   1, 1,  -1,-1/
      save fn_snap

c     numerical conditions
      open(1005,file='FLD_nu.in',status='old')
      read(1005,'(a)') prosa1
      read(1005,*) limit_factor
      read(1005,*) njacob_calc
      read(1005,*) dx_jac
      read(1005,*) err_fi
      read(1005,*) err_vi
      read(1005,*) max_jacob_iter
      close(1005)

c     According to Hill (JMPS 1952, 19), there's a critical
c     band orientation given by: arctan(\sqrt(\rho))
c     -> This is not persued in vpsc-FLD

c      if (iopt.eq.1 .or. iopt.eq.0 .or. iopt.eq.3.or.iopt.eq.4) then
      if (iopt.ne.2.and.iopt.ne.5) then
         write(*,'(6a11,1x,4x)',advance='no')
     $        ('EPS'//char(48+ijv(k,1))//
     $        char(48+ijv(k,2)),k=1,6)
         write(*,'(1x,6a11,4x)',advance='no')
     $        ('SIG'//char(48+ijv(k,1))//
     $        char(48+ijv(k,2)),k=1,6)
         write(*,'(1x,6a13)') ('EDOT'//char(48+ijv(k,1))//
     $        char(48+ijv(k,2)),k=1,6)
      elseif(iopt.eq.2.or.iopt.eq.5) then
c     writing activity for region B simulations
         write(*,'(a5, 4a9, 4(1x,a6))',advance='no')
     $        'istp',
     $        ('E'//char(48+ijv(k,1))//char(48+ijv(k,2)),k=1,3),
     $        ('E'//char(48+ijv(k,1))//char(48+ijv(k,2)),k=6,6),
     $        ('S'//char(48+ijv(k,1))//char(48+ijv(k,2)),k=1,3),
     $        ('S'//char(48+ijv(k,1))//char(48+ijv(k,2)),k=6,6)
         write(*,'(5x)',advance='no')
         write(*,'(4a9, 4(1x,a6))',advance='no')
     $        ('E'//char(48+ijv(k,1))//char(48+ijv(k,2)),k=1,3),
     $        ('E'//char(48+ijv(k,1))//char(48+ijv(k,2)),k=6,6),
     $        ('S'//char(48+ijv(k,1))//char(48+ijv(k,2)),k=1,3),
     $        ('S'//char(48+ijv(k,1))//char(48+ijv(k,2)),k=6,6)
         write(*,'(14x,1x,a9,1x,a5)',advance='no')
     $        'crt','psi'
         write(*,'(1x,4x,1x,2(a9,2x))',advance='no') 'v1','v2'
         write(*,'(1x,4x,a6,2x,a3,2a6)') 'f','nit','rho_a','rho_b'
      endif
      pi   = dacos(0.d0) * 2.d0
      psi0 = psi0 * pi / 180.d0
      psi  = psi0
      f    = f0
      time = 0.d0
      tx_3(:) = 0.d0
      vi_3(:,:) = 0.d0

c      rho_sat = 1e-4

c     open files
      if (iopt.ne.2.and.iopt.ne.5) then
         open(1001,file='region_a.out',status='unknown')
         open(1002,file='region_a.bin',
     $        form='unformatted',access='sequential',
     $        status='unknown')
      elseif (iopt.eq.2.or.iopt.eq.5) then
         open(1001,file='region_a2.out',status='unknown')
         open(1002,file='region_a.bin',form='unformatted',
     $        access='sequential',status='old')
         open(1003,file='region_b.out',status='unknown')
         open(1004,file='udots.out',status='unknown')
      endif
c     -- Loop until: iopt 0,1,3,4) EVM limit; or iopt 2) Forming limit
      istp  = 1
      N     = 0
      limit = .false.
      do 1000 while (.not.limit)
         if (iopt.eq.0.or.iopt.eq.1) then
            aux2(:) = 0.d0
c            deij = (1e-04 - deij0)/(1. -0.) * epsvm +deij0
            deij = deij0
c            if (iopt.eq.1) call fld_bc(rho,cvm,aux2,0., 1,istp,deij)
            if (iopt.eq.1) call fld_bc(rho,cvm,aux2,0., 2,istp,deij)
            if (iopt.eq.0) call fld_bc(rho,cvm,aux2,0.,-1,istp,deij)
            call onestep(istp,.false.)

            call fld_write(istp,sbar,aux33,epstot,bux33,udot,
     $           cux33,aux2,0.0,0,0.,0.,tincr,sav,1)
c------------------------------------------------------------
            eps_vm = 0.
            do i=1,3
            do j=1,3
               eps_vm = eps_vm + epstot(i,j)**2
            enddo
            enddo
            eps_vm = 2./3. * eps_vm
            eps_vm = sqrt(eps_vm)
            if (eps_vm.ge.evm_limit)  then
               limit=.true.
               call write_texture
            endif
c------------------------------------------------------------
c        alpha (linear stress states)
         elseif(iopt.eq.3) then
            aux2(:) = 0.d0
            if (abs(alpha).lt.1e-3) then
               call fld_bc(rho0,cvm,aux2,0.,-1,istp,deij0)
               call onestep(istp,.false.)
            else
c     -- Guess the initial rho only for the first step
               if (istp.eq.1) call alpha2rho(alpha,rho0) !! initially guessed rho
               deij = deij0
               ieq  = .false.   !! loop to find proper 'rho' corresponding to the given alpha
               ist  = 1
               k    = 1
               do 4 while (.not.ieq)
c     write(*,*) k, 'th iteration for alpha, ','cvm:',cvm
                  if (istp.eq.1) call generate_temp_name(
     $                 '/tmp/fld_alpha0_',fn_snap)
                  call snapshot(2,fn_snap) ! save
                  call fld_alpha_f_obj(f_obj0,alpha,rho0,cvm,deij,istp)
c     -- Exit conditions
                  if (abs(f_obj0).le.1e-2) then
                     ieq = .true.
                     goto 4
                  elseif (k.gt.max_jacob_iter.and..not.(ieq)) then
                     ieq = .true.
                     goto 4
                  endif
c     -- End of Exit conditions
                  call snapshot(1,fn_snap) ! read
c     -- Calculate Jacobian (f_0 was provided when rho = rho0)
                  rho = rho0 + 1. !! forward method
                  call fld_alpha_f_obj(f_obj1,alpha,rho,cvm,deij,istp)
                  if (dabs(f_obj1).le.1e-2) then
                     ieq = .true.
                     goto 4
                  endif
                  jacob = (f_obj1 - f_obj0) / 1.
                  rho0  = rho0   - f_obj0 / jacob
                  call snapshot(1,fn_snap) ! read
                  k = k + 1
 4             continue
            endif
            call fld_write(istp,sbar,aux33,epstot,bux33,udot,
     $           cux33,aux2,0.0,0,0.,0.,tincr,sav,1)
            if (epsvm.ge.evm_limit) limit=.true.
c------------------------------------------------------------
         elseif(iopt.eq.4) then
c           Numerical recipe
            delt_theta = -1e-3
            err_f      = 1e-2
             aux2(:)    = 0.d0
c     -- Guess the initial theta (strain ratio) based on omega (stress ratio)
            if (istp.eq.1) then
               omega = alpha *  pi / 180.d0
               theta0 = datan2(2.*dsin(omega)-dcos(omega),
     $              2.*dcos(omega)-dsin(omega))
            endif
            deij = deij0
            ieq  = .false.
            k    = 1
            do 5 while (.not.ieq)
               if (istp.eq.1) call generate_temp_name(
     $              '/tmp/fld_theta0_',fn_snap)
               call snapshot(2,fn_snap) ! save
               call fld_theta_f_obj(f_obj0,
     $              theta0,omega,cvm,deij,istp)
c     -- Exit conditions
               if (abs(f_obj0).le.err_f) then
                  ieq = .true.
                  goto 5
               elseif (k.gt.max_jacob_iter.and..not.(ieq)) then
                  ieq = .true.
                  goto 5
               endif
c     -- End of Exit conditions
               call chg_basis(sav,aux33,aux55,aux3333,1,5)
               call snapshot(1,fn_snap) !!read
c     -- Calculate Jacobian (f_0 was provided when rho = rho0) 
               theta1 = theta0 + delt_theta !! forward method
               call fld_theta_f_obj(f_obj1,theta1,
     $              omega,cvm,deij,istp)
               call chg_basis(sav,bux33,aux55,aux3333,1,5)
               write(*,'(i3.3,5(f9.3,1x),''--'',4(f9.3,1x),''-o-'',
     $2(f9.3,1x))')
     $              k,omega*180/pi,theta0*180/pi,theta1*180/pi,
     $              f_obj0,f_obj1,aux33(1,1)-aux33(3,3),
     $              aux33(2,2)-aux33(3,3),bux33(1,1)-bux33(3,3),
     $              bux33(2,2)-bux33(3,3)

               if (dabs(f_obj1).le.err_f) then
                  ieq = .true.
                  goto 5
               endif
               jacob   = (f_obj1 - f_obj0) / delt_theta
               theta0  = theta0  - f_obj1 / jacob
               call snapshot(1,fn_snap) !! read
               k = k + 1
 5          continue
            call fld_write(istp,sbar,aux33,epstot,bux33,udot,
     $           cux33,aux2,0.0,0,0.,0.,tincr,sav,1)
            if (epsvm.ge.evm_limit) limit=.true.
         elseif(iopt.eq.2) then
            if (istp.eq.1) then
               epstot_a_old(:,:) = 0.d0
               sbar_a_old(:)     = 0.d0
               epstot_a_0(:,:)   = 0.d0
               sbar_a_0(:)       = 0.d0
            elseif(istp.gt.1) then
               epstot_a_old(:,:) = epstot_a(:,:)
               sbar_a_old(:)     = sbar_a(:)
               epstot_a_0(:,:) = epstot_a_old(:,:)
               sbar_a_0(:)     = sbar_a_old(:)
            endif
            ieq   = .false.
            ifail = .false.
            kst   = 1
            if (istp.eq.1) call generate_temp_name(
     $           '/tmp/fld0_',fn_snap)
c            print *,fn_snap
            call snapshot(2,fn_snap)
c           Read next state of region A
            read(1002) tincr_a,udot_a,epstot_a,sbar_a
            tincr_a_consumed = 0.d0
            n = 0 ! Multi-stage numerical solver is deprecated.
            do 100 while(.not.ieq)
               nn = 2**n
               if (nn.gt.1000) then
                  write(*,*) 'Unexpected unstability: nn exceeded 100'
                  stop
               endif
c              update deltas
               tincr_a_delt = (tincr_a - tincr_a_consumed)/nn
               do 10 i=1,3
               do 10 j=1,3
                  epstot_a_delt(i,j) = (epstot_a(i,j) -
     $                 epstot_a_old(i,j))/nn
 10            continue
               do i=1,5
                  sbar_a_delt(i) = (sbar_a(i) - sbar_a_old(i)) / nn
               enddo
c              update current states of region A
               tincr_a_cur      =tincr_a_delt
               epstot_a_cur(:,:)=epstot_a_old(:,:) + epstot_a_delt(:,:)
               sbar_a_cur(:)    =sbar_a_old(:) + sbar_a_delt(:)
               call fld_deform_b(udot_a, tincr_a_cur, sbar_a_cur,
     $              psi, f, istp, vi, n_iter, ifail)
               if (ifail) then
c                 if failed, restore to the last current values
                  if (istp.eq.1) then
                     write(*,*) 'Failed at the first increment: '
                     write(*,*) 'consider use of a finer step size'
                     stop
                  endif

                  n = n + 1 ! refine
                  write(*,*) 'n:', n, '2^n', 2**n

c                 newly guess the initial vi
                  dum = (tincr_a - tincr_a_consumed)/(2**n)
                  !call fld_guess_vi(tx_3,vi_3,vi,dum)
                  vi(:) = 0.d0
                  call snapshot(1,fn_snap) ! read
                  goto 100
               else
                  call chg_basis(sbar_a_cur,aux33,aux55,aux3333,1,5)
                  do i=1,3
                     aux33(i,i) = aux33(i,i) - aux33(3,3)
                  enddo
                  write(1001,'(3(6(e13.6,1x),2x))')
     $                 (udot_a(ijv(k,1),ijv(k,2)),k=1,6),
     $                 (epstot_a_cur(ijv(k,1),ijv(k,2)),k=1,6),
     $                 (aux33(ijv(k,1),ijv(k,2)),k=1,6)
c                  n = 0
                  tincr_a_consumed = tincr_a_consumed +
     $                 tincr_a_cur
c                 if and only if succeed, update the old ones.
c     ** Update psi & f
                  bux33(:,:) = fijph(:,:,1)
                  call fld_update(psi,psi0,f,f0,
     $                 epstot_a_cur,epstot,bux33)
                  call fld_write(istp,sbar_a_cur,sbar,
     $                 epstot_a_cur,epstot,udot_a,udot,vi,
     $                 f,n_iter,psi,time,0.,aux33,2)

                  epstot_a_old(:,:) = epstot_a_cur(:,:)
                  sbar_a_old(:)   = sbar_a_cur(:)
                  time = time + tincr_a_cur

c                 guess vi for the next run
                  tx_3(1) = tx_3(2)
                  tx_3(2) = tx_3(3)
                  tx_3(3) = time
                  vi_3(:,1) = vi_3(:,2)
                  vi_3(:,2) = vi_3(:,3)
                  vi_3(:,3) = vi(:)
                  call fld_guess_vi(tx_3,vi_3,vi,tincr_a_cur) !

c                 ** Limit criteria
                  crt = abs(udot(3,3))/abs(udot_a(3,3))
                  if (crt.ge.limit_factor) then
                     limit = .true.
                     fld_xy(1,1) = epstot_a_cur(1,1)
                     fld_xy(1,2) = epstot_a_cur(2,2)
                     fld_xy(2,1) = aux33(1,1)
                     fld_xy(2,2) = aux33(2,2)
                     goto 1000
                  endif
                  if (dabs(tincr_a_consumed-tincr_a).le.1e-20) then
                     n = 0
                     ieq = .true.
                     goto 100
                  else
                     kst = kst + 1
                     call snapshot(2,fn_snap) ! save
                  endif
               endif
 100        continue
c         elseif (iopt.eq.3) then
         elseif (iopt.eq.5) then
c     mixed boundary condition for region B.
            if (istp.eq.1) then
               epstot_a_old(:,:) = 0.d0
               sbar_a_old(:)     = 0.d0
               epstot_a_0(:,:)   = 0.d0
               sbar_a_0(:)       = 0.d0
               f                 = f0
               psi               = psi0
               epstot(:,:) = epstot_a(:,:)
            elseif(istp.gt.1) then
               epstot_a_old(:,:) = epstot_a(:,:)
               sbar_a_old(:)     = sbar_a(:)
               epstot_a_0(:,:) = epstot_a_old(:,:)
               sbar_a_0(:)     = sbar_a_old(:)
            endif

c           Read REGION A states
            read(1002) tincr_a,udot_a,epstot_a,sbar_a
c     ---
c     Determine the rotation mat for n/t axes.
c     ---
c     br[i,j] rotates a (x,y) coordinate referred in [Sample axes] to
c     that referred in n/t axes.
            call band_r(psi,br)
            call matranspose(br,brt)
c     R[i,k] * ag[k,j] ;   R[new sa <- old sa]  ag[sa<-ca]
            call texture_rotation(br)
c        1) epstot
            call matrot(epstot,br,aux33)
            epstot(:,:) = aux33(:,:)
c        2) udot
            call matrot(udot,br,aux33)
            udot(:,:) = aux33(:,:)
c        3) sbar
            call mat5rot(sbar,br,aux5)
            sbar=aux5(:)

            call fld_bc_b(udot_a,sbar_a,psi,f,tincr_a)
            call onestep(istp, .false.)
c     Check if stress is in balance here.
c            call check_force_eq2(sbar_a,sbar,psi,f) !! sbar_a (1-2 axes) / sbar(n-t) axes

c     Rotate back to the stretching axes of region A
            call texture_rotation(brt)
            aux2(:)= 0.0
            kst = 0

c     Rotate tensors of region B to the reference frame
c     attached to region A...
c     brt[i,j] rotates xy coordinates in [n/t] axes to [sample axes].
            call band_r(psi,br)
            call matranspose(br,brt)
c        1) epstot
            call matrot(epstot,brt,aux33)
            epstot(:,:) = aux33(:,:)
c        2) udot
            call matrot(udot,brt,aux33)
            udot(:,:) = aux33(:,:)
c        3) sbar
            call mat5rot(sbar,brt,aux5)
            sbar(:) = aux5(:)
            aux2(:) = 0.
            call fld_write(istp,sbar_a,sbar,
     $           epstot_a,epstot,udot_a,udot,aux2,
     $           f,kst,psi,tincr_a,tincr,aux33,2)
            dTime = abs(tincr_a-tincr)/tincr_a
            if (dTime.gt.1e-4) write(*,*)
     $           'Warning time sync error [%] is a bit large .. ', dTime
c----------------------------------------------------------------------
c     Check if the force equilibrium and compatibility conditions are met.
c            call check_force_eq(sbar_a,sbar,psi,f)
c----------------------------------------------------------------------
c     check if compatibility is met...
c            call check_compat(epstot_a,epstot,psi,0)
            call check_compat(udot_a,udot,psi,1)
c----------------------------------------------------------------------
            bux33(:,:) = fijph(:,:,1)
c            write(*,*)'Deformation gradient...'
c            write(*,'(3(e9.3,2x))') ((bux33(i,j),j=1,3),i=1,3)
            call fld_update(psi,psi0,f,f0,
     $           epstot_a,epstot,bux33)
            crt = abs(udot(3,3))/abs(udot_a(3,3))
            if (crt.ge.limit_factor) then
               limit = .true.
               fld_xy(1,1) = epstot_a_cur(1,1)
               fld_xy(1,2) = epstot_a_cur(2,2)
               fld_xy(2,1) = aux33(1,1)
               fld_xy(2,2) = aux33(2,2)
               goto 1000
            endif
         endif ! iopt 0, 1, 2 or 3
         istp = istp + 1
 1000 continue                  ! end of iopt 1 or iopt 2 process
c     close files
      close(1001)
      close(1002)
      if (iopt.eq.2) then
         close(1003)
         close(1004)
      endif
      return
      end subroutine fld
c------------------------------------------------------------c
      subroutine fld_guess_vi(t,v,vi_guess,dt)
      real*8 t(3), v(2,3),vi_guess(2),dt,
     $     slope(2),dy(2)
      integer i
      if (t(1)*t(2)*t(3).ne.0) then
         ! linear interpolation??? or parabolic?
         do i=1,2
            slope(i) = (v(i,3)-v(i,2))/(t(3) - t(2))
         enddo
         dy(:) = 0.d0
         do i=1,2
            dy(i) = slope(i) * dt
         enddo
         do i=1,2
            vi_guess(i) = v(i,3) + dy(i)
         enddo
      else ! use the last one as the guess for next
         vi_guess(:) = v(:,3)
      endif
c      vi_guess(:) = 0.d0 ! over-ride
      return
      end subroutine fld_guess_vi
c------------------------------------------------------------c
c     Updates necessary to deal with when L12 is relaxed.
      subroutine fld_update(psi,psi0,f,f0,
     $     epstot_a,epstot_b,fij)
      real*8 psi,psi0,f,f0,epstot_a(3,3),epstot_b(3,3),
     $     pi,n(3),t(3),fij(3,3),si(3)
      pi   = dacos(0.d0) * 2.d0
c     ** Updates thickness ratio
      f = f0 * dexp(epstot_b(3,3)-epstot_a(3,3))
c$$$  following Kuroda&Tvergaard 2000 IJSS
c$$$  Initial tangential vector S^I
      call nt_psi(psi0,n,si)
      t(1) = fij(1,1) * si(1) + fij(1,2) * si(2)
      t(2) = fij(2,1) * si(1) + fij(2,2) * si(2)
      n(1) = 1./ dsqrt(si(1)**2 + si(2)**2) * t(2)
      n(2) =-1./ dsqrt(si(1)**2 + si(2)**2) * t(1)
      psi  = datan2(n(2),n(1))
      return
      end subroutine fld_update
c------------------------------------------------------------c
      subroutine fld_write(istp,sbar_a,sbar_b,epstot_a,epstot_b,
     $     udot_a,udot_b,vi_final,f,n_est,psi,time,tincr_a,
     $     sav_a,iopt)
      real*8 sbar_a(5),sbar_b(5),epstot_a(3,3),epstot_b(3,3),
     $     udot_a(3,3),udot_b(3,3),vi_final(2),f,psi,crt,
     $     aux33(3,3),bux33(3,3),aux55(5,5),aux3333(3,3,3,3),
     $     pi,tincr_a,sav_a(5),time,udot_sym(3,3),rho,rho_b
      integer i,k,n_est,ijv(6,2),istp,iopt
      data ((ijv(k,m),m=1,2),k=1,6)/1,1,2,2,3,3,2,3,1,3,1,2/
      character(100)::fmt
      pi   = dacos(0.d0) * 2.d0
      fmt  = '(6e13.4,1x,6f11.6,1x,6f8.2)'

c     ** Writing results
      if    (iopt.eq.1) then
         call chg_basis(sbar_a,aux33,aux55,aux3333,1,5)
         do 10 i=1,3
         do 10 j=1,3
            udot_sym(i,j) = 0.5 * (udot_a(i,j) + udot_a(j,i))
 10      continue
         do i=1,3
            aux33(i,i) = aux33(i,i) - aux33(3,3)
         enddo
         write(*,'(6f11.7,1x,'' || '',1x,6f11.5,'' || '',1x,6e13.5)')
     $        (epstot_a(ijv(k,1),ijv(k,2)),k=1,6),
     $        (aux33(ijv(k,1),ijv(k,2)),k=1,6),
     $        (udot_sym(ijv(k,1),ijv(k,2)),k=1,6)
c     writing to human-readable file
         write(1001,fmt)
     $        (udot_a(ijv(k,1),ijv(k,2)),k=1,6),
     $        (epstot_a(ijv(k,1),ijv(k,2)),k=1,6),
     $        (aux33(ijv(k,1),ijv(k,2)),k=1,6)
c     writing to binary file (to be used for region_b calc)
         write(1002) tincr_a,udot_a,epstot_a,sav_a
c        save to binary file
c$$$         write(fn_snap,'(i5.5)') istp
c$$$         fn_snap = 'region_a_'//trim(fn_snap)//'_aux'
c$$$         call snapshot(2,fn_snap)
      elseif(iopt.eq.2) then
         call chg_basis(sbar_b,bux33,aux55,aux3333,1,5)
         call chg_basis(sbar_a,aux33,aux55,aux3333,1,5)
         if (udot_a(1,1).ge.udot_a(2,2)) then
            rho = udot_a(2,2) / udot_a(1,1)
         else
            rho = udot_a(1,1) / udot_a(2,2)
            rho = (1. -  rho) + 1
         endif
         if (udot_b(1,1).ge.udot_b(2,2)) then
            rho_b = udot_b(2,2) / udot_b(1,1)
         else
            rho_b = udot_b(1,1) / udot_b(2,2)
            rho_b = (1. -  rho_b) + 1
         endif
         do i=1,3
            bux33(i,i) = bux33(i,i) - bux33(3,3)
            aux33(i,i) = aux33(i,i) - aux33(3,3)
         enddo
         write(*,'(i5, 4f9.4,4(1x,f6.0))',advance='no') istp,
     $        (epstot_a(ijv(k,1),ijv(k,2)),k=1,3),
     $        (epstot_a(ijv(k,1),ijv(k,2)),k=6,6),
     $        (aux33(ijv(k,1),ijv(k,2)),k=1,3),
     $        (aux33(ijv(k,1),ijv(k,2)),k=6,6)
         write(*,'(a5)',advance='no') '  || '
         write(*,'(4f9.4,4(1x,f6.0))',advance='no')
     $        (epstot_b(ijv(k,1),ijv(k,2)),k=1,3),
     $        (epstot_b(ijv(k,1),ijv(k,2)),k=6,6),
     $        (bux33(ijv(k,1),ijv(k,2)),k=1,3),
     $        (bux33(ijv(k,1),ijv(k,2)),k=6,6)
         crt = udot_b(3,3)/udot_a(3,3)
         write(*,'(a14,1x,e9.3,1x,f6.1)',advance='no')
     $        'D33(B)/D33(A)',crt,psi*180./pi
         write(*,'(1x,a4,1x,2(e9.3,2x))',
     $        advance='no') 'vi:', vi_final
         write(*,'(1x,a4,f6.4,2x,i3,2x,2f6.2)',
     $        advance='no') 'f:', f,n_est,rho,rho_b
         write(*,*)
         write(1003,fmt)
     $        (udot_b(ijv(k,1),ijv(k,2)),k=1,6),
     $        (epstot_b(ijv(k,1),ijv(k,2)),k=1,6),
     $        (bux33(ijv(k,1),ijv(k,2)),k=1,6)
         write(1004,'(e14.7,1x,f6.1,2x,2(9(e14.7,1x),2x))')
     $        time, psi, ((udot_a(i,j),j=1,3),i=1,3),
     $        ((udot_b(i,j),j=1,3),i=1,3)
      endif
      return
      end subroutine fld_write
c------------------------------------------------------------c
c     Core of the FLD analysis
c     Find the boundary condition (v1, v2) of region B
c     that satifies the force equilibrium

c     divide the stress increment resulting from the
c     deformation occured in region A
c************************************************************c
c     See if the forming limit criterion is met
c       * Open question: should we avoid overshoot??

c     Update band orientation (psi)
c     Update inhomogeneity parameter (f)
c------------------------------------------------------------c
      subroutine fld_deform_b(udot_a,tincr_a,sbar_a,
     $     psi,f,istp,vi,n_iter,ifail)
      include 'vpsc7.dim'
      common/FLD_MK/limit_factor,njacob_calc,dx_jac,err_fi,err_vi,
     $     max_jacob_iter
      real*8 :: limit_factor,dx_jac,err_fi,err_vi
      integer:: njacob_calc,max_jacob_iter
      real*8 udot_a(3,3),tincr_a,sbar_a(5),psi,f,jac(2,2),fi(2),
     $     vi_old(2),vi(2),!n(3),t(3),
     $     f1_0,f2_0,aux55(5,5),aux3333(3,3,3,3),
     $     sa33(3,3),sb33(3,3),f0(2),dum,
     $     j_inv(2,2)
      integer i,k,istp,ist,n_iter
      logical ieq,ifail,verbose
      character*72 fn_snap
      character*100 fmt
      save jac, vi_old,fn_snap !, dx_jac
c      save vi_old

      ifail = .false.
      if (istp.eq.1) then
         jac(:,:) = 0.d0
         vi(:)    = 0.d0
      else
         vi(:) = vi(:)
         jac(:,:) = 0.d0
      endif
c------------------------------------------------------------c
      call chg_basis(sbar_a,sa33,aux55,aux3333,1,5)
      do i=1,3
         sa33(i,i) = sa33(i,i) - sa33(3,3)
      enddo
c      verbose     = .true.
      verbose     = .false.
      ieq         = .false.
      ist         = 1
      fi(:) = 0.d0
      f1_0  = 0.d0
      f2_0  = 0.d0
      tincr = tincr_a

      if (verbose) then
         write(*,*) ' vi determination'
         write(*,'(a5,2x,3(a11,1x),2x,2(a11,1x),
     $2x,4(a11,1x),
     $2x,a11,2x,5(a11,1x))')

     $        'istep','v1','v2','|d_v|','f1','f2','|f1|+|f2|',
     $        'n.sa.n','n.sb.n.F','n.sa.t','n.sb.t.F',
     $        'dx_jac','j11','j12','j21','j22'
      endif
      fmt = '(i5,2x,3(e11.4,1x),2x,2(e11.4,1x),'//
     $     '2x,4(e11.4,1x),2x,e11.4,2x,5(e11.4,1x))'

      vi_chg = 0.
      do 100 while (.not.ieq)
         if (istp.eq.1) call generate_temp_name(
     $        '/tmp/fld_region_b0_',fn_snap)
         call snapshot(2,fn_snap) ! save
         call fld_fi_obj(fi,udot_a,vi,sa33,sb33,psi,
     $        f,istp,f1_0,f2_0,ieq,err_fi)
         call fld_fi(fi,sa33,sb33,psi,f,dum1,dum2,dum3,dum4)
         if (verbose) write(*,fmt)
     $        ist,vi,dabs(vi_chg),fi,dabs(fi(1))+dabs(fi(2)),
     $        dum1,dum2,dum3,dum4,
     $        dx_jac,j_inv(1,1),j_inv(1,2),j_inv(2,1),j_inv(2,2)

cccc    -- Exit conditions below --
         if (ieq.or.(vi_chg.le.err_vi.and.ist.gt.1)) then
c            if (ist.eq.1) write(*,*) 'Equilibrium right away'
            ieq = .true.
            goto 100
         else
            call snapshot(1,fn_snap) ! read
         endif
         if (ist.ge.max_jacob_iter.and..not.(ieq)) then
c            just skip this
c            ifail = .true.
c            call snapshot(1,fn_snap) ! read
            ieq   = .true.
            goto 100
         endif
c        --
         ! calc Jacobian (Fi is provided as f0)
         f0(:) = fi(:)

         if (mod(ist,njacob_calc).eq.0 .or. ist.eq.1) then
            call fld_jac(jac,f0,udot_a,vi,sa33,sb33,psi,f,ieq,
     $           f1_0,f2_0,dx_jac,err_fi,istp)
            j_inv(:,:) = jac(:,:)
c           improvement can be made by using LU decomposition
c           to solve the equation with jac instead of calc j_inv
            call lu_inverse(j_inv,2)
         endif
c        --
c        update next vi --
         vi_old(:) = vi(:)
         vi(:) = 0.d0
         vi_chg = 0.d0
         do i=1,2
            dum = 0.d0
            do k=1,2
               dum = dum + j_inv(i,k) * fi(k)
            enddo
            vi(i) = vi_old(i) - dum
            if (dabs(dum).gt.vi_chg) vi_chg = dabs(dum)
         enddo

         ist = ist + 1
 100  continue                  ! while (.not.ieq)

      n_iter = ist
      vi_old(:) = vi(:)

      return
      end subroutine fld_deform_b

c------------------------------------------------------------c
      subroutine fld_fi_obj(fi,udot_a,vi,sa33,sb33,psi,h,istp,
     $     f1_0,f2_0,ieq,tiny)
c     Given vi, calculation Fi, the objective function
      include 'vpsc7.dim'
      real*8:: fi(2),vi(2),sa33(3,3),sb33(3,3),psi,h,f1_0,
     $     f2_0,tiny,dum1,dum2,dum3,dum4,
     $     aux55(5,5),
     $     aux3333(3,3,3,3),dum,udot_a(3,3)
      integer i
      logical ieq
      ieq = .false.
      dum = 0.d0
      udot(:,:) = udot_a(:,:)
      call fld_bc(0.,0.,vi,psi,0,istp,dum) ! boundary for region b
      call onestep(istp,.false.)
      call chg_basis(sav,sb33,aux55,aux3333,1,5)
c     plane-stress projection for sheets
      do i=1,3
         sb33(i,i) = sb33(i,i) - sb33(3,3)
      enddo
c--
      call fld_fi(fi,sa33,sb33,psi,h,dum1,dum2,dum3,dum4)
c     In case fi need to get close to non-zero values
      fi(1) = fi(1) - f1_0
      fi(2) = fi(2) - f2_0
c--
      if (dabs(fi(1)) + dabs(fi(2)).le.tiny) ieq=.true.
      return
      end subroutine fld_fi_obj
c------------------------------------------------------------c
      subroutine fld_fi(fi,sa33,sb33,psi,h,dum1,dum2,dum3,
     $     dum4)
      real*8 sa33(3,3),sb33(3,3),psi,n(3),t(3),fi(2),h,
     $     dum1,dum2,dum3,dum4
      integer i,j
      call nt_psi(psi,n,t)
      fi(:) = 0.d0
      dum1 = 0.
      dum2 = 0.
      dum3 = 0.
      dum4 = 0.
      do 1 i=1,2
      do 1 j=1,2
         dum1 = dum1 + n(i) * sa33(i,j) * n(j)
         dum2 = dum2 + n(i) * sb33(i,j) * n(j) * h
         dum3 = dum3 + n(i) * sa33(i,j) * t(j)
         dum4 = dum4 + n(i) * sb33(i,j) * t(j) * h
 1    continue
      fi(1) = dum1 - dum2 ! normal     component; snn(A) - snn(b) *f
      fi(2) = dum3 - dum4 ! tangential component; snt(A) - snt(b) *f
      return
      end subroutine fld_fi
c------------------------------------------------------------c
c     Numerical Jacobian Calculation
      subroutine fld_jac(jac,fi,udot_a,xi,sa33,sb33,psi,h,ieq,
     $     f1_0,f2_0,delt_x,tiny,istp)
      real*8 jac(2,2),fi(2),xi(2),sa33(3,3),sb33(3,3),psi,
     $     h,f1_0,f2_0, delt_x,f0(2),f_v1(2),f_v2(2),aux33(3,3),
     $     udot_a(3,3)
      real*8 x(2)
      integer istp
      logical ieq
      character *72 fn_snap
      save fn_snap
      X(:)     = xi(:)
      jac(:,:) = 0.d0
      f0(:)    = fi(:)

c     f_v1
      X(:) = xi(:)
      X(1) = X(1) + delt_x
      if (istp.eq.1) call generate_temp_name(
     $     '/tmp/fld_jac0_',fn_snap)
      call snapshot(2,fn_snap) ! save
      aux33(:,:) = sb33(:,:)
      call fld_fi_obj(f_v1,udot_a,X,sa33,aux33,psi,h,istp,
     $     f1_0,f2_0,ieq,tiny)
      call snapshot(1,fn_snap) ! read

c     f_v2
      X(:) = xi(:)
      X(2) = X(2) + delt_x
      call snapshot(2,fn_snap) ! delete this line
      aux33(:,:) = sb33(:,:)
      call fld_fi_obj(f_v2,udot_a,X,sa33,aux33,psi,h,istp,
     $     f1_0,f2_0,ieq,tiny)
      call snapshot(1,fn_snap) ! read

      jac(1,1) = (f_v1(1) - f0(1))/delt_x
      jac(1,2) = (f_v2(1) - f0(1))/delt_x
      jac(2,1) = (f_v1(2) - f0(2))/delt_x
      jac(2,2) = (f_v2(2) - f0(2))/delt_x

c     Below is slower as it requires 4 runs of fld_fi_obj
c$$$      fi(:) = 0.d0
c$$$      do 100 i=1,2
c$$$      do 100 j=1,2
c$$$         X(j) = X(j) + delt_x
c$$$         f1(:) = 0.d0
c$$$         call fld_fi_obj(f1,X,sa33,sb33,psi,h,rho,init,ieq,0)
c$$$         if (ieq) then
c$$$            xi(:) = X(:)
c$$$            goto 200
c$$$         endif
c$$$         jac(i,j) = (f1(i) -f0(i))/delt_x
c$$$         X(:) = xi(:)
c$$$ 100  continue
c$$$ 200  return

      return
      end subroutine fld_jac
c------------------------------------------------------------c
c     set boundary condition for region B with mixed boundary
c     condition. Following is the boundary condition in the
c     axes of (n-t) band. Thus, following assumes that the
c     stretching axes, to which the boundary condition is
c     referred to, are aligned with (n-t) axes such that
c     n//axis1, t//axis2
      subroutine fld_bc_b(udot_a,sbar_a,psi,f,tincr_a)
      include 'vpsc7.dim'
c     Given arguments (indent:in)
      real*8 :: udot_a(3,3),sbar_a(5),psi,f,tincr_a
c     To-be-determined variables
      real*8 :: sigma_a(3,3)!,sigma_b(3,3)
      real*8 :: sigma_a_refb(3,3) !! sigma a referred in n/t
      real*8 ::  udot_a_refb(3,3) !! udot  a referred in n/t
      real*8 :: br(3,3)
c     other misc
      integer:: i,j,k,m,ijv(6,2)
      real*8:: n(3), t(3), b_sigma_b(3,3), b_udot_b(3,3),
     $     aux55(5,5),aux3333(3,3,3,3),pi
      data ((ijv(k,m),m=1,2),k=1,6)/1,1,2,2,3,3,2,3,1,3,1,2/
      pi   = dacos(0.d0) * 2.d0
c     # plane stress condition
      call chg_basis(sbar_a,sigma_a,aux55,aux3333,1,5)
      do i=1,3
         sigma_a(i,i) = sigma_a(i,i)  - sigma_a(3,3)
      enddo
      call band_r(psi,br)
      call nt_psi(psi,n,t)
c---------------------------------------------------------------------
c     Mixed boundary condition referred in n/t axes
c---------------------------------------------------------------------
c     Stress in the reference system tied to the band (n,t)/(1,2) axes
c     o: knowns
c     x: unknowns
c---------------------------------------------------------------------
c                 | x   b_udot_nt  0  |                   | x  o  o |
c     b_udot_b  = | x   b_udot_tt  0  |          iudot =  | x  o  o |
c                 | x       x      x  |                   | x  x  x |
c---------------------------------------------------------------------
c                 | (1/f)s_nn  (1/f)s_nt   0 |            | o  o  o |
c     b_sigma_b = | (1/f)s_nt      x       0 |   iscau =  |    x  o |
c                 |    0           0       0 |            |       o |
c---------------------------------------------------------------------
c                 | x  ?           x |                    | x  x  x |
c     b_sr_b    = |    b_udot_tt   x |           idsim =  |    o  x |
c                 |                x |                    |       x |
c---------------------------------------------------------------------
c     Refer udot_a  and sigma_a in n/t axes
      call matrot(udot_a, br, udot_a_refb)
      call matrot(sigma_a,br,sigma_a_refb)

c$$$      write(*,'(3(f5.1,1x))') sigma_a(1,1), sigma_a(2,2),sigma_a(1,2)
c$$$      write(*,'(3(f5.1,1x))') sigma_a_refb(1,1), sigma_a_refb(2,2),
c$$$     $     sigma_a_refb(1,2)


c     Force equilibrium
      b_sigma_b(:,:) = 0.
      b_sigma_b(1,1) = 1./f * sigma_a_refb(1,1)
      b_sigma_b(1,2) = 1./f * sigma_a_refb(1,2)
      b_sigma_b(2,1) = b_sigma_b(1,2)

c     (1,3) and (2,3) components?
      b_sigma_b(1,3) = sigma_a_refb(1,3)/f
      b_sigma_b(3,1) = b_sigma_b(1,3)/f
      b_sigma_b(2,3) = sigma_a_refb(2,3)/f
      b_sigma_b(3,2) = b_sigma_b(2,3)/f


c$$$      write(*,'(a)')'--------------------------------------------------'
c$$$      write(*,'(4(a24,5x))') 'S^A','[b] S^A', '[b] S^B', '[b] f*S^B'
c$$$      do i=1,3
c$$$         write(*,'(4(3(f6.2,2x),5x))') (sigma_a(i,j),j=1,3),
c$$$     $        (sigma_a_refb(i,j),j=1,3),(b_sigma_b(i,j),j=1,3),
c$$$     $        (f*b_sigma_b(i,j),j=1,3)
c$$$      enddo
c$$$      write(*,'(a)')'--------------------------------------------------'

c     compatibility
      b_udot_b(:,:) = 0.
      b_udot_b(2,2) = udot_a_refb(2,2)
      b_udot_b(1,2) = udot_a_refb(1,2)
c     ## below are dummies since they are *unknowns*.
      b_udot_b(2,1) = udot_a_refb(2,1)
      b_udot_b(1,1) = udot_a_refb(1,1) ! 0.
      b_udot_b(3,3) =-b_udot_b(1,1)-b_udot_b(2,2)
      udot(:,:) = b_udot_b(:,:)

c     boundary components flags.
c     iudots
      iudot(:,:) = 0
      iudot(1,2) = 1
      iudot(1,3) = 1
      iudot(2,2) = 1
      iudot(2,3) = 1
c     iscau
      iscau(:) = 1
      iscau(2) = 0

c     Misc. flags.
c     strain-rate
      do 30 i=1,3
      do 30 j=1,3
         dsim(i,j) = (udot(i,j) + udot(j,i))/ 2.d0
 30   continue
      do k=1,6
         i=ijv(k,1)
         j=ijv(k,2)
         idsim(k) = iudot(i,j) * iudot(j,i)
      enddo

      scauchy(:,:) = b_sigma_b(:,:)
c     Below might be wrong if udot(2,2) is zero...

      call chg_basis(dbar,dsim,aux55,aux3333,2,5)
      call chg_basis(sbar,scauchy,aux55,aux3333,2,5)
      DBARNORM=VNORM(DBAR,5)
      SBARNORM=VNORM(SBAR,5)

      ictrl = 7
      ctrlincr=tincr_a

      IF(ICTRL.EQ.0) THEN
        IF(DBARNORM.NE.0.) STRAIN_CONTROL=1
        IF(DBARNORM.EQ.0.) THEN
          WRITE(*,*) 'CAN CONTROL VON MISES ONLY IF STRAIN IS IMPOSED !'
          STOP
        ENDIF
      ELSE IF(ICTRL.GE.1 .AND. ICTRL.LE.6) THEN
        IF(IDSIM(ICTRL).EQ.1 .AND. DBARNORM.NE.0.) STRAIN_CONTROL=1
        IF(ISCAU(ICTRL).EQ.1 .AND. SBARNORM.NE.0.) STRAIN_CONTROL=0
      ELSE IF(ICTRL.EQ.7) THEN
        IF(DBARNORM.NE.0.) STRAIN_CONTROL=1
      ENDIF

C *** CHECKS WHETHER THE BOUNDARY CONDITIONS ARE CONSISTENT.
      if(iudot(1,1)+iudot(2,2)+iudot(3,3).eq.2) then
        write(*,*) 'WARNING: CHECK DIAGONAL BOUNDARY CONDITIONS IUDOT'
        write(*,*) '         ENFORCING TWO DEVIATORIC COMPONENTS FIXES'
        write(*,*) '         THE OTHER BECAUSE OF INCOMPRESSIBILITY'
        print *, 'enter c to continue'
        read  *
      endif

      do i=1,2
      do j=i+1,3
        if(iudot(i,j)+iudot(j,i).eq.0) then
          write(*,*) 'CHECK OFF-DIAGONAL BOUNDARY CONDITIONS IUDOT'
          write(*,*) 'CANNOT RELAX BOTH OFF-DIAGONAL COMPONENTS'
          stop
        endif
      enddo
      enddo

      DILAT=UDOT(1,1)+UDOT(2,2)+UDOT(3,3)
      IF(DILAT.GT.1.E-6) THEN
        WRITE(*,*) 'CHECK DIAGONAL STRAIN RATE COMPONENTS UDOT'
        WRITE(*,*) 'THE IMPOSED RATE IS NOT INCOMPRESSIBLE'
        STOP
      ENDIF

      DO I=1,6
        IF(ISCAU(I)*IDSIM(I).NE.0 .OR. ISCAU(I)+IDSIM(I).NE.1) THEN
          WRITE(*,*) ' CHECK BOUNDARY CONDITS ON STRAIN-RATE AND STRESS'
          WRITE(*,'('' IDSIM = '',6I3)') IDSIM
          WRITE(*,'('' ISCAU = '',6I3)') ISCAU
          STOP
        ENDIF
      ENDDO

C *** DEFINES CONTROLLING INCREMENT AND ESTIMATES TIME INCREMENT.
C *** ULTIMATE 'TINCR' WILL BE CALCULATED AT THE END OF EACH STEP.

      IF(ICTRL.EQ.0 .AND. STRAIN_CONTROL.EQ.1) THEN
        EVMINCR=CTRLINCR
        TINCR  =EVMINCR/VNORM(DBAR,5)
      ELSE IF(ICTRL.GE.1.AND.ICTRL.LE.6.AND.STRAIN_CONTROL.EQ.1) THEN
        EIJINCR=CTRLINCR
        DSIMCTRL=DSIM(IJV(ictrl,1),IJV(ictrl,2))
        TINCR  =EIJINCR/ABS(DSIMCTRL)
        IF(.NOT.(IDSIM(ICTRL).EQ.1. AND. DSIMCTRL.NE.0.)) THEN
          WRITE(*,*) 'ICTRL        =',ICTRL
          WRITE(*,*) 'IDSIM(ICTRL) =',IDSIM(ICTRL)
          WRITE(*,*) 'DSIM (ICRTL) =',DSIMCTRL
          WRITE(*,*) 'ICTRL MUST BE 0 TO CONTROL VON MISES INCREMENT'
          WRITE(*,*) 'OR IT MUST IDENTIFY A NON-ZERO STRAIN COMPONENT'
          WRITE(*,*) 'OR IT MUST BE 7 TO IMPOSE DSIM*TINCR'
          STOP
        ENDIF
      ELSE IF(ICTRL.EQ.7 .OR. STRAIN_CONTROL.EQ.0) THEN
        TINCR=CTRLINCR
      ENDIF

      ISCAUSUM=ISCAU(1)+ISCAU(2)+ISCAU(3)+ISCAU(4)+ISCAU(5)+ISCAU(6)
      IF(INTERACTION.LE.0 .AND. ISCAUSUM.NE.0) THEN
        WRITE(*,*) ' CANNOT IMPOSE STRESS COMPONENTS FOR FC or RC CASE'
        STOP
      ENDIF



      return
      end subroutine fld_bc_b
c------------------------------------------------------------c
c     Set boundary condition for a single rho (single path)
c     Determine boundary condition of region A
c     based on the given rho & strain rate
c     (c=VM(Edot_i)) under a constant Von Mises strain rate

c     In the code, range of the given rho defines the strain
c     paths spanning the uniaxial tensile to balanced biaxial.
c      (-0.5 to 1.0) 1-axis being major tensile direction
c      (1.0  to 2.5) 2-axis being major tensile direction
c     Note that (1.0 to 2.5) is transformed to (1.0 to -0.5)
c     in which 2-axis is being the major tensile direction

c     iopt.eq.-1
c        Boundary condition for uniaxial stress test
c     iopt.eq.0
c        Boundary condition for region B
c     iopt.eq.1
c        Boundary condition for region A
c     iopt.eq.2
c        Boudnary condition for region A but allowing
c        in-plane rotation: L12!=0 thus sig12=0
c     iopt.eq.3
c        Boundary condition for region A allowing
c        in-plane rotation: L12!=0 thus sig12=0
c        and use rho as 'omega angle' between sigma11/sigma22
c        WIth that angle spanning (-pi,pi), one could cover
c        all in-plane strain ratio (including compression)
c        - useful when probing work-contours

c     In case the stretching is not occuring in the axes
c     that is inclined with the orthotropic anisotropic
c     axes of the material, it may be necessary to
c     relax the in-plane shear stress (sig12) such that
c     sig12=0 by allowing in-plane rotation, i.e., w12!=0.

      subroutine fld_bc(rho,c_vm,vi,psi,iopt,istp,deij)
      include 'vpsc7.dim'
      real*8:: rho,c_vm,vi(2),psi,n(3),t(3),aux55(5,5),deij,
     $     theta
      integer:: i,j,iopt,ijv(6,2),istp
c     iopt0    : region B specific
c         -> rho, c_vm, deij are dummies
c     iopt1/2/3: region A specific
c         -> psi, vi         are dummies
      data ((ijv(k,m),m=1,2),k=1,6)/1,1,2,2,3,3,2,3,1,3,1,2/
c     ** Define controlled vel. grad. components
      iudot(:,:) = 1
      iudot(3,3) = 0            !! free surface 33: plane-stress condition
      do k=1,6
         i = ijv(k,1)
         j = ijv(k,2)
         idsim(k) = iudot(i,j) * iudot(j,i)
      enddo
c     ** Define free surface
      if (iopt.eq.0.and.istp.eq.1) scauchy(:,:) = 0
      if (iopt.eq.1.or.iopt.eq.2.or.iopt.eq.3) scauchy(:,:) = 0
      iscau(:) = 0
      iscau(3) = 1
c     The above might be overriden later.

      if (iopt.eq.-1) then
c     Conduct a uniaxial stress test.
c     Overwrites the BC
         iudot(:,:) = 0
         iudot(1,1) = 1
         iudot(1,2) = 1
         iudot(1,3) = 1
         iudot(2,3) = 1
         udot(:,:)  = 0.
         udot(1,1)  = c_vm
         udot(2,2)  =-c_vm/2.
         udot(3,3)  =-c_vm/2.

         do k=1,6
            i = ijv(k,1)
            j = ijv(k,2)
            idsim(k) = iudot(i,j) * iudot(j,i)
         enddo

c     Free surface
         iscau(:) = 1
         iscau(1) = 0
         scauchy(:,:) = 0.

         udot_vm = 0.d0
         do i=1,3
            udot_vm = udot_vm + udot(i,i)**2
         enddo
         udot_vm = dsqrt(2./3.) * dsqrt(udot_vm)
         do i=1,3
            udot(i,i) = udot(i,i) * c_vm / udot_vm
         enddo
         ictrl=1
c         if (abs(udot(2,2)).ge.abs(udot(1,1))) ictrl = 2
c     ** Define strain rate based on the velocity gradient
         do 1 i=1,3
         do 1 j=1,3
            dsim(i,j) = (udot(i,j) + udot(j,i))/ 2.d0
 1       continue
         dsim_vm = 0.d0
         do i=1,3
            dsim_vm = dsim_vm +  dsim(i,i)**2
         enddo
         dsim_vm = dsqrt(2./3.) * dsqrt(dsim_vm)
         call chg_basis(dbar,dsim,aux55,aux3333,2,5)
         call chg_basis(sav,scauchy,aux55,aux3333,2,5)
         strain_control = 1
c         eijincr = 1e-2         ! should determine for VPSC
         eijincr = deij
         dsimctrl = dsim(ijv(ictrl,1), ijv(ictrl,2))
         if (dsimctrl.eq.0) then
            write(*,*) 'Wrong'
            stop
         endif
         tincr    = eijincr/dabs(dsimctrl)

c        Relax L12 in order to allow the in-plane
c        rigid body rotation
c$$$         if (iopt.eq.2.or.iopt.eq.3.or.iopt.eq.1) then
c$$$            iudot(2,1) = 1
c$$$            iudot(1,2) = 0
c$$$            iscau(6)   = 1
c$$$            do k=1,6
c$$$               i = ijv(k,1)
c$$$               j = ijv(k,2)
c$$$               idsim(k) = iudot(i,j) * iudot(j,i)
c$$$            enddo
c$$$         endif

c$$$c     Normalize the velocity gradient by VonMises strain rate
c$$$         udot_vm = 0.d0
c$$$         do i=1,3
c$$$            udot_vm = udot_vm + udot(i,i)**2
c$$$         enddo
c$$$         udot_vm = dsqrt(2./3.) * dsqrt(udot_vm)
c$$$         do i=1,3
c$$$            udot(i,i) = udot(i,i) * c_vm / udot_vm
c$$$         enddo
c$$$c     strain-rate
c$$$         do 4 i=1,3
c$$$         do 4 j=1,3
c$$$            dsim(i,j) = (udot(i,j) + udot(j,i))/ 2.d0
c$$$ 4       continue
c$$$c     normalized-strain rate - concerning only
c$$$c     diagonal components - may lead to a significant
c$$$c     error when the stretching is off 'orthotropic'
c$$$c     axes of rolled sheets or if when material is not
c$$$c     orthotropic et al.
c$$$         dsim_vm = 0.d0
c$$$         do i=1,3
c$$$            dsim_vm = dsim_vm +  dsim(i,i)**2
c$$$         enddo
c$$$         dsim_vm = dsqrt(2./3.) * dsqrt(dsim_vm)
c$$$         do k=1,6
c$$$            i = ijv(k,1)
c$$$            j = ijv(k,2)
c$$$            idsim(k) = iudot(i,j) * iudot(j,i)
c$$$         enddo
c$$$         call chg_basis(dbar,dsim,aux55,aux3333,2,5)
c$$$         call chg_basis(sav,scauchy,aux55,aux3333,2,5)
c$$$         DBARNORM=VNORM(DBAR,5)
c$$$         SBARNORM=VNORM(SBAR,5)
c$$$         strain_control = 1
c$$$         eijincr = deij
c$$$         ictrl = 1 !! Uniaxial along Axis-1
c$$$         dsimctrl = dsim(ijv(ictrl,1), ijv(ictrl,2))
c$$$         if (dsimctrl.eq.0) then
c$$$            write(*,*) 'Wrong'
c$$$            stop
c$$$         endif
c$$$         tincr    = eijincr/dabs(dsimctrl)
c     ** iopt0 : L^B = L^A + vn
      elseif (iopt.eq.0) then
c     ** Define the vel. grad. in region B based on udot^A
c        Assumes that 1) udot of region A; and 2) tincr
c        are given before fld_bc was called
         call nt_psi(psi,n,t)
         do 5 i=1,2
         do 5 j=1,2
            udot(i,j) = udot(i,j) + vi(i) * n(j)
 5       continue
         udot(3,3) = - udot(1,1) - udot(2,2)
         do 6 i=1,3
         do 6 j=1,3
            dsim(i,j) = (udot(i,j) + udot(j,i))/ 2.d0
 6       continue
         call chg_basis(dbar,dsim,aux55,aux3333,2,5)
         call chg_basis(sav,scauchy,aux55,aux3333,2,5)
         strain_control = 1
c         ictrl = 1              ! hard-wired major axes
         if (dabs(udot(1,1)).ge.dabs(udot(2,2))) then
            ictrl = 1
         elseif (dabs(udot(1,1)).lt.dabs(udot(2,2))) then
            ictrl = 2
         else
            write(*,*) 'Unexpected udot reported in rho_transform'
            write(*,'(3(e14.8,1x))') ((udot(i,j),j=1,3),i=1,3)
            stop
         endif
         dsimctrl = dsim(ijv(ictrl,1), ijv(ictrl,2))
         eijincr = tincr * abs(dsimctrl)
c     ** iopt1/2/3 : L^A by rho
      elseif (iopt.eq.1.or.iopt.eq.2.or.iopt.eq.3) then
         if (iopt.eq.1.or.iopt.eq.2) call fld_rho_transform(
     $        rho,ictrl,udot)
         if (iopt.eq.3) then
            theta = rho
            call th2lij0(theta,udot)
         endif
c        Normalize the udot such that the strain rate
c        remains constant indepently of strain path rho
c        ** Note that only 'diagonal' components are used - need improvement
         udot_vm = 0.d0
         do i=1,3
            udot_vm = udot_vm + udot(i,i)**2
         enddo
         udot_vm = dsqrt(2./3.) * dsqrt(udot_vm)
         do i=1,3
            udot(i,i) = udot(i,i) * c_vm / udot_vm
         enddo
         ictrl=1
         if (abs(udot(2,2)).ge.abs(udot(1,1))) ictrl = 2
c     ** Define strain rate based on the velocity gradient
         do 10 i=1,3
         do 10 j=1,3
            dsim(i,j) = (udot(i,j) + udot(j,i))/ 2.d0
 10      continue
         dsim_vm = 0.d0
         do i=1,3
            dsim_vm = dsim_vm +  dsim(i,i)**2
         enddo
         dsim_vm = dsqrt(2./3.) * dsqrt(dsim_vm)
         call chg_basis(dbar,dsim,aux55,aux3333,2,5)
         call chg_basis(sav,scauchy,aux55,aux3333,2,5)
         strain_control = 1
c         eijincr = 1e-2         ! should determine for VPSC
         eijincr = deij
         dsimctrl = dsim(ijv(ictrl,1), ijv(ictrl,2))
         if (dsimctrl.eq.0) then
            write(*,*) 'Wrong'
            stop
         endif
         tincr    = eijincr/dabs(dsimctrl)

c        Relax L12 in order to allow the in-plane
c        rigid body rotation
         if (iopt.eq.2.or.iopt.eq.3.or.iopt.eq.1) then
            iudot(2,1) = 1
            iudot(1,2) = 0
            iscau(6)   = 1
            do k=1,6
               i = ijv(k,1)
               j = ijv(k,2)
               idsim(k) = iudot(i,j) * iudot(j,i)
            enddo
         endif
      endif
c     ** Check the boundary condition
      do 25 i=1,2
      do 25 j=i+1,3
         if (iudot(i,j) + iudot(j,i).eq.0) then
            write(*,*) 'Cannot relax both off-diagonal',
     $           ' components of the velocity gradients'
         endif
 25   continue
      do 30 k=1,6
         i = ijv(k,1)
         j = ijv(k,2)
         if (iscau(k)*idsim(k).ne.0
     $        .or. iscau(k)+idsim(k).ne.1)
     $        then
            write(*,*)'check boundary cond. on strain-rate',
     $           ' and stress'
            write(*,'('' idsim = '',6i3)') idsim
            write(*,'('' iscau = '',6i3)') iscau
            stop
         endif
 30   continue
      return
      end subroutine fld_bc
c------------------------------------------------------------c
c     Define Lij0 based on Von Mises strain and
c     the associated flow rule
      subroutine lij0_vm(alpha,l0)
      real*8 alpha, L0(3,3)
      L0(:,:) = 0.
      L0(1,1) = 2./3.            - 1./3. * alpha
      L0(2,2) = 2./3.*alpha      - 1./3.
      L0(3,3) =-1./3.*(1.+alpha)
      return
      end subroutine lij0_vm
c------------------------------------------------------------c
c     Guess, rho for the given alpha using VM yield function
c     as the potential.
      subroutine alpha2rho(alpha,rho)
      real*8 alpha,rho,L0(3,3)
      call lij0_vm(alpha,l0)
      rho = L0(2,2) / L0(1,1)
      return
      end subroutine alpha2rho
c------------------------------------------------------------c
      subroutine theta2omega(theta,omega)
      real*8 theta,omega,l0(3,3)
      call th2lij0(theta,l0)
      omega = atan2(l0(2,2),l0(1,1))
      return
      end subroutine theta2omega
c------------------------------------------------------------c
c     Given Theta, determine L0
      subroutine th2lij0(th,L0)
      implicit none
      real*8 th,L0(3,3),cth,sth
      L0(:,:)=0.d0
      cth = dcos(th)
      sth = dsin(th)
      L0(1,1)= cth
      L0(2,2)= sth
      L0(3,3)= -cth-sth !! vol. cnsrv
      return
      end subroutine th2lij0
c------------------------------------------------------------c
c     Given alpha, calculate the objective f_obj
      subroutine fld_alpha_f_obj(f_obj,alpha,rho,cvm,deij,istp)
      include 'vpsc7.dim'
      real*8 f_obj,alpha,rho,aux2(2),cvm,deij,
     $     sp(3,3),aux55(5,5),aux3333(3,3,3,3)
      integer istp,i
c     Determine the L bar (velocity gradient)
      aux2(:) = 0.              !! vi(:) = 0
      call fld_bc(rho,cvm,aux2,0.,2,istp,deij)
      call onestep(istp, .false.)
      call chg_basis(sav,sp,aux55,aux3333,1,5)
      do i=1,3
         sp(i,i) = sp(i,i) - sp(3,3)
      enddo
      f_obj = sp(2,2) - alpha * sp(1,1)
      return
      end subroutine fld_alpha_f_obj
c------------------------------------------------------------c
c     Given theta (strain ratio), calculate the objective f_obj
      subroutine fld_theta_f_obj(f_obj,theta,omega,
     $     cvm,deij,istp)
      include 'vpsc7.dim'
      real*8 f_obj,theta,omega,aux2(2),cvm,deij,
     $     sp(3,3),aux55(5,5),aux3333(3,3,3,3)
      integer istp,i
c     Determine the L bar (velocity gradient)
      aux2(:) = 0.
      call fld_bc(theta,cvm,aux2,0.,3,istp,deij)
      call onestep(istp, .false.)
      call chg_basis(sav,sp,aux55,aux3333,1,5)
      do i=1,3
         sp(i,i) = sp(i,i) - sp(3,3)
      enddo

c$$$      write(*,'(a)')'plane stress:'
c$$$      write(*,'(3(e14.8,1x))') ((sp(i,j),j=1,3),i=1,3)
c$$$      write(*,'(a,f9.2)')'omega:', omega*180./3.141592
c$$$      write(*,'(a,f9.2)')'theta:', theta*180./3.141592
c$$$      write(*,'(a,2(f9.2,3x))') 'om and om',
c$$$     $     atan2(sp(2,2),sp(1,1)), omega

      f_obj = dabs(atan2(sp(2,2),sp(1,1))-omega)
      return
      end subroutine fld_theta_f_obj
c------------------------------------------------------------c
      subroutine snapshot(iopt,fn)
c     iopt1 : read
c     iopt2 : save
      character*72 fn
      integer i,iopt
      include 'vpsc7.dim'

      if (iopt.eq.1) i = ur4    ! read
      if (iopt.eq.2) i = uw2    ! save
c      print *,trim(fn)
      open(i,file=trim(fn),form='unformatted',
     $     access='sequential')

      call postmortem(iopt)
      close(i)

      return
      end subroutine snapshot
c------------------------------------------------------------c
c     Return n and t vectors determined by psi
      subroutine nt_psi(psi,n,t)
      real*8:: psi,n(3),t(3)

c     Following the Figure 1 of the manual
      n(1) = dcos(psi)
      n(2) = dsin(psi)
      t(1) =-dsin(psi)
      t(2) = dcos(psi)
      n(3) = 0.d0
      t(3) = 0.d0
      return
      end subroutine nt_psi
c------------------------------------------------------------c
c     return band rotation matrix
      subroutine band_r(psi,r)
c     Rotation matrix r rotates (xy) coordinates in the original
c     axes counter-clock wise through an angle of psi.

      real*8:: psi, r(3,3)
      r(:,:) = 0.d0
      r(1,1) =  dcos(psi)
      r(1,2) = -dsin(psi)
      r(2,1) =  dsin(psi)
      r(2,2) =  dcos(psi)
      r(3,3) = 1.
      end subroutine band_r
c------------------------------------------------------------c

c     In the code, range of the given rho defines the strain
c     paths spanning the uniaxial tensile to balanced biaxial.
c      (-0.5 to 1.0) 1-axis being major tensile direction
c      (1.0  to 2.0) 2-axis being major tensile direction
c     Note that (1.0 to 2.0) is transformed to (-0.5 to 1.0)
c     in which 2-axis is being the major tensile direction

c     With transformed rho_t value to match.
c     In the range (1.0 to +),
c      1)  rho = 1   corresponds to rho_t =  1.  (Biaxial)
c      2)  rho = 2.0 corresponds to rho_t =  0.  (plane strain)
c      3)  rho = 2.5 corresponds to rho_t = -0.5 (uniaxial-like)
c      4)  rho = 2.6 corresponds to rho_t = -0.6 (uniaxial-like)

c     Thus rho_t is linearly transformed rho
      subroutine fld_rho_transform(rho0,ictrl,udot)
      real*8 rho_t,udot(3,3),rho0
      integer ictrl
c$$$      write(*,'(a,f5.2)') 'initial rho0:',    rho0
c$$$      rho_t = -1 * (rho0 - 1.) + 1.
c$$$      rho = rho0
      call fld_rho_t(rho0,rho_t)
      if (rho0.le.1.) then
!     conventional udot is for major axes 1
         udot(:,:) = 0.d0
         udot(1,1) = 1.d0
         udot(2,2) = rho0
         udot(3,3) = -udot(1,1)-udot(2,2)
      elseif (rho0.gt.1.) then
c     The above is now valid for (- to 1.0)
c     In the range (1.0 to +) the above is transformed to
         udot(:,:) = 0.d0
         udot(1,1) = rho_t
         udot(2,2) = 1.d0
         udot(3,3) = -udot(1,1)-udot(2,2)
      endif
c      write(*,'(a,f5.2)') 'Transformed rho_t:', rho_t
c      write(*,'(a)'     ) 'udots:'
c      write(*,'(3(f5.2,1x))') ((udot(i,j),j=1,3),i=1,3)

      if (dabs(udot(1,1)).ge.dabs(udot(2,2))) then
         ictrl = 1
      elseif (dabs(udot(1,1)).lt.dabs(udot(2,2))) then
         ictrl = 2
      else
         write(*,*) 'Unexpected udot reported in fld_rho_transform'
      endif
      return
      end subroutine fld_rho_transform
c------------------------------------------------------------
      subroutine fld_rho_t(rho0,rho_t)
      implicit none
      real*8 rho0,rho_t
      if (rho0.le.1) then       ! Major axis: RD
         rho_t = rho0
      elseif(rho0.gt.1) then    ! Major axis: TD
         rho_t = -1 * (rho0 - 1.) + 1.
      endif
      return
      end subroutine fld_rho_t
c------------------------------------------------------------c
c     Limit strain for region A simulation
c     This function is closely related with computational
c     performance (time-wise) for VPSC-FLD.
      logical function is_limit(c1,epstot)
      implicit none
      real*8 epstot(3,3),c1,calc_evm

c     c1 is 'EVM' limit (not a great idea, but let's use it for the moment)
      is_limit=.false.
c     compare to Von Mises strain
      if (calc_evm(epstot).ge.c1) is_limit=.true.
      end function is_limit

c------------------------------------------------------------c
c     calculation Von Mises strain (not based on work-equivalence)
c     EVM= sqrt(3/2 Eij*2)
      real*8 function calc_evm(eps)
      implicit none
      real*8 eps(3,3),v,tiny
      integer i,j
      tiny = 1e-10
c     Is eps deviatoric? meaning that is   delt_ii e_ii = 0 ??
      v=0.d0
      do i=1,3
         v = v + eps(i,i)
      enddo
      if (dabs(v).gt.tiny) then
         write(*,*) 'Passed strain might not be plastic strain???'
         stop
      endif
c     if survived,
      calc_evm = 0.d0
      do 10 i=1,3
      do 10 j=1,3
         calc_evm = calc_evm + eps(i,j)**2
 10   continue
      calc_evm = dsqrt(calc_evm)
      return
      end function calc_evm

c----------------------------------------------------------------------c
c     Transform a matrix by rotation mat rot and returns the result as b
c     b = rot  a  rot^T
c     b_ij = rot_ik a_kl rot_jl

c     rot{b_axes<-a_axes}
      subroutine matrot(a,rot,b)
      implicit none
      real*8 a(3,3), rot(3,3), b(3,3)
      integer i,j,k,l
cf2py intent(in) a, rot
cf2py intent(out) b
      do 100 i=1,3
      do 100 j=1,3
         b(i,j) = 0.d0
      do 100 k=1,3
      do 100 l=1,3
         b(i,j) = b(i,j) + rot(i,k) * a(k,l) * rot(j,l)
 100  continue
      return
      end subroutine matrot
c----------------------------------------------------------------------c
      subroutine matranspose(a,b)
      implicit none
      real*8 a(3,3), b(3,3)
      integer i,j
      do 10 i=1,3
      do 10 j=1,3
         b(j,i) = a(i,j)
 10   continue
      return
      end subroutine matranspose
c----------------------------------------------------------------------c
      subroutine mat5rot(a,rot,b)
      implicit none
      real*8 a(5),rot(3,3),b(5),aux33(3,3),
     $     bux33(3,3),aux3333(3,3,3,3),aux55(5,5)
cf2py intent(in) a, rot
cf2py intent(out) b
      call chg_basis(a,aux33,aux55,aux3333,1,5)
      call matrot(aux33,rot,bux33)
      call chg_basis(b,bux33,aux55,aux3333,2,5)
      return
      end subroutine mat5rot
c----------------------------------------------------------------------c
c     sa is referred in 1-2 axes
c     sb is referred in n-t axes
      subroutine check_force_eq2(sa,sb,psi,f)
      real*8 sa(5),sb(5),psi,n(3),t(3),br(3,3)
      real*8 sigma_a(3,3),sigma_b(3,3)
      real*8 sigma_a_nt(3,3),sigma_b_nt(3,3)
      real*8 aux55(5,5),aux3333(3,3,3,3),tiny
      integer i,j
c     It assumes that both sa and sb are referred
c     in the same reference axes.

      call chg_basis(sa,sigma_a,aux55,aux3333,1,5)
      do i=1,3
         sigma_a(i,i) = sigma_a(i,i)  - sigma_a(3,3)
      enddo
      call chg_basis(sb,sigma_b,aux55,aux3333,1,5)
      do i=1,3
         sigma_b(i,i) = sigma_b(i,i)  - sigma_b(3,3)
      enddo

      call band_r(psi,br)
      call nt_psi(psi,n,t)
      call matrot(sigma_a,br,sigma_a_nt)
c      call matrot(sigma_b,br,sigma_b_nt)
      sigma_b_nt(:,:) = sigma_b(:,:)

      write(*,'(a,3x,f6.3)') 'F:', f
      write(*,'(4(a24,5x))') 'S^A','S^B','f*S^B','S^A - f*S^B'
      do i=1,3
         write(*,'(4(3(f6.2,2x),5x))') (sigma_a_nt(i,j),j=1,3),
     $        (sigma_b(i,j),j=1,3), (sigma_b(i,j)*f,j=1,3),
     $        (sigma_a_nt(i,j)-sigma_b(i,j)*f,j=1,3)
      enddo

      tiny = -1.
      if (abs(sigma_a(1,1)*1./f - sigma_b(1,1)) .gt. tiny)then
         write(*,'(f5.1)',advance='no') abs(sigma_a(1,1)*1./f
     $        - sigma_b(1,1))
c         write(*,*) 'Force equilibriume violated. 1'
c         stop -1
      endif
      if (abs(sigma_a(1,2)*1./f - sigma_b(1,2)) .gt. tiny)then
         write(*,'(f5.1)',advance='no') abs(sigma_a(1,2)*1./f
     $        - sigma_b(1,2))
c         write(*,*) 'Force equilibriume violated. 2'
c         stop -1
      endif
      return
      end subroutine check_force_eq2
c----------------------------------------------------------------------
      subroutine check_force_eq(sa,sb,psi,f)
      real*8 sa(5),sb(5),psi,n(3),t(3),br(3,3)
      real*8 sigma_a(3,3),sigma_b(3,3)
      real*8 sigma_a_nt(3,3),sigma_b_nt(3,3)
      real*8 aux55(5,5),aux3333(3,3,3,3),tiny
      integer i,j
c     It assumes that both sa and sb are referred
c     in the same reference axes.

      call chg_basis(sa,sigma_a,aux55,aux3333,1,5)
      do i=1,3
         sigma_a(i,i) = sigma_a(i,i)  - sigma_a(3,3)
      enddo
      call chg_basis(sb,sigma_b,aux55,aux3333,1,5)
      do i=1,3
         sigma_b(i,i) = sigma_b(i,i)  - sigma_b(3,3)
      enddo

      call band_r(psi,br)
      call nt_psi(psi,n,t)
      call matrot(sigma_a,br,sigma_a_nt)
      call matrot(sigma_b,br,sigma_b_nt)

      write(*,'(a,3x,f6.3)') 'F:', f
      write(*,'(4(a24,5x))') 'S^A','S^B','f*S^B','S^A - f*S^B'
      do i=1,3
         write(*,'(4(3(f6.2,2x),5x))') (sigma_a(i,j),j=1,3),
     $        (sigma_b(i,j),j=1,3), (sigma_b(i,j)*f,j=1,3),
     $        (sigma_a(i,j)-sigma_b(i,j)*f,j=1,3)

      enddo

      tiny = -1.
      if (abs(sigma_a(1,1)*1./f - sigma_b(1,1)) .gt. tiny)then
         write(*,'(f5.1)',advance='no') abs(sigma_a(1,1)*1./f
     $        - sigma_b(1,1))
c         write(*,*) 'Force equilibriume violated. 1'
c         stop -1
      endif
      if (abs(sigma_a(1,2)*1./f - sigma_b(1,2)) .gt. tiny)then
         write(*,'(f5.1)',advance='no') abs(sigma_a(1,2)*1./f
     $        - sigma_b(1,2))
c         write(*,*) 'Force equilibriume violated. 2'
c         stop -1
      endif
c$$$      if (abs(sigma_a(2,1)*1./f - sigma_b(2,1)) .gt. tiny)then
c$$$         write(*,'(f5.1)') abs(sigma_a(2,1)*1./f - sigma_b(2,1))
c$$$c         write(*,*) 'Force equilibriume violated. 3'
c$$$c         stop - 1
c$$$      endif

      return
      end subroutine check_force_eq
c----------------------------------------------------------------------c
      subroutine check_compat(ea,eb,psi,iopt)
      real*8 ea(3,3),eb(3,3),psi,n(3),t(3),ett_a,ent_a,
     $     ett_b,ent_b,tiny,br(3,3),ea_nt(3,3),eb_nt(3,3)
      integer iopt
      call nt_psi(psi,n,t)
      call band_r(psi,br)

      call matrot(ea,br,ea_nt)
      call matrot(eb,br,eb_nt)

c      if (iopt.eq.0) write(*,'(a,f8.5)') 'E_TT diff:',
c     $     abs(ea_nt(2,2)-eb_nt(2,2))

      if (iopt.eq.1) then

         tiny =1e-5
         if (abs(ett_a-ett_b).gt.tiny .or.
     $        abs(ent_a-ent_b).gt.tiny) then
            write(*,'(a,f8.5)') 'udot_TT diff:',
     $           abs(ea_nt(2,2)-eb_nt(2,2))
            write(*,*) 'Compatibility is violated.'
            stop -1
         endif
      endif

      return
      end subroutine check_compat
c----------------------------------------------------------------------c
c$$$      subroutine check_compat1(ea,eb,psi)
c$$$      real*8 ea(3,3),eb(3,3),psi,n(3),t(3),ett_a,ent_a,
c$$$     $     ett_b,ent_b,tiny
c$$$      integer i,j
c$$$      call nt_psi(psi,n,t)
c$$$      ett_a=0.
c$$$      ent_a=0.
c$$$      ett_b=0.
c$$$      ent_b=0.
c$$$      do 10 i=1,3
c$$$      do 10 j=1,3
c$$$         ett_a=ett_a + t(i) * ea(i,j) * t(j)
c$$$         ent_a=ent_a + n(i) * ea(i,j) * t(j)
c$$$         ett_b=ett_b + t(i) * ea(i,j) * t(j)
c$$$         ent_b=ent_b + n(i) * ea(i,j) * t(j)
c$$$ 10   continue
c$$$      tiny =1e-5
c$$$      if (abs(ett_a-ett_b).gt.tiny .or.
c$$$     $     abs(ent_a-ent_b).gt.tiny) then
c$$$         write(*,*) 'Compatibility is violated.'
c$$$         stop -1
c$$$      endif
c$$$      end subroutine check_compat1
