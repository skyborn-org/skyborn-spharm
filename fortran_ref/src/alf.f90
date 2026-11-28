!
! Optimized version of alf.f
! Associated Legendre Function computations
! Optimizations: Modern Fortran syntax, vectorization, OpenMP parallelization
!
! Original SPHEREPACK documentation maintained for compatibility
!
!
!  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
!  .                                                             .
!  .                  copyright (c) 1998 by UCAR                 .
!  .                                                             .
!  .       University Corporation for Atmospheric Research       .
!  .                                                             .
!  .                      all rights reserved                    .
!  .                                                             .
!  .                                                             .
!  .                         SPHEREPACK                          .
!  .                                                             .
!  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
!

! Optimized alfk subroutine for computing normalized associated Legendre polynomial coefficients
! [FIXED] Upgraded internal recurrence and 2**(n-1-nex) scaling to IEEE real64.
! Historical SPHEREPACK single-precision bug: when n - 1 - nex >= 128 (starting around n >= 149),
! single-precision `2.0 ** (n - 1 - nex)` overflowed IEEE-754 real32 (> 2^128) to +Infinity, causing
! `t1 / Inf = 0.0` and silently zeroing out high-degree Legendre coefficients (e.g. m=n=319 -> 0.0).
subroutine alfk(n, m, cp)
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none

    ! Input/output parameters
    integer, intent(in) :: n, m
    real, intent(out) :: cp(n/2 + 1)

    ! Local variables
    integer :: ma, nmms2, i, l, nex, iexp
    real(real64) :: fnum, fnmh, pm1, t1, t2, fden
    real(real64) :: cp2, fnnp1, fnmsq, fk, a1, b1, c1
    real(real64), allocatable :: cp64(:)
    real(real64), parameter :: sc10 = 1024.0_real64
    real(real64), parameter :: sc20 = sc10 * sc10
    real(real64), parameter :: sc40 = sc20 * sc20

    ! Initialize
    cp(1) = 0.0
    ma = abs(m)
    if (ma > n) return

    ! Handle special cases
    select case (n)
    case (0)
        cp(1) = sqrt(2.0)
        return
    case (1)
        if (ma == 0) then
            cp(1) = sqrt(1.5)
        else
            cp(1) = sqrt(0.75)
            if (m == -1) cp(1) = -cp(1)
        end if
        return
    end select

    ! Main computation for n >= 2
    if (mod(n + ma, 2) == 0) then
        nmms2 = (n - ma) / 2
        fnum = real(n + ma + 1, real64)
        fnmh = real(n - ma + 1, real64)
        pm1 = 1.0_real64
    else
        nmms2 = (n - ma - 1) / 2
        fnum = real(n + ma + 2, real64)
        fnmh = real(n - ma + 2, real64)
        pm1 = -1.0_real64
    end if

    ! Scale computation with overflow protection
    t1 = 1.0_real64 / sc20
    nex = 20
    fden = 2.0_real64

    if (nmms2 >= 1) then
        ! t1/nex are loop-carried scaling recurrences: keep sequential.
        do i = 1, nmms2
            t1 = fnum * t1 / fden
            if (t1 > sc20) then
                t1 = t1 / sc40
                nex = nex + 40
            end if
            fnum = fnum + 2.0_real64
            fden = fden + 2.0_real64
        end do
    end if

    iexp = nex + 1 - n
    do while (iexp > 512)
        t1 = t1 * (2.0_real64 ** 512)
        iexp = iexp - 512
    end do
    do while (iexp < -512)
        t1 = t1 * (2.0_real64 ** (-512))
        iexp = iexp + 512
    end do
    t1 = t1 * (2.0_real64 ** iexp)
    if (mod(ma/2, 2) /= 0) t1 = -t1

    t2 = 1.0_real64
    if (ma > 0) then
        ! t2/fnmh are loop-carried recurrences: keep sequential.
        do i = 1, ma
            t2 = fnmh * t2 / (fnmh + pm1)
            fnmh = fnmh + 2.0_real64
        end do
    end if

    allocate(cp64(n/2 + 1))
    cp64(1) = 0.0_real64
    cp2 = t1 * sqrt((real(n, real64) + 0.5_real64) * t2)
    fnnp1 = real(n, real64) * real(n + 1, real64)
    fnmsq = fnnp1 - 2.0_real64 * real(ma, real64) * real(ma, real64)
    l = (n + 1) / 2
    if (mod(n, 2) == 0 .and. mod(ma, 2) == 0) l = l + 1
    cp64(l) = cp2

    if (m < 0 .and. mod(ma, 2) /= 0) cp64(l) = -cp64(l)

    if (l > 1) then
        ! Backward recurrence in real64 before narrowing to output
        fk = real(n, real64)
        a1 = (fk - 2.0_real64) * (fk - 1.0_real64) - fnnp1
        b1 = 2.0_real64 * (fk * fk - fnmsq)
        cp64(l - 1) = b1 * cp64(l) / a1

        do while (l > 2)
            l = l - 1
            fk = fk - 2.0_real64
            a1 = (fk - 2.0_real64) * (fk - 1.0_real64) - fnnp1
            b1 = -2.0_real64 * (fk * fk - fnmsq)
            c1 = (fk + 1.0_real64) * (fk + 2.0_real64) - fnnp1
            cp64(l - 1) = -(b1 * cp64(l) + c1 * cp64(l + 1)) / a1
        end do
    end if

    cp(1:(n/2 + 1)) = real(cp64(1:(n/2 + 1)))
    deallocate(cp64)

end subroutine alfk

! Optimized lfim subroutine for vectorized computation of associated Legendre functions
subroutine lfim(init, theta, l, n, nm, pb, id, wlfim)
    implicit none

    ! Input/output parameters
    integer, intent(in) :: init, l, n, nm, id
    real, intent(in) :: theta(l)
    real, intent(inout) :: pb(id, *), wlfim(*)

    ! Local variables
    integer :: lnx, iw1, iw2, iw3

    ! Calculate workspace indices
    lnx = l * (nm + 1)
    iw1 = lnx + 1
    iw2 = iw1 + lnx
    iw3 = iw2 + lnx

    call lfim1(init, theta, l, n, nm, id, pb, wlfim, wlfim(iw1), &
              wlfim(iw2), wlfim(iw3), wlfim(iw2))

end subroutine lfim

! Optimized lfim1 worker subroutine with OpenMP parallelization
subroutine lfim1(init, theta, l, n, nm, id, p3, phz, ph1, p1, p2, cp)
    implicit none

    ! Input/output parameters
    integer, intent(in) :: init, l, n, nm, id
    real, intent(in) :: theta(l)
    real, intent(inout) :: p1(l, *), p2(l, *), p3(id, *)
    real, intent(inout) :: phz(l, *), ph1(l, *), cp(*)

    ! Local variables
    integer :: nmp1, nh, np1, i, nm1, mp1, m
    real :: ssqrt2, fn, tn, cn, fm, fnpm, fnmm, temp, cc, dd, ee
    real :: sq5s6, sq1s6

    nmp1 = nm + 1

    if (init == 0) then
        ! Initialization phase
        ssqrt2 = 1.0 / sqrt(2.0)

        !$OMP SIMD
        do i = 1, l
            phz(i, 1) = ssqrt2
        end do

        do np1 = 2, nmp1
            nh = np1 - 1
            call alfk(nh, 0, cp)

            !$OMP SIMD
            do i = 1, l
                call lfpt(nh, 0, theta(i), cp, phz(i, np1))
            end do

            call alfk(nh, 1, cp)

            !$OMP SIMD
            do i = 1, l
                call lfpt(nh, 1, theta(i), cp, ph1(i, np1))
            end do
        end do
        return
    end if

    ! Computation phase
    select case (n)
    case (0)
        !$OMP SIMD
        do i = 1, l
            p3(i, 1) = phz(i, 1)
        end do
        return

    case (1)
        !$OMP SIMD
        do i = 1, l
            p3(i, 1) = phz(i, 2)
            p3(i, 2) = ph1(i, 2)
        end do
        return

    case (2)
        sq5s6 = sqrt(5.0/6.0)
        sq1s6 = sqrt(1.0/6.0)

        !$OMP SIMD
        do i = 1, l
            p3(i, 1) = phz(i, 3)
            p3(i, 2) = ph1(i, 3)
            p3(i, 3) = sq5s6 * phz(i, 1) - sq1s6 * p3(i, 1)
            p1(i, 1) = phz(i, 2)
            p1(i, 2) = ph1(i, 2)
            p2(i, 1) = phz(i, 3)
            p2(i, 2) = ph1(i, 3)
            p2(i, 3) = p3(i, 3)
        end do
        return
    end select

    ! General case for n > 2
    nm1 = n - 1
    np1 = n + 1
    fn = real(n)
    tn = fn + fn
    cn = (tn + 1.0) / (tn - 3.0)

    !$OMP SIMD
    do i = 1, l
        p3(i, 1) = phz(i, np1)
        p3(i, 2) = ph1(i, np1)
    end do

    if (nm1 >= 3) then
        do mp1 = 3, nm1
            m = mp1 - 1
            fm = real(m)
            fnpm = fn + fm
            fnmm = fn - fm
            temp = fnpm * (fnpm - 1.0)
            cc = sqrt(cn * (fnpm - 3.0) * (fnpm - 2.0) / temp)
            dd = sqrt(cn * fnmm * (fnmm - 1.0) / temp)
            ee = sqrt((fnmm + 1.0) * (fnmm + 2.0) / temp)

            !$OMP SIMD
            do i = 1, l
                p3(i, mp1) = cc * p1(i, mp1 - 2) + dd * p1(i, mp1) - ee * p3(i, mp1 - 2)
            end do
        end do
    end if

    ! Handle remaining terms
    fnpm = fn + fn - 1.0
    temp = fnpm * (fnpm - 1.0)
    cc = sqrt(cn * (fnpm - 3.0) * (fnpm - 2.0) / temp)
    ee = sqrt(6.0 / temp)

    !$OMP SIMD
    do i = 1, l
        p3(i, n) = cc * p1(i, n - 2) - ee * p3(i, n - 2)
    end do

    fnpm = fn + fn
    temp = fnpm * (fnpm - 1.0)
    cc = sqrt(cn * (fnpm - 3.0) * (fnpm - 2.0) / temp)
    ee = sqrt(2.0 / temp)

    !$OMP SIMD
    do i = 1, l
        p3(i, n + 1) = cc * p1(i, n - 1) - ee * p3(i, n - 1)
    end do

    ! Update arrays for next iteration
    do mp1 = 1, np1
        !$OMP SIMD
        do i = 1, l
            p1(i, mp1) = p2(i, mp1)
            p2(i, mp1) = p3(i, mp1)
        end do
    end do

end subroutine lfim1

! Similar optimization for lfin subroutine
subroutine lfin(init, theta, l, m, nm, pb, id, wlfin)
    implicit none

    integer, intent(in) :: init, l, m, nm, id
    real, intent(in) :: theta(l)
    real, intent(inout) :: pb(id, *), wlfin(*)

    integer :: lnx, iw1, iw2, iw3

    lnx = l * (nm + 1)
    iw1 = lnx + 1
    iw2 = iw1 + lnx
    iw3 = iw2 + lnx

    call lfin1(init, theta, l, m, nm, id, pb, wlfin, wlfin(iw1), &
              wlfin(iw2), wlfin(iw3), wlfin(iw2))

end subroutine lfin

! Optimized lfin1 worker subroutine
subroutine lfin1(init, theta, l, m, nm, id, p3, phz, ph1, p1, p2, cp)
    implicit none

    integer, intent(in) :: init, l, m, nm, id
    real, intent(in) :: theta(l)
    real, intent(inout) :: p1(l, *), p2(l, *), p3(id, *)
    real, intent(inout) :: phz(l, *), ph1(l, *), cp(*)

    integer :: nmp1, nh, np1, i, mp1, mp3, n
    real :: ssqrt2, fm, tm, temp, cc, ee, dd, fn, tn, cn, fnpm, fnmm

    nmp1 = nm + 1

    if (init == 0) then
        ! Initialization phase
        ssqrt2 = 1.0 / sqrt(2.0)

        !$OMP SIMD
        do i = 1, l
            phz(i, 1) = ssqrt2
        end do

        do np1 = 2, nmp1
            nh = np1 - 1
            call alfk(nh, 0, cp)

            !$OMP SIMD
            do i = 1, l
                call lfpt(nh, 0, theta(i), cp, phz(i, np1))
            end do

            call alfk(nh, 1, cp)

            !$OMP SIMD
            do i = 1, l
                call lfpt(nh, 1, theta(i), cp, ph1(i, np1))
            end do
        end do
        return
    end if

    ! Computation phase
    mp1 = m + 1
    fm = real(m)
    tm = fm + fm

    select case (m)
    case (0)
        do np1 = 1, nmp1
            !$OMP SIMD
            do i = 1, l
                p3(i, np1) = phz(i, np1)
                p1(i, np1) = phz(i, np1)
            end do
        end do
        return

    case (1)
        do np1 = 2, nmp1
            !$OMP SIMD
            do i = 1, l
                p3(i, np1) = ph1(i, np1)
                p2(i, np1) = ph1(i, np1)
            end do
        end do
        return
    end select

    ! General case for m > 1
    temp = tm * (tm - 1.0)
    cc = sqrt((tm + 1.0) * (tm - 2.0) / temp)
    ee = sqrt(2.0 / temp)

    !$OMP SIMD
    do i = 1, l
        p3(i, m + 1) = cc * p1(i, m - 1) - ee * p1(i, m + 1)
    end do

    if (m == nm) return

    temp = tm * (tm + 1.0)
    cc = sqrt((tm + 3.0) * (tm - 2.0) / temp)
    ee = sqrt(6.0 / temp)

    !$OMP SIMD
    do i = 1, l
        p3(i, m + 2) = cc * p1(i, m) - ee * p1(i, m + 2)
    end do

    mp3 = m + 3
    if (nmp1 >= mp3) then
        do np1 = mp3, nmp1
            n = np1 - 1
            fn = real(n)
            tn = fn + fn
            cn = (tn + 1.0) / (tn - 3.0)
            fnpm = fn + fm
            fnmm = fn - fm
            temp = fnpm * (fnpm - 1.0)
            cc = sqrt(cn * (fnpm - 3.0) * (fnpm - 2.0) / temp)
            dd = sqrt(cn * fnmm * (fnmm - 1.0) / temp)
            ee = sqrt((fnmm + 1.0) * (fnmm + 2.0) / temp)

            !$OMP SIMD
            do i = 1, l
                p3(i, np1) = cc * p1(i, np1 - 2) + dd * p3(i, np1 - 2) - ee * p1(i, np1)
            end do
        end do
    end if

    ! Update arrays for next iteration
    do np1 = m, nmp1
        !$OMP SIMD
        do i = 1, l
            p1(i, np1) = p2(i, np1)
            p2(i, np1) = p3(i, np1)
        end do
    end do

end subroutine lfin1

! Optimized lfpt subroutine for efficient polynomial evaluation
subroutine lfpt(n, m, theta, cp, pb)
    implicit none

    ! Input/output parameters
    integer, intent(in) :: n, m
    real, intent(in) :: theta
    real, intent(in) :: cp(*)
    real, intent(out) :: pb

    ! Local variables
    integer :: ma, np1, nmod, mmod, kdo, k
    real :: cdt, sdt, ct, st, sum, cth

    pb = 0.0
    ma = abs(m)
    if (ma > n) return

    if (n <= 0) then
        if (ma <= 0) pb = sqrt(0.5)
        return
    end if

    np1 = n + 1
    nmod = mod(n, 2)
    mmod = mod(ma, 2)

    if (nmod == 0) then
        ! n even
        if (mmod == 0) then
            ! n even, m even
            kdo = n / 2 + 1
            cdt = cos(theta + theta)
            sdt = sin(theta + theta)
            ct = 1.0
            st = 0.0
            sum = 0.5 * cp(1)

            ! ct/st trigonometric recurrence and sum: keep sequential.
            do k = 2, kdo
                cth = cdt * ct - sdt * st
                st = sdt * ct + cdt * st
                ct = cth
                sum = sum + cp(k) * ct
            end do
            pb = sum
        else
            ! n even, m odd
            kdo = n / 2
            cdt = cos(theta + theta)
            sdt = sin(theta + theta)
            ct = 1.0
            st = 0.0
            sum = 0.0

            ! ct/st trigonometric recurrence and sum: keep sequential.
            do k = 1, kdo
                cth = cdt * ct - sdt * st
                st = sdt * ct + cdt * st
                ct = cth
                sum = sum + cp(k) * st
            end do
            pb = sum
        end if
    else
        ! n odd
        kdo = (n + 1) / 2
        if (mmod == 0) then
            ! n odd, m even
            cdt = cos(theta + theta)
            sdt = sin(theta + theta)
            ct = cos(theta)
            st = -sin(theta)
            sum = 0.0

            ! ct/st trigonometric recurrence and sum: keep sequential.
            do k = 1, kdo
                cth = cdt * ct - sdt * st
                st = sdt * ct + cdt * st
                ct = cth
                sum = sum + cp(k) * ct
            end do
            pb = sum
        else
            ! n odd, m odd
            cdt = cos(theta + theta)
            sdt = sin(theta + theta)
            ct = cos(theta)
            st = -sin(theta)
            sum = 0.0

            ! ct/st trigonometric recurrence and sum: keep sequential.
            do k = 1, kdo
                cth = cdt * ct - sdt * st
                st = sdt * ct + cdt * st
                ct = cth
                sum = sum + cp(k) * st
            end do
            pb = sum
        end if
    end if

end subroutine lfpt
