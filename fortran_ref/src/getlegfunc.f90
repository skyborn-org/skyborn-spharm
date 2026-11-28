!
! [FIXED] Upgraded from O(N^3) real32 alfk/lfpt calls to O(N^2) real64 Bonnet + Schwarztrauber recurrence.
! Historical bug in SPHEREPACK getlegfunc/alfk: calling single-precision alfk(n-1, m-1, cp) caused
! 2.0**(n-1-nex) overflow to +Inf for n >= 149, zeroing out high-degree sectoral modes (Addition Theorem
! failure, e.g. err ~ 169 at ntrunc=319) alongside O(N^3) runtime overhead.
!
subroutine getlegfunc(legfunc, lat, ntrunc)
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none

    ! Input/Output parameters
    integer, intent(in) :: ntrunc
    real, intent(in) :: lat
    real, intent(out) :: legfunc((ntrunc+1)*(ntrunc+2)/2)

    ! Local variables
    integer :: m, n, nm, stride, nm2_idx, nm1_idx, cur_idx, i_cur, i_nm1, i_nm2
    real(real64) :: theta, pi, cos_th, sin_th, ssqr2, nf, mf, c_num, c_den, d1
    real(real64) :: a0, b0, a1, b1, denom1, abel, bbel, cbel, val
    real(real64), allocatable :: ring(:)

    pi = acos(-1.0_real64)
    theta = 0.5_real64 * pi - (pi / 180.0_real64) * real(lat, real64)
    cos_th = cos(theta)
    sin_th = sin(theta)

    stride = ntrunc + 1
    allocate(ring(3 * stride))
    ring = 0.0_real64

    ! n = 0
    ssqr2 = sqrt(0.5_real64)
    ring(1) = ssqr2
    legfunc(1) = real(ssqr2)
    if (ntrunc == 0) then
        deallocate(ring)
        return
    end if

    ! n = 1
    i_cur = stride
    ring(i_cur + 1) = sqrt(1.5_real64) * cos_th
    ring(i_cur + 2) = sqrt(0.75_real64) * sin_th
    legfunc(2) = real(ring(i_cur + 1))
    legfunc(ntrunc + 2) = real(ring(i_cur + 2))

    do n = 2, ntrunc
        nm2_idx = mod(n + 1, 3) * stride
        nm1_idx = mod(n + 2, 3) * stride
        cur_idx = mod(n, 3) * stride

        nf = real(n, real64)
        c_num = 2.0_real64 * nf + 1.0_real64
        c_den = 2.0_real64 * nf - 3.0_real64

        ! m = 0 (3-term Bonnet recurrence)
        a0 = sqrt(c_num * (2.0_real64 * nf - 1.0_real64)) / nf
        b0 = ((nf - 1.0_real64) / nf) * sqrt(c_num / c_den)
        ring(cur_idx + 1) = a0 * cos_th * ring(nm1_idx + 1) - b0 * ring(nm2_idx + 1)
        legfunc(n + 1) = real(ring(cur_idx + 1))

        ! m = 1 (3-term Bonnet recurrence)
        denom1 = (nf - 1.0_real64) * (nf + 1.0_real64)
        a1 = sqrt(c_num * (2.0_real64 * nf - 1.0_real64) / denom1)
        if (n >= 3) then
            b1 = sqrt(c_num * nf * (nf - 2.0_real64) / (c_den * denom1))
        else
            b1 = 0.0_real64
        end if
        ring(cur_idx + 2) = a1 * cos_th * ring(nm1_idx + 2) - b1 * ring(nm2_idx + 2)
        legfunc(ntrunc + n + 1) = real(ring(cur_idx + 2))

        ! m = 2..n (Schwarztrauber recurrence)
        do m = 2, n
            mf = real(m, real64)
            d1 = (mf + nf - 1.0_real64) * (mf + nf)
            abel = sqrt(c_num * (mf + nf - 2.0_real64) * (mf + nf - 3.0_real64) / (c_den * d1))
            cbel = sqrt(((nf - mf + 1.0_real64) * (nf - mf + 2.0_real64)) / d1)
            i_nm2 = nm2_idx + m - 1
            i_cur = cur_idx + m - 1
            if (m < n - 1) then
                bbel = sqrt(c_num * (nf - mf - 1.0_real64) * (nf - mf) / (c_den * d1))
                val = abel * ring(i_nm2) + bbel * ring(nm2_idx + m + 1) - cbel * ring(i_cur)
            else
                val = abel * ring(i_nm2) - cbel * ring(i_cur)
            end if
            ring(cur_idx + m + 1) = val
            nm = m * (2 * ntrunc - m + 3) / 2 + (n - m) + 1
            legfunc(nm) = real(val)
        end do
    end do

    deallocate(ring)
end subroutine getlegfunc
