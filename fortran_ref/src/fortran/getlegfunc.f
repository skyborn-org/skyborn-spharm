c [KNOWN HISTORICAL BUG] Calls single-precision `alfk` inside O(N^2) loop,
c triggering both O(N^3) runtime and silent real32 underflow-to-zero for n >= 149
c (e.g. Addition Theorem error ~169 at ntrunc=319). Fixed in ../getlegfunc.f90
c and superseded by `rust/spharm-math/src/legendre.rs`.
      subroutine getlegfunc(legfunc,lat,ntrunc)
      integer ntrunc,m,n,nmstrt
      real lat,theta,pi,
     *     legfunc((ntrunc+1)*(ntrunc+2)/2),cp((ntrunc/2)+1)

      pi = 4.*atan(1.0)
      theta = 0.5*pi-(pi/180.)*lat
      nmstrt = 0
      do m=1,ntrunc+1
      do n=m,ntrunc+1
         nm = nmstrt + n - m + 1
         call alfk(n-1,m-1,cp)
         call lfpt(n-1,m-1,theta,cp,legfunc(nm))
      enddo
      nmstrt = nmstrt + ntrunc - m + 2
      enddo

      return
      end
