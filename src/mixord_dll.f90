!  -*- Fortran -*-
!  Time-stamp: "12:31 PM 12/14/2012 415CU7 kkim32"
! ----------------------------------------------------------------------------
!
!  MIXOR_dll - The estimator for mixed-effects ordinal regression analysis
!
! ----------------------------------------------------------------------------
!
!  Contains:
!
!     MODULE INTERFACE_FP_EQUAL
!     MODULE INTERFACE_STARTV2
!     MODULE INTERFACE_unique
!     MODULE INTERFACE_CHAMS
!     MODULE INTERFACE_CHSKY
!     MODULE INTERFACE_GEN
!     MODULE INTERFACE_GRMCV
!     MODULE INTERFACE_MPDSD
!     MODULE INTERFACE_HRECUR
!     MODULE INTERFACE_HERMIT
!     MODULE INTERFACE_QUADP
!     MODULE INTERFACE_HROOT
!     MODULE INTERFACE_KMPY
!     MODULE INTERFACE_MPYM
!     MODULE INTERFACE_YSAME
!     MODULE INTERFACE_ADJR
!     MODULE INTERFACE_ADJC
!     MODULE INTERFACE_INVS
!     MODULE INTERFACE_INVT
!     MODULE INTERFACE_GRAMT
!     MODULE INTERFACE_SCM
!
!     MODULE INTERFACE_STARTVAL
!     MODULE INTERFACE_Empirical_Prior
!     MODULE INTERFACE_MIXORDEST
!     MODULE INTERFACE_mainloop
!
!     SUBROUTINE mainloop
!
!     SUBROUTINE IDNOBS
!     SUBROUTINE STARTVAL
!     SUBROUTINE MIXORDEST
!     SUBROUTINE FREE_LOCALS
!     SUBROUTINE Empirical_Prior
!
!     SUBROUTINE unique
!     SUBROUTINE CHAMS
!     SUBROUTINE CHSKY
!     SUBROUTINE GEN
!     SUBROUTINE GRMCV
!     SUBROUTINE GRMMT
!     SUBROUTINE ADDM
!     SUBROUTINE SUBM
!     SUBROUTINE SCM
!     SUBROUTINE MPDSD
!     SUBROUTINE MPYTR
!     SUBROUTINE GRAMM
!     SUBROUTINE HRECUR
!     SUBROUTINE HERMIT
!     SUBROUTINE HROOT
!     SUBROUTINE KMPY
!     SUBROUTINE MPYM
!     SUBROUTINE RELOC
!     SUBROUTINE YSAME
!     SUBROUTINE GEND
!     SUBROUTINE ADJR
!     SUBROUTINE ADJC
!     SUBROUTINE INVS
!     SUBROUTINE INVT
!     SUBROUTINE GRAMT
!     SUBROUTINE QUADP
!     SUBROUTINE SCM


!     FUNCTION GAMMAS
!     FUNCTION PHIFN
!     FUNCTION PHIY
!     FUNCTION FP_EQUAL
!
! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_STARTV2

   MODULE INTERFACE_STARTV2

      INTERFACE

         SUBROUTINE STARTV2(MU1,SIGMA,GAMMA,R,MAXJ,NGAM,CATFQ,DENUM,WA, &
                      NFN,ICEN,KG,RADD,IDIAG,IVSEP)
            implicit none
            real (kind=8),intent(in)::catfq(:),wa,denum,radd
            real(kind=8),intent(out)::mu1,gamma(:),sigma(:)
            integer(kind=4),intent(in)::r,maxj,ngam,nfn,icen,kg,idiag,ivsep

         END SUBROUTINE STARTV2

      END INTERFACE

   END MODULE INTERFACE_STARTV2

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_unique

   MODULE INTERFACE_unique

      INTERFACE

         SUBROUTINE unique(IDs,n,ID)

            IMPLICIT NONE
            ! Arguments:
            INTEGER,INTENT(IN) :: n
            INTEGER,INTENT(IN),DIMENSION(n)  :: IDs
            INTEGER,INTENT(OUT),DIMENSION(n) :: ID

         END SUBROUTINE unique

      END INTERFACE

   END MODULE INTERFACE_unique

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_FP_EQUAL

   MODULE INTERFACE_FP_EQUAL

      INTERFACE

         LOGICAL FUNCTION FP_EQUAL(A,B)

            DOUBLE PRECISION A,B

         END FUNCTION FP_EQUAL

      END INTERFACE

   END MODULE INTERFACE_FP_EQUAL

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_CHAMS

   MODULE INTERFACE_CHAMS

      INTERFACE

         SUBROUTINE CHAMS(A, B, N, MSA, MSB) 

            IMPLICIT NONE
            INTEGER, INTENT(IN) :: N 
            REAL(KIND=8), DIMENSION(N,N), INTENT(INOUT) :: A 
            REAL(KIND=8), DIMENSION(N,N), INTENT(INOUT) :: B 
            INTEGER, INTENT(IN) :: MSA 
            INTEGER, INTENT(IN) :: MSB 

         END SUBROUTINE CHAMS 

      END INTERFACE

   END MODULE INTERFACE_CHAMS

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_CHSKY

   MODULE INTERFACE_CHSKY

      INTERFACE

         SUBROUTINE CHSKY(A,B,N,NONPOS)
            implicit none
            real(kind=8),intent(in)::a(:)
            real(kind=8),intent(out)::b(:)
            integer(kind=4),intent(in)::n
            integer(kind=4),intent(out)::nonpos
         END SUBROUTINE CHSKY

      END INTERFACE

   END MODULE INTERFACE_CHSKY

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_GEN

   MODULE INTERFACE_GEN

      INTERFACE

         SUBROUTINE GEN (A,C,M,N,MS)                                       
             implicit none
            REAL (KIND=8), INTENT(OUT), DIMENSION(:)     :: A
            real (kind=8), intent(in) :: c
            integer(kind=4), intent(in) :: m,n,ms

         END SUBROUTINE GEN

      END INTERFACE

   END MODULE INTERFACE_GEN

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_GRMCV

   MODULE INTERFACE_GRMCV

      INTERFACE

         SUBROUTINE GRMCV(A,B,X,C,N)                                       
            implicit none
            real (kind=8), intent(inout), dimension(:) :: a,b
            real (kind=8), intent(in), dimension(:) :: x
            real(kind=8), intent(in) :: c
            integer(kind=4), intent(in) :: n

         END SUBROUTINE GRMCV

      END INTERFACE

   END MODULE INTERFACE_GRMCV

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_MPDSD

   MODULE INTERFACE_MPDSD

      INTERFACE

         SUBROUTINE MPDSD(A,B,C,N) 

           IMPLICIT NONE
           REAL(KIND=8), DIMENSION(:), INTENT(IN) :: A 
           REAL(KIND=8), DIMENSION(:), INTENT(INOUT) :: B,C 
           INTEGER, INTENT(IN) :: N 

         END SUBROUTINE MPDSD

      END INTERFACE

   END MODULE INTERFACE_MPDSD

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_HRECUR

   MODULE INTERFACE_HRECUR

      INTERFACE

         SUBROUTINE HRECUR(PN,DPN,PN1,X,NN)
            implicit none
            real(kind=8),intent(in)::x
            real(kind=8),intent(out)::pn,dpn,pn1
            integer(kind=4),intent(in)::nn
         END SUBROUTINE HRECUR

      END INTERFACE

   END MODULE INTERFACE_HRECUR

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_HROOT

   MODULE INTERFACE_HROOT

      INTERFACE

         SUBROUTINE HROOT(X,NN,DPN,PN1,EPSQ)

            USE INTERFACE_HRECUR
            implicit none
            real(kind=8),intent(in)::epsq
            real(kind=8),intent(inout)::x
            real(kind=8),intent(out)::dpn,pn1
            integer(kind=4),intent(in)::nn

         END SUBROUTINE HROOT

      END INTERFACE

   END MODULE INTERFACE_HROOT

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_HERMIT

   MODULE INTERFACE_HERMIT

      INTERFACE

         SUBROUTINE HERMIT(X,A,NN,EPSQ)

            USE INTERFACE_HROOT
            implicit none
            real(kind=8),intent(in)::epsq
            real(kind=8),intent(out)::x(:),a(:)
            integer(kind=4),intent(in)::nn

         END SUBROUTINE HERMIT

      END INTERFACE

   END MODULE INTERFACE_HERMIT

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_QUADP

   MODULE INTERFACE_QUADP

      INTERFACE

         SUBROUTINE QUADP(B,B1,A,NQ1,NQ,NDIM,IUNIF,WA1,WA2,SCA)

            USE INTERFACE_HERMIT
            IMPLICIT NONE
            INTEGER, INTENT(IN)                                    :: NQ1,NDIM,IUNIF
            INTEGER, INTENT(OUT)                                   :: NQ
            REAL (KIND=8), INTENT(INOUT), DIMENSION(NQ1)           :: B1,WA1
            REAL (KIND=8), INTENT(INOUT), DIMENSION(NQ1*NDIM*NDIM) :: B,WA2
            REAL (KIND=8), INTENT(INOUT), DIMENSION(NQ1*NDIM)      :: A
            REAL (KIND=8), INTENT(INOUT), OPTIONAL                 :: SCA

         END SUBROUTINE QUADP

      END INTERFACE

   END MODULE INTERFACE_QUADP

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_KMPY

   MODULE INTERFACE_KMPY

      INTERFACE

         SUBROUTINE KMPY(A,B,C,P,Q,MS,R,S)  
            implicit none
            integer(kind=4),intent(in)::p,q,ms,r,s
            real(kind=8),intent(in)::a(p,q),b(r,s)
            real(kind=8),intent(out)::c(:)
         END SUBROUTINE KMPY

      END INTERFACE

   END MODULE INTERFACE_KMPY

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_MPYM

   MODULE INTERFACE_MPYM

      INTERFACE

         SUBROUTINE MPYM(A,B,C,MA,NA,MSA,MSB,NB)                          
            implicit none
            real(kind=8),intent(in)::a(:)
            real(kind=8),intent(inout)::b(:),c(:)
            integer(kind=4),intent(in)::ma,na,msa,msb,nb
         END SUBROUTINE MPYM

      END INTERFACE

   END MODULE INTERFACE_MPYM

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_YSAME

   MODULE INTERFACE_YSAME

      INTERFACE

         SUBROUTINE YSAME(YVEC,N,NSAME)

            USE INTERFACE_FP_EQUAL
            implicit none
            integer(kind=4),intent(in)::n
            real(kind=8),intent(in)::yvec(:)
            integer(kind=4),intent(out)::nsame

         END SUBROUTINE YSAME

      END INTERFACE

   END MODULE INTERFACE_YSAME
! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_ADJR

   MODULE INTERFACE_ADJR

      INTERFACE

         SUBROUTINE ADJR(A,B,C,MA,N,MB)
            implicit none
            integer(kind=4),intent(in)::n,ma,mb
            real(kind=8),intent(in)::a(ma,n),b(mb,n)
            real(kind=8),intent(out)::c(:)
         END SUBROUTINE ADJR

      END INTERFACE

   END MODULE INTERFACE_ADJR

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_ADJC

   MODULE INTERFACE_ADJC

      INTERFACE

         SUBROUTINE ADJC(A,B,C,M,NA,NB)
            implicit none
            real(kind=8),intent(in)::a(:),b(:)
            real(kind=8),intent(out)::c(:)
            integer(kind=4),intent(in)::m,na,nb
         END SUBROUTINE ADJC

      END INTERFACE

   END MODULE INTERFACE_ADJC

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_INVS

   MODULE INTERFACE_INVS

      INTERFACE

         SUBROUTINE INVS(A,N,C,W,IER,VERBOSE)  
            implicit none
            real(kind=8),intent(inout)::a(:)
            real(kind=8),intent(out)::w(:),c
            integer(kind=4),intent(in)::n
            INTEGER, INTENT(inout), OPTIONAL :: IER
            LOGICAL, INTENT(in), OPTIONAL :: VERBOSE
         END SUBROUTINE INVS

      END INTERFACE

   END MODULE INTERFACE_INVS
! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_INVT

   MODULE INTERFACE_INVT

      INTERFACE

         SUBROUTINE INVT(A,N,C)                                        
            implicit none
            real(kind=8),intent(inout),dimension(:) :: a
            real(kind=8),intent(out) :: c
            integer(kind=4),intent(in) :: n

         END SUBROUTINE INVT

      END INTERFACE

   END MODULE INTERFACE_INVT
! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_GRAMT

   MODULE INTERFACE_GRAMT

      INTERFACE

         SUBROUTINE GRAMT(A,C,M,N)
            implicit none
            integer(kind=4),intent(in)::m,n
            REAL(kind=8),intent(in)::A(m,n)
            real(kind=8),intent(out)::C(:)
         END SUBROUTINE GRAMT

      END INTERFACE

   END MODULE INTERFACE_GRAMT

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_SCM

   MODULE INTERFACE_SCM

      INTERFACE

        SUBROUTINE SCM(A,C,B,M,N,MS)     
            implicit none
            real(kind=8), intent(in) :: a(:),c
            real(kind=8), intent(out) :: b(:)
            integer(kind=4), intent(in) :: m,n,ms
         END SUBROUTINE SCM

      END INTERFACE

   END MODULE INTERFACE_SCM

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_STARTVAL

   MODULE INTERFACE_STARTVAL

      INTERFACE

         SUBROUTINE STARTVAL(N,NPAR,NTOT,NALL,NOMU,P,R,RR,KS,NGAM,NRP1,IER, &
                    MAXJ,NFN,ICEN,IDIAG,IADD,IVSEP,IWT,KG,ICODE, &
                    IDNI,ALLDAT,MU,ALPHA,TAU,SIGMA,GAM,WA,RADD,sum2)

            USE INTERFACE_STARTV2
            USE INTERFACE_GRMCV
            USE INTERFACE_MPYM
            USE INTERFACE_INVS
            USE INTERFACE_FP_EQUAL
            IMPLICIT NONE
            ! Arguments:
            INTEGER, INTENT(IN)                            :: N,NPAR,NTOT,NALL,NRP1
            INTEGER, INTENT(IN)                            :: NOMU,P,R,RR,KS,NGAM
            INTEGER, INTENT(INOUT)                         :: IER
            INTEGER, INTENT(IN)                            :: MAXJ,NFN,ICEN
            INTEGER, INTENT(IN)                            :: IDIAG,IADD,IVSEP,IWT,KG
            INTEGER, INTENT(IN), DIMENSION(2*N)            :: IDNI
            REAL (KIND=8), INTENT(IN), DIMENSION(MAXJ)     :: ICODE
            REAL (KIND=8), INTENT(INOUT), DIMENSION(2*NALL + 1) :: ALLDAT
            REAL (KIND=8), INTENT(INOUT), DIMENSION(R)     :: MU
            REAL (KIND=8), INTENT(INOUT), DIMENSION(P)     :: ALPHA
            REAL (KIND=8), INTENT(INOUT), DIMENSION(KS)    :: TAU
            REAL (KIND=8), INTENT(INOUT), DIMENSION(RR)    :: SIGMA
            REAL (KIND=8), INTENT(INOUT), DIMENSION(NGAM)  :: GAM
            REAL (KIND=8), INTENT(OUT)                     :: WA,RADD,sum2

         END SUBROUTINE STARTVAL

      END INTERFACE

   END MODULE INTERFACE_STARTVAL

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_Empirical_Prior

   MODULE INTERFACE_Empirical_Prior

      INTERFACE

         SUBROUTINE Empirical_Prior(N,R,IPRIOR,IRT,IRTT,NQ,NQR,NQRR,&
                             ITHETA,ITHETAV,IAQ,IBQ,IWORKR)

            USE INTERFACE_MPDSD
            IMPLICIT NONE
            ! Arguments:
            INTEGER, INTENT(IN)                            :: N,R,IPRIOR,IRT,IRTT
            INTEGER, INTENT(IN)                            :: NQR,NQRR,NQ
            REAL (KIND=8), INTENT(INOUT), DIMENSION(IRT)   :: ITHETA
            REAL (KIND=8), INTENT(INOUT), DIMENSION(IRTT)  :: ITHETAV
            REAL (KIND=8), INTENT(INOUT), DIMENSION(R)     :: IWORKR
            REAL (KIND=8), INTENT(IN), DIMENSION(NQR)      :: IAQ
            REAL (KIND=8), INTENT(IN), DIMENSION(NQRR)     :: IBQ

         END SUBROUTINE Empirical_Prior

      END INTERFACE

   END MODULE INTERFACE_Empirical_Prior

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_MIXORDEST

   MODULE INTERFACE_MIXORDEST

      INTERFACE

         SUBROUTINE MIXORDEST(N,NPAR,NRP1,NALL,IDNI,NOMU,P,R,RR,KS,NGAM,IER, &
                     CONV,MAXJ,ALLDAT,MU,ALPHA,TAU,SIGMA,GAM, &
                     IT,RIDGEMAX,RLOGL,SE,AIC,SBC,DEV,AICD,SBCD,NQ1,AQUAD, &
                     IPRIOR,IUNIF,NFN,ICEN,IDIAG,IVSEP,IWT,KG,ICODE, &
                     WA,RADD,NPARR,IDER2,EBmean,EBvar,IRT,CHOLe,sum2)
                    
            USE INTERFACE_CHAMS
            USE INTERFACE_CHSKY
            USE INTERFACE_GEN
            USE INTERFACE_GRMCV
            USE INTERFACE_QUADP
            USE INTERFACE_KMPY
            USE INTERFACE_MPYM
            USE INTERFACE_YSAME
            USE INTERFACE_INVT
            USE INTERFACE_GRAMT
            USE INTERFACE_FP_EQUAL
            USE INTERFACE_Empirical_Prior
            IMPLICIT NONE
            ! Arguments:
            INTEGER, INTENT(IN)                            :: N,NPAR,NRP1,NALL,NQ1,AQUAD
            INTEGER, INTENT(IN), DIMENSION(2*N)            :: IDNI
            INTEGER, INTENT(OUT)                           :: IT
            INTEGER, INTENT(INOUT)                         :: NOMU,P,R,RR,KS,NGAM,IRT
            INTEGER, INTENT(INOUT)                         :: IER,NPARR
            INTEGER, INTENT(IN)                            :: IPRIOR,IUNIF,MAXJ,NFN
            INTEGER, INTENT(IN)                            :: ICEN,IDIAG,IVSEP,IWT,KG,CHOLe
            REAL (KIND=8), INTENT(IN), DIMENSION(2*NALL+1) :: ALLDAT
            REAL (KIND=8), INTENT(INOUT), DIMENSION(R)     :: MU
            REAL (KIND=8), INTENT(INOUT), DIMENSION(P)     :: ALPHA
            REAL (KIND=8), INTENT(INOUT), DIMENSION(KS)    :: TAU
            REAL (KIND=8), INTENT(INOUT), DIMENSION(RR)    :: SIGMA
            REAL (KIND=8), INTENT(INOUT), DIMENSION(NGAM)  :: GAM
            REAL (KIND=8), INTENT(IN), DIMENSION(MAXJ)     :: ICODE
            REAL (KIND=8), INTENT(IN)                      :: RADD,CONV
            REAL (KIND=8), INTENT(OUT)                     :: RLOGL,AIC,SBC,DEV,WA,sum2
            REAL (KIND=8), INTENT(OUT)                     :: AICD,SBCD
            REAL (KIND=8), INTENT(OUT)                     :: RIDGEMAX
            REAL (KIND=8), INTENT(INOUT), DIMENSION(NPAR)  :: SE
            REAL (KIND=8), INTENT(INOUT), DIMENSION(NPAR*NPAR) :: IDER2
            real(kind=8), intent(inout), dimension(n,irt)    :: EBmean
            real(kind=8), intent(inout), dimension(n,irt*(irt+1)/2)   :: EBvar

         END SUBROUTINE MIXORDEST

      END INTERFACE

   END MODULE INTERFACE_MIXORDEST

! ----------------------------------------------------------------------------
!@  MODULE INTERFACE_mainloop

   MODULE INTERFACE_mainloop

      INTERFACE

         SUBROUTINE mainloop(Y, X, W, WTs, NPAR, NTOT, N, IDS, &
            !                \__________/  \__/  \__/  \__/ 
            !                   data          from READAT
            !
                             P, R, RR, KS, NGAM, &
            !                \_________________/
            !                # of parameters
            !
                             MU, ALPHA, TAU, SIGMA, GAM, &
            !                \________________________/
            !                mean,cov.,scale,random eff.,threshold
            !
                             IT, RIDGEMAX, RLOGL, SE, AIC, SBC, DEV, AICD, SBCD, &
            !                \_/ \______/  \___/  \_/  \_____/  \____________/
            !                iter Ridge Log L Dev. Criterion  Log Lx(-2)Criterion
            !
                             CONV, MAXJ, IWT, IPRIOR, IUNIF, NQ1, AQUAD, NFN, &
            !
                             ICEN, KG, IADD, IDIAG, NOMU, IVSEP, ICODE, CHOL, &
            !                \______________________________________________/
            !                read from .def file  
            !                       
                             NPARR,         IDER2, EBmean, EBvar,   IRT,       CEN)
            !                \___/          \___/  \____/  \___/    \_/
            !                              Hessian EB means EB var EB size  Offset column
            !
            USE INTERFACE_STARTVAL
            USE INTERFACE_MIXORDEST 
            IMPLICIT NONE
            ! Arguments:
            INTEGER, INTENT(IN)                            :: NTOT, N
            INTEGER, INTENT(IN)                            :: NQ1,IUNIF,AQUAD
            INTEGER, INTENT(IN), DIMENSION(NTOT)           :: IDS
            INTEGER, INTENT(OUT)                           :: IT
            INTEGER, INTENT(IN)                            :: ICEN,IDIAG,IPRIOR,IADD
            INTEGER, INTENT(IN)                            :: IVSEP,IWT,KG
            INTEGER, INTENT(IN)                            :: MAXJ,NFN,CHOL
            INTEGER, INTENT(INOUT)                         :: NOMU,P,R,RR,KS,NGAM,IRT
            INTEGER, INTENT(INOUT)                         :: NPAR,NPARR
            INTEGER, INTENT(IN), DIMENSION(NTOT), OPTIONAL :: CEN ! note: changed as an integer
            REAL (KIND=8), INTENT(IN), DIMENSION(NTOT)     :: Y
            REAL (KIND=8), INTENT(IN), DIMENSION(NTOT,R)   :: X
            REAL (KIND=8), INTENT(IN), DIMENSION(NTOT,P)   :: W
            REAL (KIND=8), INTENT(IN), DIMENSION(NTOT)     :: WTs
            REAL (KIND=8), INTENT(IN), DIMENSION(MAXJ)     :: ICODE
            REAL (KIND=8), INTENT(INOUT), DIMENSION(R)     :: MU
            REAL (KIND=8), INTENT(INOUT), DIMENSION(P)     :: ALPHA
            REAL (KIND=8), INTENT(INOUT), DIMENSION(KS)    :: TAU
            REAL (KIND=8), INTENT(INOUT), DIMENSION(RR)    :: SIGMA
            REAL (KIND=8), INTENT(INOUT), DIMENSION(NGAM)  :: GAM
            REAL (KIND=8), INTENT(OUT)                     :: RLOGL,AIC,SBC
            REAL (KIND=8), INTENT(OUT)                     :: DEV,AICD,SBCD
            REAL (KIND=8), INTENT(OUT)                     :: RIDGEMAX
            REAL (KIND=8), INTENT(IN)                      :: CONV
            REAL (KIND=8), INTENT(INOUT), DIMENSION(NPAR)  :: SE
            REAL (KIND=8), INTENT(INOUT), DIMENSION(NPAR*NPAR) :: IDER2
            real(kind=8), intent(inout), dimension(N,IRT)    :: EBmean
            real(kind=8), intent(inout), dimension(N,IRT*(IRT+1)/2)   :: EBvar

         END SUBROUTINE mainloop

      END INTERFACE

   END MODULE INTERFACE_mainloop

! ----------------------------------------------------------------------------



!
!*********************************************************************
!               **********************                         
!               *SUBROUTINE mainloop(Y,X,W,WTs,NPAR,NTOT,IDS, &
!                         P,R,RR,KS,NGAM,MU,ALPHA,TAU,SIGMA,GAM, &
!                         IT,RIDGEMAX,RLOGL,SE,AIC,SBC,DEV,AICD,SBCD, &
!                         CONV,MAXJ,IWT,IPRIOR,IUNIF,&
!                         NQ1,NFN,ICEN,KG,IADD,IDIAG,NOMU,IVSEP, &
!                         ICODE,CHOL,NPARR,IDER2,ebmean,ebvar,irt,cen)
!               **********************                         
!*********************************************************************

SUBROUTINE mainloop(Y,X,W,WTs,NPAR,NTOT,N,IDS, &
                    P,R,RR,KS,NGAM,MU,ALPHA,TAU,SIGMA,GAM, &
                    IT,RIDGEMAX,RLOGL,SE,AIC,SBC,DEV,AICD,SBCD, &
                    CONV,MAXJ,IWT,IPRIOR,IUNIF,&
                    NQ1,AQUAD,NFN,ICEN,KG,IADD,IDIAG,NOMU,IVSEP, &
                    ICODE,CHOL,NPARR,IDER2,ebmean,ebvar,IRT,CEN)

   ! FOR ORDER FOR EACH SUBJECT'S DATA ==>                      
   ! Y  = VECTOR - NI DEPENDENT VARIABLES                       
   ! X  = VECTOR - RNI (R*NI) DESIGN FOR RANDOM EFFECTS         
   ! W  = VECTOR - PNI (P*NI) FIXED COVARIATES          
   !   
   ! N    (total number of level 2 units)
   ! NTOT (total number of observations)
   ! IDNI   = VECTOR (2 * MAXN) contains IDs and number of      
   !          observations per ID                               
   ! FOR ORDER FOR EACH SUBJECT'S DATA ==>                      
   ! ID = SCALAR - SUBJECT'S ID                                 
   ! NI = SCALAR - NOBS FOR SUBJECT                             
   !                                      
   ! P        : DIMENSION OF FIXED ALPHA SPACE
   ! R        : DIMENSION OF RANDOM BETA SPACE
   ! RR       : number of random effects 
   !            ->r.e. correl RR = (R * (R+1)) / 2
   !            ->r.e. indep  RR = R
   ! KS       : number of scale covariates
   ! NGAM     : dimension of thresholds
   !
   ! MU       : mean value
   ! ALPHA    : coeff. for expl vars, location
   ! TAU      : coeff. for expl vars, scale
   ! SIGMA    : random effects
   ! GAM      : Thresholds
   !
   ! IT       : Total Iterations
   ! RIDGEMAX : MaxMUm Ridge
   ! RLOGL    : Log Likelihood
   ! AIC      : Akaike's Information Criterion
   ! SBC      : Schwarz's Bayesian Criterion
   ! DEV      : Log Likelihood x (-2)
   ! AICD     : Akaike's Information Criterion x (-2)
   ! SBCD     : Schwarz's Bayesian Criterion x (-2)
   !
   ! NQ1      : Quad Pts per Dim
   ! NFN      : Response function 
   !            (0:normal, 1:logistic, 2:complementary log-log, 3:log-log
   ! IWT      = 0 EACH PERSON (2ND LEVEL UNIT) IS WEIGHTED EQUALLY
   !            1 FOR DIFFERENTIAL WEIGHTING FOR EACH PERSON (2ND LEVEL)
   !
   ! IPRIOR   = 0 specified PRIOR FOR RANDOM EFFECTS
   !            1 EMPIRICAL PRIOR FOR RANDOM EFFECTS     
   ! IUNIF    = 0 NORMAL PRIOR FOR RANDOM EFFECTS                 
   !            1 rectagular PRIOR FOR RANDOM EFFECTS    
   ! ICEN     = 0 no right-censoring (or censored observations are coded
   !                                EXPLICITLY as last category)
   !            1 right-censoring 
   ! NQ1      : number of quadrature nodes per dimension          
   ! AQUAD    = 1 use adaptive quadrature
   !          = 0 use non-adaptive
   !
   ! NFN      = 0 probit                                          
   !            1 logistic                                                
   !            2 complementary log-log
   !            3 log-log
   ! KG       = number of covariates to interact with threshold parameters
   !            (the first KG variables in W)                        
   !
   ! IADD     =  1      add covariates & mean of random effects to   thresholds 
   !          = -1 subtract covariates & mean of random effects from thresholds 
   !
   ! NCON     = number of transforms of estimated parameters 
   !            (linear re-expressions)                          
   !
   ! IDIAG    = 0 correlated random effects                       
   !            1 independent random effects                              
   !
   ! NOMU     = 0 estimate the mean of the random effects         
   !            1 don't estimate the mean of the random effects 
   !            (assumed to be zero)
   !
   ! IVSEP    = 0 R random effects don't indicate separate groups of subjects
   !            1 R random effects do    indicate separate groups of subjects
   !
   ! CHOL     = 0 Return var/cov matrix of random effects
   !          = 1 Return Cholesky of random effect matrix
   !
   
   USE INTERFACE_STARTVAL
   USE INTERFACE_MIXORDEST
   IMPLICIT NONE
   INTEGER, INTENT(IN)                            :: NTOT,N
   INTEGER, INTENT(IN)                            :: NQ1,IUNIF,AQUAD
   INTEGER, INTENT(IN), DIMENSION(NTOT)           :: IDS
   INTEGER, INTENT(OUT)                           :: IT
   INTEGER, INTENT(IN)                            :: IDIAG,IPRIOR,IADD
   INTEGER, INTENT(IN)                            :: IVSEP,IWT,KG
   INTEGER, INTENT(IN)                            :: MAXJ,NFN,CHOL
   INTEGER, INTENT(INOUT)                         :: icen,NOMU,P,R,RR,KS,NGAM,IRT
   INTEGER, INTENT(INOUT)                         :: NPAR,NPARR
   INTEGER, INTENT(IN), DIMENSION(NTOT), OPTIONAL :: CEN
   REAL (KIND=8), INTENT(IN), DIMENSION(NTOT)     :: Y
   REAL (KIND=8), INTENT(IN), DIMENSION(NTOT,R)   :: X
   REAL (KIND=8), INTENT(IN), DIMENSION(NTOT,P)   :: W
   REAL (KIND=8), INTENT(IN), DIMENSION(NTOT)     :: WTs
   REAL (KIND=8), INTENT(IN), DIMENSION(MAXJ)     :: ICODE
   REAL (KIND=8), INTENT(INOUT), DIMENSION(R)     :: MU
   REAL (KIND=8), INTENT(INOUT), DIMENSION(P)     :: ALPHA
   REAL (KIND=8), INTENT(INOUT), DIMENSION(KS)    :: TAU
   REAL (KIND=8), INTENT(INOUT), DIMENSION(RR)    :: SIGMA
   REAL (KIND=8), INTENT(INOUT), DIMENSION(NGAM)  :: GAM
   REAL (KIND=8), INTENT(OUT)                     :: RLOGL, AIC, SBC
   REAL (KIND=8), INTENT(OUT)                     :: DEV, AICD, SBCD
   REAL (KIND=8), INTENT(OUT)                     :: RIDGEMAX
   REAL (KIND=8), INTENT(IN)                      :: CONV
   REAL (KIND=8), INTENT(INOUT), DIMENSION(NPAR)  :: SE
   REAL (KIND=8), INTENT(INOUT), DIMENSION(NPAR*NPAR) :: IDER2
   real(kind=8), intent(out), dimension(n,irt)      :: EBmean
    real(kind=8), intent(out), dimension(n,irt*(irt+1)/2)    :: EBvar
! Local variables:
   !INTEGER                                        :: N
   INTEGER                                        :: I,H,J,L!HR,
   INTEGER                                        :: NALL,NRP1
   INTEGER                                        :: ICT,IER,IICT,JJ,M
   INTEGER, ALLOCATABLE                           :: IDNI(:),NWT(:)
   REAL (KIND=8)                                  :: RADD,WA,sum2
   REAL (KIND=8), ALLOCATABLE                     :: ALLDAT(:),WT(:)
!   real(kind=8), allocatable::isigmas(:,:), isigmavs(:,:)
   ! ALLDAT = VECTOR ((MAXNI*(R+P+1)) * MAXN) ALL DATA          
   !          or equals (MAXNI*(R+P+2)) * MAXN) if iofs=1   


   IF (ICEN == 1 .and. .not. present(CEN)) icen = 0

   ! figures out IDNI
   CALL IDNOBS()


   NRP1 = 1 + R + P
   IF (ICEN .EQ. 1) NRP1 = NRP1 + 1

   IF (IWT .EQ. 0) THEN
      NALL = NRP1*NTOT  
   ELSE
      NALL = NRP1*NTOT + N
   ENDIF

   ! ******* kkim32: Put Y,X,W back into ALLDAT
   ALLOCATE (ALLDAT(2*NALL + 1))
   ALLDAT(:) = 0.0D0

   M   = 0
   JJ  = 1
   ICT = 0
   IICT = 0
   DO J = 1, NTOT
      ICT = 1 + NRP1*(J-1) + IICT
      IF (IWT .EQ. 1) THEN
         M = M +1
         IF (M .EQ. NWT(JJ)) THEN
            ALLDAT(1 + NRP1*J + IICT) = WT(JJ)
            JJ = JJ + 1
            IICT = IICT +1
            M = 0
         ENDIF
      ENDIF
      ALLDAT(ICT) = Y(J)
   END DO
   M   = 0
   JJ  = 1
   ICT = 0
   IICT = 0
   DO J = 1, NTOT
      IF (IWT .EQ. 1) M = M +1
      DO H=1,R
         ICT = 1 + NRP1*(J-1)+ H + IICT
         IF (IWT .EQ. 1) THEN
            IF (H .EQ. R .AND. M .EQ. NWT(JJ)) THEN
               JJ = JJ + 1
               IICT = IICT +1
               M = 0
            ENDIF
         ENDIF
         ALLDAT(ICT) = X(J,H)
      END DO
   END DO
   M   = 0
   JJ  = 1
   ICT = 0
   IICT = 0
   DO J = 1, NTOT
      IF (IWT .EQ. 1) M = M +1
      DO L=1,P
         ICT = 1 +  NRP1*(J-1)+ R + L + IICT
         IF (IWT .EQ. 1) THEN
            IF (L .EQ. P .AND. M .EQ. NWT(JJ)) THEN
               JJ = JJ + 1
               IICT = IICT +1
               M = 0
            ENDIF
         ENDIF
         ALLDAT(ICT) = W(J,L) 
      END DO
   END DO
   IF (ICEN .EQ. 1) THEN
      M   = 0
      JJ  = 1
      ICT = 0
      IICT = 0
      DO J = 1, NTOT
         ICT = 1 + NRP1*(J-1)+ R + P + 1 + IICT
         IF (IWT .EQ. 1) THEN
            M = M +1
            IF (M .EQ. NWT(JJ)) THEN
               JJ = JJ + 1
               IICT = IICT +1
               M = 0
            ENDIF
         ENDIF
         ALLDAT(ICT) = FLOAT(CEN(J))
      END DO   
   ELSE
   ENDIF

  !write(*,*) NTOT,N,NALL
  !WRITE(6,"(F15.1)")(ALLDAT(J),J=1,30)

  !WRITE(6,"(F15.1)")(Y(J),J=1,50)
  !DO H=1,R
  !WRITE(6,"(F15.1)")(X(J,H),J=1,30)
  !END DO
  
  !DO L=1,P
  !WRITE(6,"(F15.1)")(W(J,L),J=1,30)
  !END DO
  !write(*,*) ( WTs(J), J=1,NTOT)
  !write(*,*) ( WT(J), J=1,N)
  

   !IF (ICEN .EQ. 1) WRITE(*,*)(CEN(J),J=1,10)


   ! ******* END putting Y,X,W back into ALLDAT

   CALL STARTVAL(N,NPAR,NTOT,NALL,NOMU,P,R,RR,KS,NGAM,NRP1,IER, &
                 MAXJ,NFN,ICEN,IDIAG,IADD,IVSEP,IWT,KG,ICODE, &
                 IDNI,ALLDAT,MU,ALPHA,TAU,SIGMA,GAM,WA,RADD,sum2)


   CALL MIXORDEST(N,NPAR,NRP1,NALL,IDNI,NOMU,P,R,RR,KS,NGAM,IER, &
                  CONV,MAXJ,ALLDAT,MU,ALPHA,TAU,SIGMA,GAM, &
                  IT,RIDGEMAX,RLOGL,SE,AIC,SBC,DEV,AICD,SBCD,NQ1,AQUAD, &
                  IPRIOR,IUNIF,NFN,ICEN,IDIAG,IVSEP,IWT,KG,ICODE, &
                  WA,RADD,NPARR,IDER2,EBmean,EBvar,IRT,CHOL,sum2)

   ! All global and any local arrays that were allocated
   ! are freed in preparation for starting over again.
   !CALL FREE_GLOBALS()
   !Deallocate local arrays
   !IF (ALLOCATED(GAM0)) DEALLOCATE(GAM0)
   !IF (ALLOCATED(SIGMA0)) DEALLOCATE(SIGMA0)
   !IF (ALLOCATED(ALPHA0)) DEALLOCATE(ALPHA0)
   !IF (ALLOCATED(TAU0)) DEALLOCATE(TAU0)
   !IF (ALLOCATED(MU0)) DEALLOCATE(MU0)

   DEALLOCATE( ALLDAT, IDNI )
   IF (IWT .EQ. 1) DEALLOCATE( WT )

CONTAINS

!*********************************************************************
!         **********************
!         *  SUBROUTINE IDNOBS()
!         **********************
!
!  instead of getting this info from READAT, figure out IDNI
!
!  CALL IDNOBS()
!
!*********************************************************************
   SUBROUTINE IDNOBS()

   IMPLICIT NONE
   INTEGER :: I,J,K,IDTEMP,IDOLD,IC2,PASS,NSize
   REAL (KIND = 8) :: WTTEMP,WTOLD
   LOGICAL FIRST


   NSize = size(IDS) ! NTOT
   !write(*,*) NSize

   DO PASS = 1,2

      IF (PASS .EQ. 2) THEN
         ! write(6,"(' TG Allocating alldat(',I6,'), IDNI(',I6,')')")
         !+         ICOUNT,2*N
         ! IDNI has IDs and Nobs per ID
         ALLOCATE (IDNI(2*N))
         IDNI(:) = 0
         IF (IWT .EQ. 1) THEN ! DIFFERENT WEIGHTING FOR EACH PERSON
            ALLOCATE (WT(N)) 
            WT(:) = 0.0D0
            ALLOCATE (NWT(N))
            NWT(:)     = 0
         ENDIF
      ENDIF
  
      I      = 1
      K      = 1
      FIRST  = .TRUE.

      DO J=1,NSize  ! loop over all observations
         IDTEMP = IDS(J)
         IF (IWT .EQ. 1) WTTEMP  = WTs(J)
         ! SET IDNI ACCORDINGLY
         IF (.NOT. FIRST) THEN
            IF (R .GE. 1 .AND. IDTEMP .EQ. IDOLD) THEN
               K     = K+1
            ELSE
               IF (IWT .EQ. 1) THEN
                  IF (PASS .EQ. 2) WT(I) = WTOLD
                  IF (PASS .EQ. 2) NWT(I) = K
                  WTOLD  = WTTEMP
               ENDIF
               IC2 = 2*I
               IF (PASS .EQ. 2) THEN
                  IDNI(IC2-1) = IDOLD
                  IDNI(IC2) = K
               ENDIF
               K     = 1
               I     = I+1
            ENDIF
         ELSE
            IF (IWT .EQ. 1) WTOLD  = WTTEMP
         ENDIF
      
         ! PUT TEMPORARY VALUES INTO DATA VECTORS AND MATRICES
         IDOLD = IDTEMP
         FIRST  = .FALSE.

      END DO   ! loop back to read next observations

      IC2 = 2*I
      IF (PASS .EQ. 2) THEN
         IDNI(IC2-1) = IDOLD
         IDNI(IC2) = K
         IF (IWT .EQ. 1) THEN
            WT(I) = WTOLD
            NWT(I) = K
         ENDIF
      ENDIF
      !N = I

   END DO

   END SUBROUTINE IDNOBS


END SUBROUTINE mainloop


!*********************************************************************
!               **********************
!               *  SUBROUTINE STARTVAL()
!               **********************
!   starting values for the regression coefficients
!                                                                       
!*********************************************************************
SUBROUTINE STARTVAL(N,NPAR,NTOT,NALL,NOMU,P,R,RR,KS,NGAM,NRP1,IER, &
                    MAXJ,NFN,ICEN,IDIAG,IADD,IVSEP,IWT,KG,ICODE, &
                    IDNI,ALLDAT,MU,ALPHA,TAU,SIGMA,GAM,WA,RADD,sum2)

   USE INTERFACE_STARTV2
   USE INTERFACE_GRMCV
   USE INTERFACE_MPYM
   USE INTERFACE_INVS
   USE INTERFACE_FP_EQUAL
   IMPLICIT NONE
   INTEGER, INTENT(IN)                            :: N,NPAR,NTOT,NALL,NRP1
   INTEGER, INTENT(IN)                            :: NOMU,P,R,RR,KS,NGAM
   INTEGER, INTENT(INOUT)                         :: IER
   INTEGER, INTENT(IN)                            :: MAXJ,NFN,ICEN
   INTEGER, INTENT(IN)                            :: IDIAG,IADD,IVSEP,IWT,KG
   INTEGER, INTENT(IN), DIMENSION(2*N)            :: IDNI
   REAL (KIND=8), INTENT(IN), DIMENSION(MAXJ)     :: ICODE
   REAL (KIND=8), INTENT(INOUT), DIMENSION(2*NALL+1) :: ALLDAT
   REAL (KIND=8), INTENT(INOUT), DIMENSION(R)     :: MU
   REAL (KIND=8), INTENT(INOUT), DIMENSION(P)     :: ALPHA
   REAL (KIND=8), INTENT(INOUT), DIMENSION(KS)    :: TAU
   REAL (KIND=8), INTENT(INOUT), DIMENSION(RR)    :: SIGMA
   REAL (KIND=8), INTENT(INOUT), DIMENSION(NGAM)  :: GAM
   REAL (KIND=8), INTENT(OUT)                     :: WA,RADD,sum2
   
   ! Local variables:
   INTEGER                                        :: IC,IC2,IC3,IC4
   INTEGER                                        :: ICAT,II,IR,J!,J2,J3
   INTEGER                                        :: H,K,KI,K2,K3,L,L2!,K4
   INTEGER                                        :: ICW!,KIND
   INTEGER                                        :: P1,P1P1
   INTEGER                                        :: ICCY
   REAL (KIND=8)                                  :: DENUM,WSUM!,RNTOT
   REAL (KIND=8)                                  :: UNO,CATADD,YDEV
   REAL (KIND=8)                                  :: DET!TEMPSUM,
   REAL (KIND=8), ALLOCATABLE, DIMENSION(:)       :: ICATFQ,IWORKCAT
   REAL (KIND=8), ALLOCATABLE, DIMENSION(:)       :: IWORKP1,IWP1P1
   REAL (KIND=8), ALLOCATABLE, DIMENSION(:)       :: IWRKP1
   !CHARACTER*160                                  :: LABTEMP,TEMPFORM
   LOGICAL                                        :: KEEP


   ! parameters         
   UNO  = 1.0D0

   ! NALL is the number of elements for the data read in from filedat 
   ! that ALLDAT() contains 

   !RNTOT = DBLE(NTOT)
   

   ! calculate means, standard deviations, proportions and starting values

   !MAXJXJ = MAXJ*MAXXJ
   ALLOCATE(ICATFQ(MAXJ))
   !ALLOCATE(ICATFQX(MAXJXJ))
   ALLOCATE(IWORKCAT(MAXJ))

   ICATFQ  = 0.0D0    ! Array assignment gets every element
   !ICATFQX = 0.0D0    ! Array assignment gets every element

   II = 1
   KI = 1
   WSUM = 0.0D0
   SUM2 = 0.0D0
   
   DO K  = 1,NALL
      K2 = K - (II - 1)
      IR = MOD(K2,NRP1)
      IF (IR .EQ. 0) IR = NRP1
      
      IF (IR .EQ. 1) THEN
         IF (IWT .EQ. 1 .AND. KI .EQ. (IDNI(II*2)+1)) THEN
            WSUM = WSUM + ALLDAT(K)*DBLE(IDNI(II*2))
            SUM2 = SUM2 + ALLDAT(K)
            KEEP = .FALSE.
            KI = 1
            II = II + 1
         ELSE
            KEEP = .TRUE.
            KI = KI + 1
         ENDIF
      ENDIF
      
      IF(KEEP) THEN
         IC     = NALL + IR
         IC2    = NALL + NRP1 + IR
         IC3    = NALL + NRP1 + NRP1 + IR

         ! get the minMUms

         IF (K .LE. NRP1) THEN
            ALLDAT(IC) = ALLDAT(K)
         ELSE
            IF (ALLDAT(K) .LT. ALLDAT(IC)) ALLDAT(IC) = ALLDAT(K)
         ENDIF

         ! get the maxiMUms

         IF (K2 .LE. NRP1) THEN
            ALLDAT(IC2) = ALLDAT(K)
         ELSE
            IF (ALLDAT(K) .GT. ALLDAT(IC2)) ALLDAT(IC2) = ALLDAT(K)
         ENDIF

         ! get the sums
         ! multiply by the weight if appropriate

         IF (IWT .EQ. 0) THEN
            ALLDAT(IC3) = ALLDAT(IC3) + ALLDAT(K)
         ELSE
            ICW = (K + (IDNI(II*2)*NRP1) -IR+1) - ((KI-2)*NRP1)
            ALLDAT(IC3) = ALLDAT(IC3) + ALLDAT(ICW)*ALLDAT(K)
         ENDIF

         ! for Y, get the frequency in each category
         ! and add the weight if appropriate

         IF (IR .EQ. 1) THEN
            IF (IWT .EQ. 0) THEN
               CATADD = 1.0D0
            ELSE
               CATADD = ALLDAT(ICW)
            ENDIF
            
            DO K3 = 1,MAXJ
               IF (FP_EQUAL(ALLDAT(K) , ICODE(K3))) THEN
                  ICATFQ(K3) = ICATFQ(K3) + CATADD
                  ICAT      = K3
               ENDIF
            END DO
         ENDIF

         ! for one of the Xs or Ws get the crosstab by Y
         ! and add the weight if appropriate

         !IF (ICATYX .EQ. 1 .AND. IR .EQ. NCATYX) THEN
         !   DO K4 = 1,MAXXJ
         !      IF (ALLDAT(K) .GE. (CODEX(K4)-0.0001D0)  &
         !         .AND. ALLDAT(K) .LE. (CODEX(K4)+0.0001D0)) THEN
         !          KIND = ((ICAT-1)*MAXXJ) + K4
         !          ICATFQX(KIND) = ICATFQX(KIND) + CATADD
         !      ELSE
         !      ENDIF
         !   END DO
         !ELSE
         !ENDIF
      ENDIF
   END DO

   !    calculate the means

   ICCY = 0
   DO K  = 1,NRP1
      IC3    = NALL + NRP1 + NRP1 + K
      IF (IWT .EQ. 0) THEN
         ALLDAT(IC3) = ALLDAT(IC3) / DBLE(NTOT)
      ELSE
         ALLDAT(IC3) = ALLDAT(IC3) / WSUM
      ENDIF

      !  see if the mean of the 1 random effect is 1
      !  for intraclass correlation calculation later

      IF (K .EQ. 2 .AND. R .EQ. 1 .AND. ALLDAT(IC3)  &
         .GT. 0.999D0 .AND. ALLDAT(IC3) .LT. 1.001D0) &
         ICCY = ICCY + 1

   END DO

   ! get starting values for the p regression coefficient plus
   ! the mean of the 2nd random effect (when r = 2)
   !
   ! use IWORKP1() and IWRKP1() as work vectors (P1 = p + r - 1)
   !     IWP1P1() as a work vector (P1 * (P1+1) / 2)

   IC = 0

   IF (NOMU .EQ. 0) THEN
      P1 = P + R - 1
   ELSE
      P1 = P - 1
   ENDIF
   
   P1P1 = (P1 * (P1+1)) / 2
   ALLOCATE(IWORKP1(MAX(P1,NRP1)))
   ALLOCATE(IWRKP1(P1))
   ALLOCATE(IWP1P1(P1P1))
   
   IWRKP1 = 0.0D0    ! Array assignment gets every element
   IWP1P1 = 0.0D0    ! Array assignment gets every element

   ! get the sums of squared deviations about the means
   ! and work vectors for the regression coefficients

   II = 1
   KI = 1
   
   DO K  = 1,NALL
      K2 = K - (II - 1)
      IR     = MOD(K2,NRP1)
      IF (IR .EQ. 0) IR = NRP1
      
      IF (IR .EQ. 1) THEN
         IF (IWT .EQ. 1 .AND. KI .EQ. (IDNI(II*2)+1)) THEN
            KEEP = .FALSE.
            KI = 1
            II = II + 1
         ELSE
            KEEP = .TRUE.
            KI = KI + 1
         ENDIF
      ENDIF
      
      IF(KEEP) THEN
         IC3    = NALL + NRP1 + NRP1 + IR
         IC4    = NALL + NRP1 + NRP1 + NRP1 + IR
         IF (IWT .EQ. 0) THEN
            ALLDAT(IC4) = ALLDAT(IC4) + ((ALLDAT(K) - ALLDAT(IC3))**2)
            IF (IR .EQ. 1) YDEV = ALLDAT(K) - ALLDAT(IC3)

            ! this assumes that the second column equals the intercept
            ! so you don't need to get the regression coefficient for
            ! ir eq 2

            IF (IR .GT. 2) THEN
               ! Don't allow IR2 to run off the end of the
               ! work vector TG 1/29/01
               IF(IR-2 .LE. UBOUND(IWORKP1,1)) THEN
                  IWORKP1(IR - 2) = ALLDAT(K) - ALLDAT(IC3)
               ENDIF
               IF (IR .EQ. NRP1) THEN
                  CALL GRMCV(IWP1P1,IWP1P1,IWORKP1,UNO,P1)
                  DO L = 1,P1
                     IWRKP1(L) = IWRKP1(L) + (YDEV * IWORKP1(L))
                  END DO
               ENDIF
            ENDIF
         ELSE
            ICW = (K + (IDNI(II*2) * NRP1) - IR+1) - ((KI-2)*NRP1)
            ALLDAT(IC4) = ALLDAT(IC4) + ALLDAT(ICW) * &
                   ((ALLDAT(K) - ALLDAT(IC3))**2)
            
            IF (IR .EQ. 1) YDEV = ALLDAT(ICW)*(ALLDAT(K) - ALLDAT(IC3))

            IF (IR .GT. 2) THEN
               ! Don't allow IR2 to run off the end of the
               ! work vector TG 1/29/01
               IF(IR-2 .LE. UBOUND(IWORKP1,1)) THEN
                  IWORKP1(IR - 2) = ALLDAT(K) - ALLDAT(IC3)
               ENDIF
               IF (IR .EQ. NRP1) THEN
                  CALL GRMCV(IWP1P1,IWP1P1,IWORKP1,ALLDAT(ICW),P1)
                  DO L = 1,P1
                     IWRKP1(L) = IWRKP1(L) + (YDEV * IWORKP1(L))
                  END DO
               ENDIF
            ENDIF
         ENDIF
      ENDIF
   END DO

   ! calculate the standard deviations

   DO K  = 1,NRP1
      IC4    = NALL + NRP1 + NRP1 + NRP1 + K
      IF (IWT .EQ. 0) THEN
         ALLDAT(IC4) = DSQRT(ALLDAT(IC4) / DBLE(NTOT-1))
      ELSE
         ALLDAT(IC4) = DSQRT(ALLDAT(IC4) / (WSUM - 1.0D0))
      ENDIF

      !  see if the SD of the 1 random effect is 0
      !  for intraclass correlation calculation later

      IF (K .EQ. 2 .AND. R .EQ. 1  &
         .AND. ALLDAT(IC4) .GT. -0.001D0  &
         .AND. ALLDAT(IC4) .LT. 0.001D0)  &
          ICCY = ICCY + 1
   END DO

   IF (IADD .EQ. -1) THEN
      RADD = -1.0D0
   !   WRITE(2,5574) 
   !   5574 FORMAT(1x,'Covariate(s) and random-effect(s) mean subtracted from thresholds', &
   !      /,1x,'==> positive coefficient = positive association between regressor', &
   !      /,1x,'    and ordinal outcome',/)
   ELSEIF (IADD .EQ. 1) THEN
      RADD =  1.0D0
   !   WRITE(2,5575) 
   !   5575 FORMAT(1x,'Covariate(s) and random-effect(s) mean added to thresholds', &
   !      /,1x,'==> positive coefficient = negative association between regressor', &
   !      /,1x,'    and ordinal outcome',/)
   ENDIF

 

   !WRITE(2,257)
   !257 FORMAT(//,1x,'Descriptive statistics for all variables', &
   !            /,1x,'----------------------------------------',/)
   !WRITE(2,357)
   !357 FORMAT(1x,'Variable',5X,'     MinMUm',5x,'     MaxMUm', &
   !        5x,'        Mean',5x,' Stand. Dev.',/)
   IC  = NALL + 1
   IC2 = NALL + NRP1 + 1
   IC3 = NALL + NRP1 + NRP1 + 1
   IC4 = NALL + NRP1 + NRP1 + NRP1 + 1
   !WRITE(2,377) YLABEL,ALLDAT(IC),ALLDAT(IC2),ALLDAT(IC3),ALLDAT(IC4)
  
   IF (R .GE. 1) THEN
      DO H = 1,R
         IC  = IC  + 1
         IC2 = IC2 + 1
         IC3 = IC3 + 1
         IC4 = IC4 + 1
   !      WRITE(2,377)BLAB(H),ALLDAT(IC),ALLDAT(IC2), &
   !               ALLDAT(IC3),ALLDAT(IC4)
      END DO
   ENDIF
   
   IF (P .GE. 1) THEN
      DO L = 1,P
         IC  = IC  + 1
         IC2 = IC2 + 1
         IC3 = IC3 + 1
         IC4 = IC4 + 1
   !      WRITE(2,377)ALAB(L),ALLDAT(IC), ALLDAT(IC2), &
   !               ALLDAT(IC3),ALLDAT(IC4)
      END DO
   ENDIF
   
   IF (ICEN .GE. 1) THEN
      IC  = IC  + 1
      IC2 = IC2 + 1
      IC3 = IC3 + 1
      IC4 = IC4 + 1
   !   WRITE(2,377)'Event   ',ALLDAT(IC),ALLDAT(IC2) &
   !               ,ALLDAT(IC3),ALLDAT(IC4)
   ENDIF

   !377 FORMAT(1x,A8,4(5X,F12.5))

   !WRITE(2,258) YLabel
   !258 FORMAT(//,1x,'Categories of the response variable ',A8, &
   !            /,1x,'--------------------------------------------',/)
   !WRITE(2,457)
   !457 FORMAT(1X,'Category',5X,'   Frequency',5x,'  Proportion',/)

   DO J = 1,MAXJ
      IF (IWT .EQ. 0) THEN
         DENUM = DBLE(NTOT)
      ELSE
         DENUM = WSUM
      ENDIF
      IWORKCAT(J) = ICATFQ(J) / DENUM
   !   WRITE(2,"(1X,F8.2,5X,F12.2,5x,f12.5)") ICODE(J),ICATFQ(J),IWORKCAT(J)
   END DO

   !IF (ICATYX .EQ. 1) THEN
   !   WRITE(2,358)XLabel,YLabel
   !   358 FORMAT(//,1x,'Crosstabulation of variable ',A8, &
   !              ' by the response variable ',A8, &
   !              /,1x,'----------------------------------------------------------------------',/)
   !   WRITE(2,372)YLabel,XLabel,(ICODE(J),J=1,MAXJ)
   !   372 FORMAT(1X,8X,A8,/,1X,8X,'--------',/,1X,A8,8F8.2)
   !   LABTEMP = '--------'
      
   !   DO J = 1,MAXJ
   !      L2 = J*8
   !      LABTEMP = LABTEMP(1:L2)//'--------'
   !   END DO
      
   !   L2 = L2 + 8
   !   LABTEMP = LABTEMP(1:L2)//'   Total'
   !   WRITE(2,"(1X,A80)") LABTEMP

   !   TEMPFORM = '(11X,'
   !   L2 = 5
   !   DO J = 1,MAXJ
   !      TEMPFORM = TEMPFORM(1:L2)//"' (',F4.2,') ',"
   !      L2 = L2 + 15
   !   END DO
   !   TEMPFORM = TEMPFORM(1:L2)//')'

   !   DO J = 1,MAXXJ
   !      TEMPSUM = 0.0D0
         
   !      DO J2= 1,MAXJ
   !         KIND = ((J2-1)*MAXXJ) + J
   !         TEMPSUM = TEMPSUM + ICATFQX(KIND) 
   !      END DO
         
   !      DO J2= 1,MAXJ
   !         KIND = ((J2-1)*MAXXJ) + J
   !         IWORKCAT(J2) = ICATFQX(KIND) / TEMPSUM
   !      END DO
         
   !      WRITE(2,"(/,1X,F8.2,8F8.1)")  &
   !         CODEX(J),(ICATFQX(J3),J3=J,MAXJXJ,MAXXJ),TEMPSUM
   !      WRITE(2,TEMPFORM)(IWORKCAT(J3),J3=1,MAXJ)
   !   END DO
   !   WRITE(2,"(/,1X,'Total   ',8F8.1)")(ICATFQ(J),J=1,MAXJ),DENUM
   !ELSE
   !ENDIF

   ! done writing out descriptives
   

 !  IF (ISTART .NE. 1) THEN  ! if not read from file

      ! calculate the starting values for the regression coefficients

      ALPHA(:) = 0

      IF (P1 .GT. 0) THEN
        CALL INVS(IWP1P1,P1,DET,IWORKP1,IER,.FALSE.)
        CALL MPYM(IWP1P1,IWRKP1,IWORKP1,P1,P1,1,0,1)

        WA = 0.0D0
        IC3 = NALL + NRP1 + NRP1 + 1
        IC4 = NALL + NRP1 + NRP1 + NRP1 + 1

!       Changed the next IF since it seems like if nomu=1 then p1 = p-1 and not p 
!       IF (P1 .EQ. P) THEN
        IF (P1 .EQ. (P-1)) THEN
           DO L = 1,P
              ALPHA(L) = (0-IWORKP1(L)) / ALLDAT(IC4)
              WA = WA + (ALPHA(L) * ALLDAT(IC3+R+L))
           END DO
        ELSEIF (P1 .GT. P) THEN
           IF (NOMU .EQ. 0) THEN
              DO H = 2,R
                 MU(H) = (0-IWORKP1(H-1)) / ALLDAT(IC4)
                 WA = WA + (MU(H) * ALLDAT(IC3+H))
              END DO
           ENDIF
         
           DO L = 1,P
              L2 = L+1
              ALPHA(L) = (0-IWORKP1(L2)) / ALLDAT(IC4)
              WA = WA + (ALPHA(L) * ALLDAT(IC3+R+L))
           END DO
        ENDIF
      ELSE
        WA = 0.0D0
      ENDIF

      ! call starting value routine for intercept, 
      ! thresholds, and random-effects variance-covariance matrix
      IF (R .GT. 0 .OR. NGAM .GT. 0) THEN
        CALL STARTV2(MU(1),SIGMA,GAM,R,MAXJ, &
             NGAM,ICATFQ,DENUM,WA,NFN,ICEN,KG,RADD,IDIAG,IVSEP)
      ENDIF

   IF (KS .GT. 0) THEN
      TAU(:) = 0.1D0
   ENDIF
!   ENDIF     ! ISTART .NE. 1
   ! write out AND save the starting values
!END SUBROUTINE STARTVALPRE

   !WRITE(2,"(//,1x,'Starting values',/,1x,'---------------',/)")
   !IF (r .ge. 1 .AND. NOMU .EQ. 0) THEN
   !   WRITE(2,"(1x,'mean       ',10F7.3)") (MU(H),H=1,R)
   !   ALLOCATE(MU0(R))
   !   CALL RELOC(MU,MU0,R,1,0)
   !ENDIF
   
   IF (P .GT. 0) THEN
      IF(R.EQ.0) ALPHA(1) = MU(1)
   !   WRITE(2,"(1x,'covariates ',10F7.3)") (ALPHA(L),L=1,P)
   !   ALLOCATE(ALPHA0(P))
   !   CALL RELOC(ALPHA,ALPHA0,P,1,0)
   ENDIF
   
   !IF (r .ge. 1) THEN
   !   WRITE(2,"(1x,'var. terms ',10F7.3)") (SIGMA(HR),HR=1,RR)
   !   ALLOCATE(SIGMA0(RR))
   !   CALL RELOC(SIGMA,SIGMA0,RR,1,0)
   !ENDIF
   
   !IF (NGAM .GT. 0) THEN
   !   WRITE(2,"(1x,'thresholds ',10F7.3)") (GAM(L),l=1,ngam)
   !   ALLOCATE(GAM0(NGAM))
   !   CALL RELOC(GAM,GAM0,NGAM,1,0)
   !ENDIF

   !IF (KS .GT. 0) THEN
   !   WRITE(2,"(1x,'scale terms',10F7.3)") (TAU(L),l=1,ks)
   !   ALLOCATE(TAU0(KS))
   !   CALL RELOC(TAU,TAU0,KS,1,0)
   !ENDIF

   DEALLOCATE( ICATFQ,IWORKCAT,IWORKP1,IWRKP1,IWP1P1 ) !ICATFQX,


END SUBROUTINE STARTVAL

!*********************************************************************
!               **********************
!               *  SUBROUTINE MIXORDEST()
!               **********************
!
!                                                                       
!*********************************************************************
SUBROUTINE MIXORDEST(N,NPAR,NRP1,NALL,IDNI,NOMU,P,R,RR,KS,NGAM,IER, &
                     CONV,MAXJ,ALLDAT,MU,ALPHA,TAU,SIGMA,GAM, &
                     IT,RIDGEMAX,RLOGL,SE,AIC,SBC,DEV,AICD,SBCD,NQ1,AQUAD, &
                     IPRIOR,IUNIF,NFN,ICEN,IDIAG,IVSEP,IWT,KG,ICODE, &
                     WA,RADD,NPARR,IDER2,EBmean,EBvar,IRT,CHOLe,sum2)

   USE INTERFACE_CHAMS
   USE INTERFACE_CHSKY
   USE INTERFACE_GEN
   USE INTERFACE_GRMCV
   USE INTERFACE_QUADP
   USE INTERFACE_KMPY
   USE INTERFACE_MPYM
   USE INTERFACE_YSAME
   USE INTERFACE_INVT
   USE INTERFACE_GRAMT
   USE INTERFACE_FP_EQUAL
   USE INTERFACE_Empirical_Prior
   use interface_scm
   IMPLICIT NONE
   INTEGER, INTENT(IN)                            :: N,NPAR,NRP1,NALL,NQ1,AQUAD
   INTEGER, INTENT(IN), DIMENSION(2*N)            :: IDNI
   INTEGER, INTENT(OUT)                           :: IT
   INTEGER, INTENT(INOUT)                         :: NOMU,P,R,RR,KS,NGAM,IRT
   INTEGER, INTENT(INOUT)                         :: IER,NPARR
   INTEGER, INTENT(IN)                            :: IPRIOR,IUNIF,MAXJ,NFN
   INTEGER, INTENT(IN)                            :: ICEN,IDIAG,IVSEP,IWT,KG,CHOLe
   REAL (KIND=8), INTENT(IN), DIMENSION(2*NALL+1) :: ALLDAT
   REAL (KIND=8), INTENT(INOUT), DIMENSION(R)     :: MU
   REAL (KIND=8), INTENT(INOUT), DIMENSION(P)     :: ALPHA
   REAL (KIND=8), INTENT(INOUT), DIMENSION(KS)    :: TAU
   REAL (KIND=8), INTENT(INOUT), DIMENSION(RR)    :: SIGMA
   REAL (KIND=8), INTENT(INOUT), DIMENSION(NGAM)  :: GAM
   REAL (KIND=8), INTENT(IN), DIMENSION(MAXJ)     :: ICODE
   REAL (KIND=8), INTENT(IN)                      :: RADD,CONV
   REAL (KIND=8), INTENT(OUT)                     :: RLOGL,AIC,SBC,DEV,WA,sum2
   REAL (KIND=8), INTENT(OUT)                     :: AICD,SBCD,RIDGEMAX
   REAL (KIND=8), INTENT(INOUT), DIMENSION(NPAR)  :: SE
   REAL (KIND=8), INTENT(INOUT), DIMENSION(NPAR*NPAR) :: IDER2
   real(kind=8), intent(inout), dimension(n,irt)    :: EBmean
   real(kind=8), intent(inout), dimension(n,irt*(irt+1)/2)   :: EBvar
! Local variables:
   INTEGER                                        :: TERMIN
   INTEGER                                        :: H,H2,IC,IC0,IC2,ICOUNT
   INTEGER                                        :: IDI,IH,IND,INDC,INDD,INDD2
   INTEGER                                        :: IW,J,K,KK,KKK,L,L0,L1,L2
   INTEGER                                        :: LM,NII,I,IX,PNII,R2,RNII
   INTEGER                                        :: RIDGEIT,Q,ITLAST,IFIN
   INTEGER                                        :: ISIG,NONPOS,IGO
   INTEGER                                        :: NQ,NQR,NQRR,NGAM1
   INTEGER                                        :: IRTT
   integer::pabove, pbelow, asCount, sCount, place, place0
   REAL (KIND=8)                                  :: DZT,RNPAR,RIDGE,RLOGLP
   REAL (KIND=8)                                  :: BIGCOR,CSTAT,DERIV,DERIV0
   REAL (KIND=8)                                  :: DERIV1,DET,HPROB,PROB
   REAL (KIND=8)                                  :: PROBP0,PROBP1,PSUM,QMULT
   REAL (KIND=8)                                  :: RLOGDIFF,SCAL,SIGN
   REAL (KIND=8)                                  :: STEP,SUMW,WTSUBI,XMU,XTB
   REAL (KIND=8)                                  :: Z,Z0,Z1,WTAU
   REAL (KIND=8)                                  :: PHIFN,phiy,sbcn
   REAL (KIND=8), ALLOCATABLE, DIMENSION(:)       :: IDER,IDER2S,ISIGTAU
   REAL (KIND=8), ALLOCATABLE, DIMENSION(:)       :: IWG
   REAL (KIND=8), ALLOCATABLE, DIMENSION(:)       :: IAQEMP,IBQ1,IAQ0,IBQ0,IAQ,IBQ
   REAL (KIND=8), ALLOCATABLE, DIMENSION(:)       :: IAQ1,IAQ2,IXI,IYI,IWI,IEV
   REAL (KIND=8), ALLOCATABLE, DIMENSION(:)       :: IDERP,IDERQ,ILIK
   REAL (KIND=8), ALLOCATABLE, DIMENSION(:)       :: ICHWRK,ICOREC
   REAL (KIND=8), ALLOCATABLE, DIMENSION(:)       :: IWORKR,IWRKR,IWORKR2
   REAL (KIND=8), ALLOCATABLE, DIMENSION(:)       :: ITHETA,ITHETAV
   REAL (KIND=8), ALLOCATABLE::sstar(:), sstar2(:), work(:), sstar2t(:), &
      asstar2(:), asstar2t(:),chol(:),cholIder2(:)
    real(kind=8)::sdev,phiratio
   !CHARACTER (LEN=8), ALLOCATABLE                 :: TEMPLABEL(:)
   LOGICAL CATJ

   ALLOCATE(ISIGTAU(R))
   ALLOCATE(IWORKR(R))

   ALLOCATE(IDER(NPAR))
   ALLOCATE(IDER2S(NPAR*NPAR))
   allocate(cholIder2(NPARR))
   allocate(chol(rr))

    chol = sigma

   ! IRT = dimension of theta
   RNPAR = DBLE(NPAR)

    !IRT = R
   !IF (IVSEP .EQ. 1) IRT = 1
   ALLOCATE(ITHETA(IRT))
   IRTT = IRT*(IRT+1)/2
   !if (IVSEP .EQ. 1) IRTT = R
   ALLOCATE(ITHETAV(IRTT))

   IF (ICEN.EQ.1) THEN
      NGAM1 = MAXJ-1
   ELSE 
      NGAM1 = MAXJ-2
   ENDIF
   ALLOCATE(IWG(NGAM1))


   ! ALLOCATE  nquad points per dimension (nq1)

   ALLOCATE(IBQ1(NQ1))

   ! ALLOCATE  nq1**r

    nqr = nq1 ** IRT
    nqrr = IRT * nqr

   ALLOCATE(IAQ(NQR))
   ALLOCATE(IAQ0(NQR))
   ALLOCATE(IAQEMP(NQR))

   ! ALLOCATE  R * (nq1**r)

   ALLOCATE(IBQ(NQRR))
   ALLOCATE(IBQ0(NQRR))

   ! get quadrature nodes & weights  

   ALLOCATE(IAQ1(NQ1))
   ALLOCATE(IAQ2(NQRR))

   CALL QUADP(IBQ0,IBQ1,IAQ0,NQ1,NQ,IRT,IUNIF,IAQ1,IAQ2)

   ! *****************************************************
   !  set up for iterations
   ! *****************************************************
   
   ITLAST = 0 
   RIDGE  = 0.0D0
   RIDGEMAX = 0.0D0
   ridgeit = 0
   IFIN   = 1
   !NORI   = 0
   !ND     = 0
   rloglp = -999999999999999.0D0

   ! isig   = 1 when INVS has problems (0 otherwise)
   ! ier    = 1 when INVS has problems (0 otherwise)
   ! nonpos = 0 when CHSKY detects a non pos-def matrix (1 when pos-def)

   ISIG   = 0
   IER    = 0
   NONPOS = 1
   !IRBAD  = 0

   IGO    = 1
   IT     = 0
   termin = 1


   ! *****************************************************
   !  start iterations
   ! *****************************************************

   DO WHILE(termin .NE. 0)
   IF (IGO .EQ. 1 .OR. ITLAST .EQ. 1) THEN

      IT  = IT + 1

      ! do step-halving for first 10 iterations

      IF (IT .LE. 10) THEN
         STEP = 0.5D0
      ELSE
         STEP = 1.0D0
      ENDIF

      ! get the exponential of the diagonal of SIGMA

      IF (IFIN .EQ. 0) THEN
         IH = 0
         DO H=1,R
            DO H2=1,H
               IH = IH + 1
               IF (H .EQ. H2) ISIGTAU(H) = DLOG(CHOL(IH))
            END DO
         END DO
      ENDIF

      ! ***
      ! LIKELIHOOD INITIALIZATION
      ! ***
      !                                              
      !  calculate and save the conditional likelihood - ILIK() 
      !                     the marginal likelihood    - hprob 
      !                     the log-likelihood value   - rlogl

      CALL gen(cholIDER2,0.0d0,npar,npar,1)
      CALL gen(IDER,0.0d0,npar,1,0)
      rlogl = 0.0d0

      IF (IPRIOR .EQ. 1) THEN
         DO Q = 1,NQ
            IAQEMP(Q) = 0.0D0
         END DO
      ENDIF

      ! *************************************************************
      ! GO OVER ALL SUBJECTS
      !    IUN = UNIT NUMBER FOR OUTPUT (0 = SCREEN)
      ! *************************************************************
      ICOUNT = 0

      IF (ALLOCATED(IDERP)) THEN
         IF(UBOUND(IDERP,1) .NE. NPAR) THEN
            DEALLOCATE(IDERP)
            ALLOCATE(IDERP(NPAR))
         ENDIF
      ELSE
         ALLOCATE(IDERP(NPAR))
      ENDIF

      DO I = 1,N
          
          ibq = ibq0
        iaq = iaq0
        if(it >= 5 .and. aquad .ne. 0) then
             place = 1
             do k=1,IRT
                 call getSDev(k,IRT,ebvar(i,:), sdev)
                 do q=1,nq
                     ibq(place) = ebmean(i,k) + sdev*ibq0(place)
                     call get_phi_ratio(ibq(place),ibq0(place), phiRatio)
!                     write(9,*) ibq(place), ibq0(place), phiRatio
                     place0 = mod(place-1,nq)+1
                     iaq(place0) = sdev*phiRatio*iaq(place0)
                     place = place + 1
                 end do
             end do
        end if
        
   
         hprob = 0.0d0
         CALL gen(IDERP,0.0d0,npar,1,0)
   
         IC2  = 2*(I-1)+1
         IDI  = IDNI(IC2)!response outside range for subject
         IC2  = IC2+1
         NII  = IDNI(IC2)
         RNII = NII * R
         PNII = NII * P
   
         ! THE YI(K) VECTOR  K = 1 ... NI(I)   DEPENDENT VARIABLE VECTOR
   
         IF (ALLOCATED(IYI)) THEN
            IF(UBOUND(IYI,1) .NE. NII) THEN
               DEALLOCATE(IYI)
               ALLOCATE(IYI(NII))
            ENDIF
         ELSE
            ALLOCATE(IYI(NII))
         ENDIF
         
         DO K = 1,NII
            IC2         = ICOUNT + (NRP1 * (K-1)) + 1 
            IYI(K) = ALLDAT(IC2)
         END DO
   
         ! THE X(K, H) MATRIX  K = 1 .. NI(I)  H = 1 .. R
   
         IF (R .GE. 1) THEN
            
            IF (ALLOCATED(IXI)) THEN
               IF(UBOUND(IXI,1) .NE. RNII) THEN
                  DEALLOCATE(IXI)
                  ALLOCATE(IXI(RNII))
               ENDIF
            ELSE
               ALLOCATE(IXI(RNII))
            ENDIF
            
            IC = 0
            DO H = 1,R
               DO K = 1,NII
                  IC           = IC + 1
                  IC2          = ICOUNT + (NRP1 * (K-1) + H+1) 
                  IXI(IC) = ALLDAT(IC2)
               END DO
            END DO
         ELSE
         ENDIF
   
         ! THE W(K, L) MATRIX  K = 1 .. NI(I)  L = 1 .. P
   
         IF (P .GE. 1) THEN
            
            IF (ALLOCATED(IWI)) THEN
               IF(UBOUND(IWI,1) .NE. PNII) THEN
                  DEALLOCATE(IWI)
                  ALLOCATE(IWI(PNII))
               ENDIF
            ELSE
               ALLOCATE(IWI(PNII))
            ENDIF
            
            IC = 0
            DO L  = 1,P   
               DO K  = 1,NII
                  IC           = IC + 1
                  IC2          = ICOUNT + (NRP1 * (K-1) + L+R+1)
                  IWI(IC) = ALLDAT(IC2)
               END DO
            END DO
         ENDIF
   
         ! THE EVENT VARIABLE variable EV(K) vector  K = 1 .. NI(I) 
   
         IF (ICEN .GE. 1) THEN
            
            IF (ALLOCATED(IEV)) THEN
               IF(UBOUND(IEV,1) .NE. NII) THEN
                  DEALLOCATE(IEV)
                  ALLOCATE(IEV(NII))
               ENDIF
            ELSE
               ALLOCATE(IEV(NII))
            ENDIF
            
            IC = 0
            DO K  = 1,NII
               IC           = IC + 1
               IC2          = ICOUNT + (NRP1 * (K-1) + P+R+2)
               IEV(IC) = ALLDAT(IC2)
            END DO
         ELSE
         ENDIF
   
         IF (ICEN .EQ. 0) THEN
            ICOUNT = ICOUNT + NII + RNII + PNII 
         ELSE
            ICOUNT = ICOUNT + NII + RNII + PNII + NII
         ENDIF
   
         ! THE WEIGHT FOR EACH LEVEL-2 UNIT
   
         IF (IWT .EQ. 1) THEN
            ICOUNT = ICOUNT + 1
            WTSUBI = ALLDAT(ICOUNT)
         ELSE
            WTSUBI = 1.0D0
         ENDIF
   
   
         ! GO OVER THE QUADRATURE POINTS
   
         IF (ALLOCATED(IDERQ)) THEN
            IF(UBOUND(IDERQ,1) .NE. NPAR) THEN
               DEALLOCATE(IDERQ)
               ALLOCATE(IDERQ(NPAR))
            ENDIF
         ELSE
            ALLOCATE(IDERQ(NPAR))
         ENDIF
         
         IF (ALLOCATED(ILIK)) THEN
            IF(UBOUND(ILIK,1) .NE. NQ) THEN
               DEALLOCATE(ILIK)
               ALLOCATE(ILIK(NQ))
            ENDIF
         ELSE
            ALLOCATE(ILIK(NQ))
         ENDIF

         DO q=1,nq    
      
            psum          = 0.0d0
            ILIK(Q)  = 0.0d0
            CALL gen(IDERQ,0.0d0,npar,1,0)

            DO k=1,nii   

               IF (NGAM .GE. 1) THEN
                  DO L=1,NGAM1
                     IWG(L) = GAM(L)
                  END DO
               ENDIF

               wa  = 0.0D0
               wtau = 0.0D0
               IF (P .GE. 1) THEN
                  DO l=1,p
                     iw = k  + (l-1)*nii
                     wa = wa + ALPHA(L)*IWI(iw)
            
                     IF (KG .GT. 0 .AND. L .LE. KG) THEN
                        DO L0=1,NGAM1
                           LM = L*NGAM1+L0
                           IWG(L0)=IWG(L0)+GAM(LM)*IWI(IW)*RADD
                     !       IWG(L0) = IWG(L0) + GAM(LM)*IWI(IW)
                        END DO
                     ENDIF
                     IF (KS .GT. 0 .AND. L .LE. KS) THEN
                        wtau = wtau + TAU(L)*IWI(iw)
                     ENDIF
                  END DO
               ENDIF
         
               xmu = 0.0D0
               xtb = 0.0D0
               IF (R .GE. 1) THEN
                  DO H = 1,R
                     IF (IVSEP .EQ. 0) THEN
                        H2 = Q + (h-1)*NQ
                     ELSE
                        H2 = Q
                     ENDIF
                     IWORKR(H) = IBQ(H2)
                  END DO
                  IF (IDIAG .EQ. 0 .AND. IVSEP .EQ. 0) THEN
                     CALL MPYM(CHOL,IWORKR,IWORKR,R,R,3,0,1)
                  ELSE
                     CALL MPYM(CHOL,IWORKR,IWORKR,R,R,2,0,1)
                  ENDIF
                  DO h=1,r
                     ix  = k   + (h-1)    * nii
                     IF (NOMU .EQ. 0) THEN
                        xmu = xmu + MU(H)    * IXI(ix)
                     ELSE
                        XMU = 0.0D0
                     ENDIF
                     xtb = xtb + IWORKR(H) * IXI(ix)
                  END DO
               ENDIF
         
               !   probp0 is j-1 part of numerator  pdf function in derivatives for all 
               !   probp1 is jth part of numerator  pdf function in derivatives for all
               !   prob  is denominator cdf function in derivatives for all parameters
         
               catj = .TRUE.
               j    =  0
               
               DO while (catj)
                  j    = j + 1 
                  IF (j .gt. MAXJ) THEN
                     !write(6,*)'response outside range for subject ',idi
                     catj = .FALSE.
                  ELSEIF (FP_EQUAL(IYI(K) , ICODE(J))) THEN 
                     z    = xmu + wa + xtb
         
                     IF (ICEN .EQ. 0) THEN
                        INDC = 1
                     ELSEIF (ICEN .EQ. 1) THEN
                        IF (IEV(K) .EQ. 0) INDC = 0
                        IF (IEV(K) .NE. 0) INDC = 1
                     ENDIF
         
                     IF (j .eq. 1) THEN 
                        z1     =  RADD*z / DEXP(WTAU)
                        IF (INDC.EQ.1) THEN
                           prob   = phifn(z1,nfn)
                           probp0 = 0.0d0
                           probp1 = phiy(z1,nfn)
                        ELSE
                           prob   = 1.0D0 - phifn(z1,nfn)                 
                           probp0 = 0.0d0
                           probp1 = 0.0d0 - phiy(z1,nfn)
                        ENDIF
                        dzt = (0.0d0 - probp1)/prob * z1
                     ELSEIF (j .gt. 1 .and. j .lt. maxj) THEN
                        IF (j .eq. 2) THEN 
                           Z0   = radd*Z  / DEXP(WTAU)
                        ELSE
                           z0   = (IWG(j-2) + RADD*z) / DEXP(WTAU)
                        ENDIF
                        z1   = (IWG(j-1) + RADD*z)  / DEXP(WTAU)
                        IF (INDC.EQ.1) THEN
                           prob   = phifn(z1,nfn) - phifn(z0,nfn)
                           probp0 = 0.0d0 - phiy(z0,nfn)
                           probp1 = phiy(z1,nfn)
                        ELSE
                           prob = 1.0D0 - phifn(z1,nfn)                 
                           probp0 = 0.0d0
                           probp1 = 0.0d0 - phiy(z1,nfn)
                        ENDIF
                     dzt = ((0.0d0 - probp0)*z0 - probp1*z1)/prob
                     ELSEIF (j .eq. maxj) THEN
                        IF (j .eq. 2) THEN 
                           z0   = RADD*z  / DEXP(WTAU)
                        ELSE
                           z0   = (IWG(j-2) + RADD*z) / DEXP(WTAU)
                        ENDIF
                        IF (ICEN.EQ.0) THEN
                           prob   = 1.0d0 - phifn(z0,nfn) 
                           probp0 = 0.0d0 - phiy(z0,nfn)
                           probp1 = 0.0d0
                           dzt = (0.0d0 - probp0)/prob * z0
                        ELSEIF(ICEN.EQ.1) THEN
                           z1     = (IWG(j-1) + RADD*z) / DEXP(WTAU)
                           IF(FP_EQUAL(IEV(K) , 1.0D0)) THEN
                              prob   = phifn(z1,nfn) - phifn(z0,nfn)
                              probp0 = 0.0d0 - phiy(z0,nfn)
                              probp1 = phiy(z1,nfn)
                              dzt = (0.0d0 - probp0)/prob * z0
                           ELSE
                              prob = 1.0D0 - phifn(z1,nfn)                 
                              probp0 = 0.0d0
                              probp1 = 0.0d0 - phiy(z1,nfn)
                              dzt = (0.0d0 - probp1)/prob * z0
                           ENDIF
                        ENDIF
                     ENDIF
            
                     IF (PROB .LT. .1D-305) PROB = .1D-305
                     psum = psum + DLOG(prob)
            
                     !    write(6,*)' before deriv',i,q,k
                     IF (prob  .le. 0.0D0) THEN
                        deriv = 0.0d0
                     ELSE
                        deriv = RADD*(probp0+probp1)/prob
                     ENDIF
            
                     !    write(6,*)' after deriv'
            
                     IF (R .GE. 1 .AND. NOMU .EQ. 0) THEN
                        DO h=1,r
                           ix  = k + (h-1) * nii
                           IDERQ(H) = IDERQ(H)+deriv*IXI(ix)/DEXP(WTAU)
                        END DO
                     ELSE
                     ENDIF
            
                     IF (P .GE. 1) THEN
                        IF (NOMU .EQ. 0) THEN
                           ind = r
                        ELSE
                           ind = 0
                        ENDIF
                        do l = 1,p
                           ind= ind + 1
                           iw = k + (l-1)*nii
                           IDERQ(IND) = IDERQ(IND) + deriv*IWI(iw)/DEXP(WTAU)
                        END DO
                     ELSE
                     ENDIF

                     IF (KS .GT. 0) THEN
                        IF (NOMU .EQ. 0) THEN
                           ind = r + p
                        ELSE
                           ind = p
                        ENDIF
                        DO L=1,KS
                           ind= ind + 1
                           iw = k + (l-1)*nii
                           IDERQ(IND)   = IDERQ(IND) + DZT*IWI(iw)
                        END DO
                     ENDIF
            
                     IF (R .GE. 1) THEN
                        IF (NOMU .EQ. 0) THEN
                           ind = r + p + ks
                        ELSE
                           ind = p + ks
                        ENDIF
                        R2 = R * R
         
                        IF (ALLOCATED(IWRKR)) THEN
                           IF(UBOUND(IWRKR,1) .NE. R) THEN
                              DEALLOCATE(IWRKR)
                              ALLOCATE(IWRKR(R))
                           ENDIF
                        ELSE
                           ALLOCATE(IWRKR(R))
                        ENDIF
                        
                        IF (ALLOCATED(IWORKR2)) THEN
                           IF(UBOUND(IWORKR2,1) .NE. R2) THEN
                              DEALLOCATE(IWORKR2)
                              ALLOCATE(IWORKR2(R2))
                           ENDIF
                        ELSE
                           ALLOCATE(IWORKR2(R2))
                        ENDIF
         
                        DO H = 1,R
                           ix  = k + (h-1)*nii
                           IF (IVSEP .EQ. 0) THEN
                              H2 = Q + (h-1)*NQ
                           ELSE
                              H2 = Q
                           ENDIF
                           IWORKR(H) = IBQ(H2)
                           IWRKR(H) = IXI(ix)
                        END DO
                        
                        CALL KMPY(IWRKR,IWORKR,IWORKR2,R,1,0,1,R)   
                        IF (IDIAG .EQ. 0 .AND. IVSEP .EQ. 0) THEN
                           CALL CHAMS(IWORKR2,IWORKR2,R,0,3)
                        ELSE
                           CALL CHAMS(IWORKR2,IWORKR2,R,0,2)
                        ENDIF
            
                        IH = 0
                        DO h  = 1,r
                           IF (IDIAG .EQ. 1 .OR. IVSEP .EQ. 1) THEN 
                              IH  = IH  + 1
                              ind = ind + 1
                              IDERQ(IND) = IDERQ(IND) + deriv*IWORKR2(IH) & 
                                           /DEXP(WTAU)
                           ELSE
                           
                              DO h2 = 1,h
                                 IH  = IH  + 1
                                 ind = ind + 1
                                 ! estimate sigma 
                                 IF (IFIN .NE. 0 .OR. H .NE. H2) THEN
                                    IDERQ(IND) = IDERQ(IND)+deriv*IWORKR2(IH) &
                                                 /DEXP(WTAU)
                                 ELSE
                                    ! estimate tau = ln(diagonal elements of sigma)
                                    IF (DERIV .LT. 0.0D0 .AND. IWORKR2(IH)  &
                                        .LT. 0.0D0) SIGN =  1.0D0
                                    IF (DERIV .LT. 0.0D0 .AND. IWORKR2(IH)  &
                                        .GE. 0.0D0) SIGN = -1.0D0
                                    IF (DERIV .GE. 0.0D0 .AND. IWORKR2(IH)  &
                                        .LT. 0.0D0) SIGN = -1.0D0
                                    IF (DERIV .GE. 0.0D0 .AND. IWORKR2(IH)  &
                                        .GE. 0.0D0) SIGN =  1.0D0
                                    IDERQ(IND) = IDERQ(IND) + SIGN *  &
                                        DEXP(DLOG(DABS(deriv)) +DLOG(CHOL(IH)) &
                                        +DLOG(DABS(IWORKR2(IH)))) / DEXP(WTAU)
                                 ENDIF
                              END DO
                           ENDIF
                        END DO ! h = 1,r
                     ELSE
                     ENDIF
         
                     !    WRITE(6,*)' before deriv0 and deriv1'
         
                     IF (prob .le. 0.0D0) THEN
                        deriv0 = 0.0d0
                        deriv1 = 0.0d0
                     ELSE
                        deriv0 =         (probp0/prob)  / DEXP(WTAU)
                        deriv1 =         (probp1/prob)  / DEXP(WTAU)
                     ENDIF
            
                     !    WRITE(6,*)' after deriv0 and deriv1'
            
                     IF (NGAM .gt. 0) THEN
                        IF (NOMU .EQ. 0) THEN
                           ic = p+r+rr+ks
                        ELSE
                           ic = p+rr+ks
                        ENDIF
            
                        CSTAT = 1.0D0
                        IF (ICEN .EQ. 1) then
                            if(IEV(K) .EQ. 0) CSTAT = 0.0D0
                        endif
                        
                        IF (J .GT. 2) IDERQ(IC+J-2)=IDERQ(IC+J-2)+deriv0*CSTAT 
                        
                        IF(J .GE. 2 .AND. J .LE. NGAM1+1) &
                            IDERQ(IC+J-1)=IDERQ(IC+J-1)+deriv1 
         
                        IF(KG .GT. 0) THEN
                           IF (NOMU .EQ. 0) THEN
                              ic0= p+r+rr+ngam1
                           ELSE
                              ic0= p+rr+ngam1
                           ENDIF
                           DO L=1,KG
                              ic = ic0+ (l-1)*ngam1
                              iw = k + (l-1)*nii
                              IF (J.GT.2) IDERQ(IC+J-2) =  &
                                   IDERQ(IC+J-2)+deriv0*CSTAT*IWI(iw)*RADD
                              IF (J.GE.2 .AND. J.LE.NGAM1+1) IDERQ(IC+J-1) =  &
                                   IDERQ(IC+J-1)+deriv1*IWI(iw)*RADD
                           END DO
                        ENDIF
                     ENDIF
         
                     catj = .FALSE.
         
                  ENDIF
               END DO
         
            END DO    ! DO k=1,nii 
      
            ILIK(Q) = DEXP(psum)
            hprob  = hprob + DEXP ( psum + DLOG(IAQ(Q)))
      
            DO L  = 1,npar
               IDERP(L) = IDERP(L) + IDERQ(L)*ILIK(Q)*IAQ(Q)  
            END DO
      
         END DO    ! DO q=1,nq 
   
         ! ** done with quadrature points

         IF (HPROB .LT. .1D-305) HPROB = .1D-305
         rlogl = rlogl + WTSUBI*DLOG(hprob)
   
         scal       = DEXP( 0.0d0 - DLOG(hprob))
         
         DO l   = 1,npar
            IDER(L)  = IDER(L) + IDERP(L)* DEXP(DLOG(WTSUBI) -DLOG(hprob))
            IDERP(L) = IDERP(L)*scal
         END DO
         
         IF (IPRIOR .EQ. 1) THEN
            DO q   = 1,nq
               IAQEMP(Q) = IAQEMP(Q) + DEXP( (DLOG(WTSUBI)) + &
                  (DLOG(ILIK(Q))) + (DLOG(IAQ(Q))) - DLOG(hprob))
            END DO
         ENDIF
   
         ! scal2 = (1.0d0 + hprob)*wtsubi
         ! CALL grmcv(cholIDER2,cholIDER2,IDERP,scal2,npar)
         CALL grmcv(cholIDER2,cholIDER2,IDERP,wtsubi,npar)
   
         ! ***************************************************************************
         ! write out the BAYES estimates at final iteration (IFIN=2or3) ! ONLY IF IRES=1 
   
         IF ((IFIN .GE. 2 .or. (it >= 4 .and. aquad .ne. 0))) then! .AND. R .GE. 1) THEN ! .AND. IRES .EQ. 1) THEN
            IH = 0
            
            ITHETA  = 0.0D0    ! Array assignment gets every element
            ITHETAV = 0.0D0    ! Array assignment gets every element
            
            DO Q = 1,NQ
               QMULT = ILIK(Q) * IAQ(Q)
               DO H = 1,IRT
                  H2 = Q + (h-1)*NQ
                  ITHETA(H) = ITHETA(H) + IBQ(H2)*QMULT
               END DO
            END DO
            
            DO H = 1,IRT
               ITHETA(H) = ITHETA(H)/HPROB
            END DO
   
            DO Q=1,NQ
               QMULT = ILIK(Q) * IAQ(Q)
               DO H = 1,IRT
                  H2 = Q + (h-1)*NQ
                  IWORKR(H) = IBQ(H2) - ITHETA(H)
               END DO
               CALL GRMCV(ITHETAV,ITHETAV,IWORKR,QMULT,IRT)
            END DO
   
            IH = 0
            DO H = 1,IRT
               DO H2 = 1,H
                  IH = IH+1
                  ITHETAV(IH)=ITHETAV(IH)/HPROB
               END DO
            END DO
            
            ebmean(i,:) = itheta
            ebvar(i,:) = ithetav

            !IF (IWT .EQ. 0) THEN
            !   WRITE(9,"(2I15)")IDI,NII
            !ELSEIF (IWT .EQ. 1) THEN
            !   WRITE(9,"(I15,F12.5,I15)")IDI,WTSUBI,NII
            !ENDIF
            !WRITE(9,"(5F15.6)")(ITHETA(H),H=1,IRT)
            !!WRITE(9,"(5F15.6)")(ITHETAV(HR),HR=1,IRTT)
   
         ENDIF
   
      END DO   ! end of subject loop:  DO I = 1,N

      ! ************************************************
      !  done with subjects                            
      ! ************************************************

      IF (ALLOCATED(ICOREC)) THEN
         IF(UBOUND(ICOREC,1) .NE. NPAR) THEN
            DEALLOCATE(ICOREC)
            ALLOCATE(ICOREC(NPAR))
         ENDIF
      ELSE
         ALLOCATE(ICOREC(NPAR))
      ENDIF

      !   ADD RIDGE TO DIAGONAL ELEMENTS OF DER2 PRIOR TO INVERSION
      !   (unless at final iteration itlast=1)

      IF(ITLAST.EQ.0)THEN
         DO KKK=1,NPAR
            KK=(KKK*(KKK+1))/2
            cholIDER2(KK)=cholIDER2(KK)+RIDGE*cholIDER2(KK)
         END DO
      ENDIF

      ! check to see if the matrix of second derivatives is positive-definite
      ! (nonpos = 1) or not positive definite (nonpos = 0)

      IF (ALLOCATED(ICHWRK)) THEN
         IF(UBOUND(ICHWRK,1) .NE. NPARR) THEN
            DEALLOCATE(ICHWRK)
            ALLOCATE(ICHWRK(NPARR))
         ENDIF
      ELSE
         ALLOCATE(ICHWRK(NPARR))
      ENDIF

! do the inversion of the information matrix via the Cholesky      
      CALL CHSKY(cholIDER2,ICHWRK,npar,NONPOS)
      CALL INVT(ICHWRK,NPAR,DET)
      CALL CHAMS(ICHWRK,IDER2S,NPAR,3,0)
      CALL GRAMT(IDER2S,cholIDER2,NPAR,NPAR)
!     CALL INVS(cholIDER2,npar,det,ICOREC,IER,.FALSE.)
      CALL MPYM(cholIDER2,IDER,ICOREC,npar,npar,1,0,1)

      ! DO STEP-HALVING FOR FIRST 10 ITERATIONS

      DO L = 1,NPAR
         ICOREC(L) = STEP*ICOREC(L)
      END DO

      BIGCOR = 0.0D0

      IF (r .ge. 1 .AND. NOMU .EQ. 0) THEN
         DO h = 1,r
            MU(H)    = MU(H) + ICOREC(H)
            IF (DABS(ICOREC(H)) .GT. BIGCOR) BIGCOR = DABS(ICOREC(H))
         END DO
      ENDIF   

      IF (p .ge. 1) THEN
         DO l = 1,p
            IF (NOMU .EQ. 0) THEN
               ind = l + r
            ELSE
               ind = l
            ENDIF
            ALPHA(L) = ALPHA(L) + ICOREC(IND)
            IF (DABS(ICOREC(IND)) .GT. BIGCOR) BIGCOR = DABS(ICOREC(IND))
         END DO
      ENDIF

! keep the scaling terms at their starting values for the first 10 iterations
      IF (it .gt. 10 .and. ks .ge. 1) THEN
         DO l = 1,ks
            IF (NOMU .EQ. 0) THEN
               ind = l + r + p
            ELSE
               ind = l + p
            ENDIF
            TAU(L) = TAU(L) + ICOREC(IND)
            IF (DABS(ICOREC(IND)) .GT. BIGCOR) BIGCOR = DABS(ICOREC(IND))
         END DO
      ENDIF

! keep the variance-covariance parameters at their starting values for the first 5 iterations
      IF (it .gt. 5 .and. r .ge. 1) THEN
         INDD = 1
         INDD2= 0
         
         DO h = 1,rr
            IF (NOMU .EQ. 0) THEN
               ind = r + p + ks + h 
            ELSE
               ind = p + ks + h
            ENDIF
            
            IF (H .EQ. INDD) THEN
               ! on the diagonal
               IF (IDIAG .EQ. 0 .AND. IVSEP .EQ. 0) THEN
                  INDD2 = INDD2 + 1
                  INDD  = INDD  + (INDD2 + 1)
               ELSE
                  INDD  = INDD  + 1
               ENDIF
               
               ! make sure that the first variance term is positive
               IF (IFIN .NE. 0) THEN
                  IF (H.EQ.1 .AND.  &
                     (0.0D0-ICOREC(IND) .GT. CHOL(H)))THEN
                     CHOL(H) = 0.10D0* DABS(CHOL(H) + ICOREC(IND))
                  ELSE
                     CHOL(H) = CHOL(H) + ICOREC(IND)
                  ENDIF

                  ! DON'T shift to estimation of tau if sigma gets small 
                  ! INCREASE THE RIDGE ONCE INSTEAD
                  !  IF(SIGMA(H) .LE. .1D0 .AND. NORI .EQ. 0) THEN
                  !     IFIN=0
                  !     NORI=NORI+1
                  !     RIDGE = RIDGE +.1D0
                  !     WRITE(2,"(///,1X,'==> Reparameterized Estimation Occurred')")
                  !  ENDIF
      
               ELSE
                  ISIGTAU(INDD2) = ISIGTAU(INDD2) + ICOREC(IND)
                  CHOL(H)  = DEXP(ISIGTAU(INDD2))

                  ! reduce number of random effects if sigma get too small
                  IF(CHOL(H) .LE. .000000001D0) ISIG = 1
               ENDIF
            ELSE
               ! off the diagonal
               CHOL(H)  = CHOL(H) + ICOREC(IND)
            ENDIF 
            
            IF (DABS(ICOREC(IND)) .GT. BIGCOR) BIGCOR = DABS(ICOREC(IND))
         END DO ! h = 1,rr
      ENDIF

      IF (ngam .gt. 0) THEN
         L = 0
         DO l1 = 1,KG+1 
            DO l2 = 1,ngam1 
               L = L + 1
               IF (NOMU .EQ. 0) THEN
                  ind = r + p + rr + ks + l
               ELSE
                  ind = p + rr + ks + l
               ENDIF
               
               ! make sure that gamma is increasing 
               ! (NOT FOR THE INTERACTIONS WITH THRESHOLDS)
               
               GAM(L) = GAM(L) + ICOREC(IND)
               IF (L1 .EQ. 1 .AND. L2 .GT. 1) then
                    if(GAM(L) .LE. GAM(L-1)) GAM(L) =  GAM(L-1) + .1D0
                end if
               IF (DABS(ICOREC(IND)) .GT. BIGCOR)  &
                   BIGCOR = DABS(ICOREC(IND))
            END DO
         END DO
      ENDIF

      ! PRINT OUT RESULTS FOR THIS ITERATION
      ! unless iteration was for computation of information matrix only
      ! (ITLAST = 1)

      IF (ITLAST .EQ. 1) THEN
         termin = 0
         CALL FREE_LOCALS()
         EXIT     ! escape from the loop
      ENDIF

      !WRITE(IUN,65)IT
      !65 FORMAT(/,1X,'**************',/,1X,'ITERATION ',I4,/, &
      !            1X,'**************',/)

      !WRITE(IUN,"(1X,'Log Likelihood   = ',F12.3,//)") rlogl

      ! save current rlogl to check at next iteration

      RLOGDIFF = RLOGL - RLOGLP
      RLOGLP   = RLOGL

      ! calculate versions of RLOGL 
      IF (IWT .EQ. 0) THEN
          SBCN = DBLE(N)
      ELSE
          SBCN = SUM2
      END IF
      AIC  = RLOGL - RNPAR
      SBC  = RLOGL - 0.5 * RNPAR * DLOG(DBLE(SBCN))
      DEV  = -2*RLOGL
      AICD = -2*(RLOGL-RNPAR)
      SBCD = -2*(RLOGL-0.5*RNPAR*DLOG(DBLE(SBCN)))

      !IF (r .ge. 1 .AND. NOMU .EQ. 0) THEN
      !   WRITE(IUN,"(1X,'mu         ',7F10.6)") (MU(H),H=1,R)
      !ENDIF
      
      !IF (P .GT. 0) THEN
      !   WRITE(IUN,"(1X,'alpha      ',7F10.6)") (ALPHA(L),L=1,P)
      !ENDIF
      
      !IF (KS .GT. 0) THEN
      !   WRITE(IUN,"(1X,'tau        ',7F10.6)") (TAU(L),L=1,KS)
      !ENDIF
      
      !IF (r .ge. 1) THEN
      !   WRITE(IUN,"(1X,'sigma      ',7F10.6)") (SIGMA(HR),HR=1,RR)
      !ENDIF
      
      !IF (ngam .gt. 0) THEN
      !   WRITE(IUN,"(1X,'gamma      ',7F10.6)") (GAM(L),l=1,ngam)
      !ENDIF
      
      !WRITE(IUN,"(1X,'corrections',7F10.6)") (ICOREC(L),L=1,NPAR)

      ! check to see if the log-likelihood doesn't increase
      ! and if so THEN increase the ridge by 0.1 each time 
!     IF(RLOGDIFF .LT. 0.0D0) RIDGE = RIDGE + 0.1D0

     ! determine if an iteration is bad and up the ridge
     ! take the ridge off after 20 good iterations
     IF (RLOGDIFF/RLOGLP > .000001) THEN
        RIDGEIT = 0
        RIDGE = RIDGE + .1D0
        IF (RIDGE .GT. RIDGEMAX) RIDGEMAX = RIDGE
     END IF
     IF (RLOGDIFF/RLOGLP <= .000001 .AND. RIDGEIT < 20) THEN
        RIDGEIT = RIDGEIT+1
     ELSE IF (RLOGDIFF/RLOGLP <= .000001 .AND. RIDGEIT >= 20) THEN
        RIDGE = 0.0D0
     END IF

      ! check to see if there are numerical problems and if so
      ! reduce the number of random-effects by 1
      !IF(IER .EQ. 1 .OR. NONPOS .EQ. 0 .OR. ISIG .EQ. 1) IRBAD =IRBAD+1
      !IF(IRBAD .EQ. 1) THEN
      !   WRITE(6,878)IT, ier, nonpos, isig
      !   WRITE(2,879)IT, ier, nonpos, isig
      !   878 FORMAT(///,1X,'==> WARNING!  Estimation Difficulties at Iteration', &
      !             I4,/,1x,'==> will proceed with one less random effect', &
      !                /,' ier = ',i4,' nonpos = ',i4,' isig = ',i4)
      !   879 FORMAT(///,1X,'==> WARNING!  Estimation Difficulties Occurred at Iteration', &
      !             I4,/,1x,'==> The model was fit with one less random effect than was requested', &
      !                /,' ier = ',i4,' nonpos = ',i4,' isig = ',i4)
      !ENDIF      

      ! change weights and nodes for empirical prior

      IF (IPRIOR .EQ. 1) THEN
         SUMW=0.0D0
         
         DO Q = 1,NQ
            IAQ(Q) = IAQEMP(Q) / DBLE(N)
            SUMW  = SUMW   + IAQ(Q)
         END DO
         DO Q = 1,NQ
            IAQ(Q) = IAQ(Q)/SUMW
         END DO
         !WRITE(IUN,"(1X,'quad weight',7F10.6)") (IAQ(L),L=1,NQ)

      ENDIF

      ! CHECK IF CONVERGENCE HAS BEEN REACHED
      ! IF NOT RETURN TO ITERATIONS 
      ! IF IT HAS PROCEED TO STANDARD ERROR CALCULATIONS
      ! AND PRINT OUT FINAL RESULTS

      IGO = 0
      IF (BIGCOR .GE. CONV) THEN
         IGO = 1
      ELSE
         IFIN = IFIN + 2
         IF (RIDGE .GT. 0.0) ITLAST = 1
         IF (IFIN .EQ. 2)    ITLAST = 1
         IF (IFIN .GE. 3 .AND. R .GE. 1) ITLAST = 1 !kkim32 .AND. IRES .EQ. 1) ITLAST = 1
         IF (IFIN .GE. 3 .AND. R .EQ. 0) IGO = 0 !kkim32 .AND. IRES .EQ. 0)
      ENDIF
      
      ! Signal caller that there's more to do
      termin = 1
   ELSE
      
      ! Signal caller that we're done iterations
      termin = 0
      CALL FREE_LOCALS()

   ENDIF

   END DO

    allocate(sstar(rr*rr))
    allocate(sstar2(rr*rr))
    allocate(sstar2t(rr*rr))
    allocate(work(npar*npar))
    allocate(asstar2(npar*npar))
    allocate(asstar2t(npar*npar))

    if(CHOLe .ne. 1) then
        if(IVSEP .ne. 1 .and. IDIAG .ne. 1) then
            call getSStar(chol,r,rr,sstar)
            call scm(sstar,2.0D0,sstar2,rr,rr,0)
            call trp(sstar2,sstar2t,rr,rr)
            call mpym(sstar,chol,sigma,rr,rr,0,0,1)
        else
                sstar2 = 0
                do i=1,r
                    sstar2(i+(i-1)*r) = chol(i)*2
                    sigma(i) = chol(i)*chol(i)
                end do
                sstar2t = sstar2
        end if

        pabove = p+ks+r*(1-nomu)
        pbelow = ngam
        asstar2 = 0
        asstar2t = 0
        asCount = 1
        sCount = 1
        do i=1,npar
            do j=1,npar
                if(i > pabove .and. i <= npar - pbelow) then
                    if(j > pabove .and. j <= npar - pbelow) then
                        asstar2(asCount) = sstar2(sCount)
                        asstar2t(asCount) = sstar2t(sCount)
                        sCount = sCount + 1
                    end if
                else if(i .eq. j) then
                    asstar2(asCount) = 1
                    asstar2t(asCount) = 1
                end if
                asCount = asCount + 1
            end do
        end do
        
        call mpym(asstar2,cholider2,work,npar,npar,0,1,npar)
        call mpym(work,asstar2t,ider2,npar,npar,0,0,npar)
    else
        call chams(cholider2, ider2, npar, 1, 0)
        sigma=chol
    end if
    
   !ALLOCATE(SE(NPAR))
   IC = 0
   DO I = 1,NPAR
      DO J = 1,npar
         IC = IC + 1
         IF (I .EQ. J) THEN
            SE(I)   = DSQRT(ider2(IC))
         ENDIF
      END DO
   END DO
   
    !call scm(asstar2,2.0D0,adjvar,npar,npar,0)

   !CALL Empirical_Prior(N,R,IPRIOR,IRT,IRTT,NQ,NQR,NQRR,&
   !                          ITHETA,ITHETAV,IAQ,IBQ,IWORKR)

   DEALLOCATE( cholIDER2, chol )!SE, 
   DEALLOCATE( IDER,IDER2S,ISIGTAU,IWG )
   DEALLOCATE( IAQ, IBQ, IAQ1, IAQ2, IAQEMP, IBQ1 )
   DEALLOCATE( ITHETA, ITHETAV )
   deallocate(iaq0, ibq0)
   DEALLOCATE( IWORKR )
   deallocate(sstar, sstar2, sstar2t, asstar2, asstar2t, work)
    
   CONTAINS

! ************************************************
!               **********************                         
!               *  SUBROUTINE FREE_LOCALS()
!               **********************                         
!                                                                       
! Deallocate locally used arrays prior to exiting the routine
! on the final iteration.  It is in a routine because there
! are two exit points from the routine and this must be called
! from each.  Note that the locals MUST be explicitly deallocated
! because otherwise the next session will find they are already
! allocated and will generate an error when it attempts to 
! allocate them.  Tony Gray 3/7/01
!
! CALL FREE_LOCALS()
!                                                                       
! ************************************************
   SUBROUTINE FREE_LOCALS()
      !IF(ALLOCATED(TEMPLABEL)) DEALLOCATE(TEMPLABEL)
      IF(ALLOCATED(IWRKR)) THEN
         DEALLOCATE(IWRKR)
         DEALLOCATE(IWORKR2)
      ENDIF
      IF (ALLOCATED(IDERP)) DEALLOCATE(IDERP)
      IF (ALLOCATED(IDERQ)) THEN
         DEALLOCATE(IDERQ)
         DEALLOCATE(ILIK)
      ENDIF
      IF (ALLOCATED(IYI)) DEALLOCATE(IYI)
      IF (ALLOCATED(IXI)) DEALLOCATE(IXI)
      IF (ALLOCATED(IWI)) DEALLOCATE(IWI)
      IF (ALLOCATED(IEV)) DEALLOCATE(IEV)
      IF (ALLOCATED(ICOREC)) DEALLOCATE(ICOREC)
      IF (ALLOCATED(ICHWRK)) DEALLOCATE(ICHWRK)
   END SUBROUTINE FREE_LOCALS

 !  RETURN

END SUBROUTINE MIXORDEST

!*********************************************************************
SUBROUTINE Empirical_Prior(N,R,IPRIOR,IRT,IRTT,NQ,NQR,NQRR,&
                             ITHETA,ITHETAV,IAQ,IBQ,IWORKR)

   USE INTERFACE_MPDSD
   use interface_grmcv
   IMPLICIT NONE
   INTEGER, INTENT(IN)                            :: N,R,IPRIOR
   INTEGER, INTENT(IN)                            :: IRT,IRTT,NQR,NQRR,NQ
   REAL (KIND=8), INTENT(INOUT), DIMENSION(IRT)   :: ITHETA
   REAL (KIND=8), INTENT(INOUT), DIMENSION(IRTT)  :: ITHETAV
   REAL (KIND=8), INTENT(INOUT), DIMENSION(R)     :: IWORKR
   REAL (KIND=8), INTENT(IN), DIMENSION(NQR)      :: IAQ
   REAL (KIND=8), INTENT(IN), DIMENSION(NQRR)     :: IBQ
   ! Local variables:
   INTEGER                                        :: IH, H, H2, Q !HR,
   REAL (KIND=8)                                  :: QMULT
   REAL (KIND=8), ALLOCATABLE, DIMENSION(:)       :: ITHETAV2

   ALLOCATE(ITHETAV2(IRTT))


   ! print out the empirical prior

   IF (IPRIOR .EQ. 1) THEN

      !WRITE(2,656)
      !656 FORMAT(///,1x,'--------------------------------', &
      !             /,1X,'* Empirical Prior Distribution *', &
      !             /,1x,'--------------------------------')

      !IF (r .eq. 1) THEN
      !   WRITE(2,657)
      !   657 FORMAT(/,1X,' #',5X,'  Weight',4x,'     Node', &
      !              /,1x,'--',5x,'--------',4x,'---------')
      !ELSEIF (r .gt. 1) THEN
      !   WRITE(2,6657)
      !   6657 FORMAT(/,1X,' #',5X,'  Weight',4x,'     Node (r = 1, .. , R)', &
      !               /,1x,'--',5x,'--------',4x,'-------------------------')
      !ENDIF

      ITHETAV  = 0.0D0    ! Array assignment gets every element
      ITHETAV2 = 0.0D0    ! Array assignment gets every element
      
      DO Q = 1,NQ
         DO H = 1,IRT
            H2 = Q + (h-1)*NQ
            ITHETA(H) = ITHETA(H) + IBQ(H2)*IAQ(Q)
         END DO
      END DO
      
      CALL GRMCV(ITHETAV2,ITHETAV2,ITHETA,DBLE(N),IRT)

      DO Q=1,NQ
         QMULT = DBLE(N) * IAQ(Q)
         DO H = 1,IRT
            H2 = Q + (h-1)*NQ
            IWORKR(H) = IBQ(H2)  
         END DO
         !WRITE(2,6659) Q,IAQ(Q),(IWORKR(H2),H2=1,IRT)
         !6659 FORMAT(1x,I2,5X,F8.5,5X,8F9.5)
         CALL GRMCV(ITHETAV,ITHETAV,IWORKR,QMULT,IRT)
      END DO

      IH = 0
      DO H = 1,IRT
         DO H2 = 1,H
            IH = IH+1
            ITHETAV(IH)=(ITHETAV(IH)-ITHETAV2(IH)) /DBLE(N-1)
            IF (H.EQ.H2) THEN
               IWORKR(H) = 1.0D0 / DSQRT(ITHETAV(IH))
            ENDIF
         END DO
      END DO

      !WRITE(2,"(/,1X,'Empirical Prior Mean Vector')")
      !WRITE(2,3388)(ITHETA(H),H=1,IRT)
      !WRITE(2,667)
      !667 FORMAT(/,1X,'Empirical Prior Variance-Covariance Matrix (in packed form)')
      !WRITE(2,3388)(ITHETAV(HR),HR=1,IRTT)
      IF (IRT .GT.1) THEN
         CALL MPDSD(IWORKR,ITHETAV,ITHETAV,IRT)
         !WRITE(2,668)
         !668 FORMAT(/,1X,'Empirical Prior Correlation matrix (in packed form)')
         !WRITE(2,3388)(ITHETAV(HR),HR=1,IRTT)
      ENDIF
       
      !3388 FORMAT(5F15.6)

   ENDIF

   DEALLOCATE( ITHETAV2 )

END SUBROUTINE Empirical_Prior

!*********************************************************************

!  SUBROUTINTE STARTV2 (MU1,SIGMA,GAMMA,R,MAXJ,NGAM,CATFQ, 
!              DENUM,WA,NFN,ICEN,KG,RADD,IDIAG,IVSEP)     
!                                                         
!  GET STARTING VALUES FOR INTERCEPT, THRESHOLDS, AND     
!  VARIANCE-COVARIANCE MATRIX OF RANDOM EFFECTS           
!                                                         
!  Parameters Sent                                        
!  R      = Number of Random Effects                      
!  MAXJ   = Number of Ordered Categories                  
!  NGAM   = Number of thresholds to estimate              
!  CATFQ  = MAXJ Category Frequencies                     
!  DENUM  = Total Number of Level-1 observations          
!  WA     = Predicted value of Y minus the intercept      
!  NFN    = 0 for Probit                                  
!         = 1 for Logistic                                
!         = 2 for Complementary Log-Log                   
!         = 3 for Log-Log                                 
!  RADD   = 1 add alpha and mu to thresholds              
!           (covariates and mean of random effects)       
!         =-1 subtract beta and mu from thresholds        
!  IDIAG  = 0 for correlated random effects               
!         = 1 for independent random effects              
!  IVSEP  = 0 R random effects aren't indicator variables 
!         = 1 R random effects are indicator variables    
!           (between-subjects indicator variables)        
!                                                         
!  Parameters Returned                                    
!  MU1    = Starting value for intercept                  
!  GAMMA  = NGAM Starting values for Thresholds           
!  SIGMA  = (R*(R+1))/2 Starting values for Variance      
!           Covariance matrix of random effects           
!                                                         
!*********************************************************************

SUBROUTINE STARTV2(MU1,SIGMA,GAMMA,R,MAXJ,NGAM,CATFQ,DENUM,WA, &
                      NFN,ICEN,KG,RADD,IDIAG,IVSEP)
    implicit none
    real (kind=8),intent(in)::catfq(:),wa,denum,radd
    real(kind=8),intent(out)::mu1,gamma(:),sigma(:)
    integer(kind=4),intent(in)::r,maxj,ngam,nfn,icen,kg,idiag,ivsep
    
    real(kind=8)::pi,cumsum,cumprob,cumodds,rlncumod,tempparm
    integer(kind=4)::j,ngam1,kgn,kgm,i,ir,i2

   DATA PI/3.141592654D0/

   cumsum = 0.0D0
   DO j=1,maxj-1
      cumsum   = cumsum + CATFQ(J)
      cumprob  = cumsum / DENUM
      cumodds  = cumsum / (DENUM - cumsum)
      Rlncumod = DLOG(cumodds)
      IF (nfn .eq. 0) THEN
         tempparm = 0.625 * (Rlncumod + wa)
      ELSEIF (nfn .eq. 1) THEN
         tempparm = Rlncumod + wa
      ELSEIF (nfn .eq. 2) THEN
         tempparm = DLOG(0.0D0 - DLOG(1.0d0 - cumprob)) + wa
      ELSEIF (nfn .eq. 3) THEN
         tempparm = DLOG(0.0D0 - DLOG(cumprob)) + wa
      ELSE
      ENDIF
      IF (j .EQ. 1) THEN
         MU1 = RADD*tempparm
      ELSE
         GAMMA(j-1) = tempparm - RADD*MU1
      ENDIF
   END DO

   ! FOR RIGHT CENSORING AN EXTRA GAMMA NEEDS TO BE ESTIMATED
   !  arbitrarily assume half of the last category is censored 

   NGAM1 = MAXJ-2
   IF (ICEN .EQ. 1) THEN
      NGAM1 = MAXJ-1
      cumprob  = cumprob + (0.5d0 * (1.0d0 - cumprob))
      cumodds  = cumprob / (1.0d0 - cumprob)
      Rlncumod = DLOG(cumodds)
      IF (nfn .eq. 0) THEN
         tempparm = 0.625 * (Rlncumod + wa)
      ELSEIF (nfn .eq. 1) THEN
         tempparm = Rlncumod + wa
      ELSEIF (nfn .eq. 2) THEN
         tempparm = DLOG(0.0D0 - DLOG(1.0d0 - cumprob)) + wa
      ELSEIF (nfn .eq. 3) THEN
         tempparm = DLOG(0.0D0 - DLOG(cumprob)) + wa
      ELSE
      ENDIF
      GAMMA(MAXJ-1) = tempparm - RADD*MU1
   ELSE 
   ENDIF

   ! FOR interactions with GAMMA - assign the same GAMMAs for all

   IF (KG .GT. 0) THEN
      DO KGN = NGAM1+1,NGAM
         KGM = MOD(KGN,NGAM1)  
         IF (KGM .EQ. 0) KGM=NGAM1
         GAMMA(KGN) = GAMMA(KGM)*0
      END DO
   ENDIF


   IF (R .EQ. 1) THEN
      SIGMA(1) = .31623d0
      IF (nfn .EQ. 1) SIGMA(1) = SIGMA(1) * DSQRT(pi*pi/3.0D0)
      IF (nfn .GE. 2) SIGMA(1) = SIGMA(1) * DSQRT(pi*pi/6.0D0)
   ELSEIF (R .GT. 1 .AND. IDIAG .EQ. 0 .AND. IVSEP .EQ. 0) THEN
      IR = 0
      DO I  = 1,R
         DO I2 = 1,I
            IR = IR + 1
            IF (I2 .EQ. I) THEN
               IF (I .EQ. 1) THEN
                  SIGMA(IR) = 1.0d0
               ELSE
                  SIGMA(IR) = 0.5d0
               ENDIF
            ELSE
               SIGMA(IR) = 0.0d0
            ENDIF
            IF (nfn .EQ. 1) SIGMA(IR) = SIGMA(IR)*DSQRT(pi*pi/3.0D0)
            IF (nfn .GE. 2) SIGMA(IR) = SIGMA(IR)*DSQRT(pi*pi/6.0D0)
         END DO
      END DO
   ELSEIF (R .GT. 1 .AND. (IDIAG .EQ. 1 .OR. IVSEP .EQ. 1)) THEN
      IR = 0
      DO I  = 1,R
         IR = IR + 1
         IF (I .EQ. 1) THEN
            SIGMA(IR) = 1.0d0
         ELSE
            SIGMA(IR) = 1.0d0
         ENDIF
         IF (nfn .EQ. 1) SIGMA(IR) = SIGMA(IR)*DSQRT(pi*pi/3.0D0)
         IF (nfn .GE. 2) SIGMA(IR) = SIGMA(IR)*DSQRT(pi*pi/6.0D0)
      END DO
   ENDIF

   RETURN
END SUBROUTINE STARTV2

!*********************************************************************
!               **********************
!               *  SUBROUTINE unique(IDs,n,ID)
!               **********************
!
!*********************************************************************

subroutine unique(IDs,n,ID)
!
! Purpose:
! To sort and remove the duplicate ID numbers
!
   IMPLICIT NONE
   INTEGER,INTENT(IN) :: n
   INTEGER,INTENT(IN),DIMENSION(n)  :: IDs
   INTEGER,INTENT(OUT),DIMENSION(n) :: ID
   INTEGER:: i,j,k,n1,swap

   ID(:) = 0

   !remove the duplicate ID
   j=1
   ID(j)=IDs(j)
   do i=1,n
      if (ID(j)==IDs(i)) then
         continue
      else
         j=j+1
         ID(j)=IDs(i)
      end if
   end do

   !sort the ID
   n1=COUNT(ID>0)
   do i=1,n1-1

      !find the location of minMUm value
       k=i
       do j=i+1,n1
          IF(ID(j)<ID(k)) THEN
            k=j
          END if
       end do

      !swap the minMUm value
      IF(i/=k) THEN
        swap=ID(i)
        ID(i)=ID(k)
        ID(k)=swap
      END if
   end do

end subroutine unique

!*********************************************************************
!                        **********************                         
!                        *  SUBROUTINE CHAMS  *                         
!                        **********************                         
!                                                                       
!         CHANGE MODE OF STORAGE OF A SQUARE MATRIX FROM ONE 
!         PACKED FORM TO ANOTHER
!                                                                       
!         CALL CHAMS (A,B,N,MSA,MSB)                                    
!                                                                       
!         A .......... INPUT MATRIX, N BY N                             
!         B .......... OUTPUT MATRIX, N BY N                            
!         N .......... NUMBER OF ROWS IN A, EQUAL TO NUMBER OF COLUMNS  
!         MSA ........ STORAGE MODE OF A, 0..4                               
!         MSB ........ DESIRED STORAGE MODE OF B, 0..4, UNEQUAL TO MSA        
!                                                                       
!     Storage modes: 0 SQUARE, 1 PACKED SYMMETRIC, 2 DIAGONAL, 
!                    3 PACKED LOWER TRIANGLE
!                    4 PACKED UPPER TRIANGLE
!
! About Matrix storage modes:
! 
! Matrices are stored in one of three formats in memory:
! Square, diagonal, or packed.  For matrices with some symmetry,
! (upper or lower triangular, symmetric, or diagonal), you don't
! have to store every element because some of the elements are known
! by virtue of the symmetry.  This routine handles conversion between
! the different storage modes, taking advantage of the symmetry to 
! fill in the missing elements.  
!
! 0 - Square matrix, every element included, stored in row major order.
! 1 - Packed symmetric - Mat(I,J) = Mat(J,I), so only the upper triangle
!     is actually stored, and the elements are copied across the diagonal
!     on expansion.
! 2 - Diagonal: linear array of N elements, representing Mat(I,I), where
!     all nondiagonal elements are 0
! 3 - Packed lower triangle - Mat(I,J) = 0 for I < J, others are stored 
!     as an upper triangle (like packed symmetric) and are transposed 
!     on expansion.  Upper triangle is zeroed on expansion.
! 4 - Packed upper triangle - Mat(I,J) = 0 for I > J, and is stored as
!     a packed array.  On expansion no transpose is required, but lower
!     triangle is zeroed. 
!
! The 'packed' formats (1,3,4) are all stored in the same way, as an 
! upper triangle contiguously recorded in memory, so for 1 and 4 you get
! elements (1,1) (1,2) (2,2) (1,3) (2,3) (3,3) (1,4) (2,4) (3,4) (4,4) etc.
! and for 3 you get 
!          (1,1) (2,1) (2,2) (3,1) (3,2) (3,3) (4,1) (4,2) (4,3) (4,4) etc.
! though to pack and unpack type 3, it is first transposed and then handled
! like an upper triangle.  1/2/01 Tony Gray
!
! Modified 1/30/01 to always use array indices within bounds to -chk 
! option can be enabled in compiler.  TG
!*********************************************************************

SUBROUTINE CHAMS(A,B,N,MSA,MSB)   

   IMPLICIT NONE
   INTEGER, INTENT(IN) :: N, MSA, MSB
   REAL(KIND=8), DIMENSION(N,N), INTENT(INOUT) :: A,B
                                
   INTEGER :: MMS, K, J, I, L, JR, M, LL, IR
   
   ! To select among the conversions, take 5* the source format
   ! and add the destination format.  Some of the conversions take
   ! more than one step, which is why there's a loop.  For the 
   ! ones that take more than one step, the select variable is 
   ! modified to give the next step in the process
   MMS=5*MSA+MSB
   
   DO  ! keep looping till all conversions are done
   
      SELECT CASE (MMS)
      
      CASE (1,4) ! SQUARE TO PACKED UPPER TRIANGLE : 0->1, 0->4
         K = 0                                                             
         DO J=1,N                                                         
            DO I=1,J              
               K = K + 1         
               ! B(K,1) = A(I,J)  
               B(MOD(K-1,N) + 1,(K-1)/N +1) = A(I,J)  
            END DO
         END DO
         EXIT
         
      CASE (2)   ! SQUARE TO DIAGONAL  : 0->2
         DO J=1,N                                                      
            B(J,1) = A(J,J) 
         END DO
         EXIT
      
      CASE (3)   ! first step of SQUARE TO PACKED LOWER TRIANGLE : 0->3
                 ! or the last step of 3 -> 0
         DO J=2,N                                                        
            DO I=1,J-1
               IF(MSA.EQ.3) THEN
                  ! complete 3-> 0 by copying upper tri to lower and 
                  ! zeroing the upper.
                  B(J,I)=B(I,J)       
                  B(I,J)=0.0D0       
               ELSE
                  ! start 0-> 3 by copying lower tri to upper and 
                  ! zeroing the lower.
                  A(I,J)=A(J,I)     
                  A(J,I)=0.0D0     
               ENDIF
            END DO
         END DO
         
         IF (MSA .LT. 3) THEN
            MMS = 1  ! Finish 0-> 3 by packing upper triangle into B
         ELSEIF (MSA .EQ. 3) THEN
            EXIT
         ELSE
            MMS = -1 ! Illegal case, error
         ENDIF
      
      CASE (5)   ! SYMMETRIC TO SQUARE : 1->0
         L = N + 1                                                         
         ! Take a triangular array stored in A() w/o its empty elements
         ! and unpack it into B() and then copy elements across the 
         ! diagonal to make B() symmetric NxN.  TG 
         K = (N*(N+1))/2 + 1 ! count of elements in triang. less one 
         DO JR=N,1,-1                                                      
            DO IR=JR,1,-1          
               K = K - 1            
               ! B(IR,JR) = A(K,1)     
               B(IR,JR) = A(MOD(K-1,N) + 1,(K-1)/N +1)
               ! IF(ir > N .or. jr > n .OR. K > N) then
               !    write(6,indices) 
               ! END IF
            END DO
         END DO
         
         ! Now copy across the diagonal to make a complete array
         DO J=2,N                                                      
            DO I=1,J                                                      
               B(J,I) = B(I,J) 
            END DO
         END DO
         EXIT
         
      CASE (10)  ! DIAGONAL TO SQUARE : 2->0
         DO J=1,N                                                      
            DO I=1,N                                                      
               B(I,J) = 0.0D0         
            END DO
         END DO
         DO J=1,N                                                      
            B(J,J) = A(J,1)          
         END DO
         
         EXIT
      
      CASE (7,17,22) ! SYMMETRIC TO DIAGONAL: 1->2, 3->2, 4->2 
         DO J=1,N                                                      
            K = J*(J+1)/2
            B(J,1) = A(MOD(K-1,N) + 1,(K-1)/N +1)
         END DO
         EXIT
      
      CASE (11,13,14) ! DIAGONAL TO SYMMETRIC : 2->1, 2->3, 2->4
         L = N + 1                                                         
         K = (N*L)/2                                                       
         DO J=1,N                                                      
            M = N + 1 - J           
            L = L - 1           
            !B(K,1) = A(L,1)    
            B(MOD(K-1,N) + 1,(K-1)/N +1) = A(L,1)    
            K = K - M         
         END DO
         L = 2                                                             
         DO J=2,N                                                      
            LL = L + J - 2   
            DO I=L,LL                                                     
               !B(I,1) = 0.0D0 
               B(MOD(I-1,N) + 1,(I-1)/N +1) = 0.0D0 
            END DO
            L = L + J        
         END DO
     
         EXIT
      
      CASE (15,20) ! LOWER/UPPER TRUE TRIANGLE TO SQUARE : 3->0, 4->0
         L = N + 1                                                         
         K = (L*N)/2 + 1                                                   
         DO JR=N,1,-1
            DO IR=JR,1,-1
               K = K - 1      
               !B(IR,JR) = A(K,1) 
               B(IR,JR) = A(MOD(K-1,N) + 1,(K-1)/N +1)
            END DO
         END DO
         
         DO J=2,N                                                      
            L = J - 1           
            DO I=1,L                                                      
               B(J,I) = 0.0D0  
            END DO
         END DO
         IF(MSA.EQ.3) THEN
            MMS = 3 ! If upper true triangle, transpose to lower
         ELSE
            EXIT
         ENDIF
      
      CASE DEFAULT
         !ERROR MESSAGE FROM SUBROUTINE CHAMS: ' // &
         !       'ILLICIT COMBINATION OF STORAGE MODES
      END SELECT
   END DO
   
END SUBROUTINE CHAMS

!*********************************************************************
!                  **********************                         
!                  *  SUBROUTINE  CHSKY *                         
!                  **********************                         
!   CALL CHSKY(A,B,N,NONPOS)                                      
!                                                                 
!   A .......... INPUT MATRIX, N BY N, SYMMETRIC (MSA=1)          
!   B .......... OUTPUT MATRIX, N BY N, CHOLESKY FACTOR, TRUE     
!                LOWER TRIANGULAR (MSB=3)                         
!   N .......... NUMBER OF ROWS IN A, EQUAL TO NUMBER OF COLUMNS  
!   NONPOS ..... OUTPUT VARIABLE, EQUALS  1  IF  A  IS POSITIVE-  
!                DEFINITE, EQUALS  0  OTHERWISE                   
!                                                                       
!*********************************************************************

SUBROUTINE CHSKY(A,B,N,NONPOS)
    implicit none
    real(kind=8),intent(in)::a(:)
    real(kind=8),intent(out)::b(:)
    integer(kind=4),intent(in)::n
    integer(kind=4),intent(out)::nonpos

    real(kind=8)::x,y
    integer(kind=4)::n1,kc,ifir,j,i,ii,jc,ic,k
   
   IF(A(1).LT.1.D-30) THEN
      NONPOS=0                                                          
   ELSE
      NONPOS=1                                                          
      X=DSQRT(A(1))                                                     
      B(1)=X                                                            
     
      IF(N .GT. 1) THEN
         N1=N-1          
         KC=1           
         IFIR=1        
         DO J=1,N1    
            KC=KC+J  
            B(KC)=A(KC)/X   
         END DO
         
         DO I=1,N1         
            
            IFIR=IFIR+I   
            KC=IFIR      
            X=0.0D0     
            
            DO J=1,I                                                      
               X=X+B(KC)*B(KC)    
               KC=KC+1 
            END DO
            
            X = A(KC) - X              
            
            IF(X.LT.1.D-30) THEN
               NONPOS=0               
               EXIT
            END IF
            
            X=DSQRT(X)             
            B(KC)=X               
            II=I+1               
            
            IF (II.EQ.N) EXIT
            
            JC=IFIR             
            
            DO J=II,N1                                                    
               JC=JC+J          
               IC=JC           
               KC=IFIR        
               Y=0.0D0       
               
               DO K=1,I    
                  Y=Y+B(IC)*B(KC)    
                  KC=KC+1           
                  IC=IC+1          
               END DO
               
               B(IC)=(A(IC)-Y)/X  
            END DO
         END DO
      END IF
   END IF
   
END SUBROUTINE CHSKY

!*********************************************************************
!                                                                       
!                  **********************                         
!                  *  SUBROUTINE   GEN  *                         
!                  **********************                         
!                                                                       
!   GENERATE A MATRIX ALL STORED ELEMENTS OF WHICH ARE EQUAL TO A 
!   SPECIFIED CONSTANT                                            
!                                                                 
!   CALL GEN (A,X,M,N,MS)                                         
!                                                                 
!   A .......... OUTPUT MATRIX, M BY N                            
!   X .......... INPUT CONSTANT OR VARIABLE, SPECIFIES VALUE FOR  
!                ELEMENTS OF A                                    
!   M .......... NUMBER OF ROWS IN A                              
!   N .......... NUMBER OF COLUMNS IN A                           
!   MS ......... DESIRED STORAGE MODE OF A                        
!                                                                       
!*********************************************************************

SUBROUTINE GEN (A,C,M,N,MS)
    implicit none
    integer (kind=4), intent(in) :: m,n,ms
    REAL (KIND=8), INTENT(OUT) :: a(:)
    real(kind=8), intent(in) :: c
    integer(kind=4):: L
            
  
   SELECT CASE (MS)
    CASE (0)
          L=M*N                                                             
    CASE (1,3,4)
          L=(N*(N+1))/2                                                     
    CASE (2)
          L=N                                                               
   END SELECT
   
   a(1:L) = C
END SUBROUTINE GEN

!*********************************************************************
!                                                                       
!                  **********************                         
!                  *  SUBROUTINE GRMCV  *                         
!                  **********************                         
!                                                                 
!   CALCULATE THE MATRIX PRODUCT OF A VECTOR, MULTIPLY IT BY A    
!   SCALAR, AND ADD IT TO A SYMMETRIC (MS=1) MATRIX ALREADY IN    
!   MEMORY                                                        
!                                                                 
!   CALL GRMCV (A,B,X,C,N)                                        
!                                                                 
!   A .......... INPUT MATRIX, N BY N, SYMMETRIC (MSA=1)          
!   B .......... OUTPUT MATRIX, N BY N, SYMMETRIC (MSB=1), RESULT 
!                OF ADDITION                                      
!   X .......... INPUT VECTOR OF LENGTH N                         
!   C .......... INPUT VARIABLE OR CONSTANT (SCALAR)              
!   N .......... NUMBER OF ROWS IN A, EQUAL TO NUMBER OF COLUMNS, 
!                EQUAL TO LENGTH OF X                             
!                                                                       
!*********************************************************************

SUBROUTINE GRMCV(A,B,X,C,N)                                       
   implicit none
   real (kind=8), intent(out) :: b(:)
   real (kind=8), intent(in) :: a(:),x(:),c
   integer(kind=4), intent(in) :: n
   
   integer(kind=4) :: ic,i,j
   IC=0                                                              
   DO I=1,N                                                      
      DO J=1,I                                                      
         IC=IC+1                                                           
         B(IC)=A(IC)+C*X(I)*X(J)                                           
      END DO
   END DO
END SUBROUTINE GRMCV

!*********************************************************************
!                                                                 
!                  **********************                         
!                  *  SUBROUTINE GRMMT  *                         
!                  **********************                         
!                                                                 
!   OBTAIN THE GRAMIAN MATRIX OF GENERALIZED PRODUCTS OF COLUMN   
!   VECTORS OF A SPECIFIED MATRIX BY POST- AND PREMULTIPLYING A   
!   METRIC MATRIX (SYMMETRIC OR DIAGONAL, MS=1,2) BY THE MATRIX   
!   AND ITS TRANSPOSE
!                                                                 
!   CALL GRMMT (A,B,C,M,N,MSB,W)                                  
!                                                                 
!   A .......... INPUT MATRIX, M BY N, GENERAL RECTANGULAR (MSA=0)
!   B .......... INPUT MATRIX, M BY M, METRIC MATRIX, SYMMETRIC OR
!                DIAGONAL                                         
!   C .......... OUTPUT MATRIX, N BY N, GENERALIZED GRAMIAN,      
!                SYMMETRIC (MSC=1)                                
!   M .......... NUMBER OF ROWS IN A                              
!   N .......... NUMBER OF COLUMNS IN A                           
!   MSB ........ STORAGE MODE OF B                                
!   W .......... WORKING VECTOR OF LENGTH M                       
!                                                                 
!*********************************************************************

SUBROUTINE GRMMT(A,B,C,M,N,MSB,W)    
    implicit none
    integer(kind=4), intent(in) :: m,n,msb
    real(kind=8), intent(in) :: a(m,n),b(:)
    real(kind=8), intent(out) :: c(:),w(:)
    
    real(kind=8) :: x
    integer(kind=4) :: k,i,ic,j,kc,kk,l

   IF (MSB .GT. 1) THEN
   
      IC=0                                                              
      DO I=1,N                                                      
         DO J=1,I                                                      
            X=0.0D0          
            DO K=1,M        
               X=X+A(K,I)*A(K,J)*B(K) 
            END DO
            IC=IC+1     
            C(IC)=X    
         END DO
      END DO
   
   ELSEIF (MSB .EQ. 1) THEN
   
      KC=0                                                              
      DO I=1,N                                                      
         KK=0                                                              
         
         DO K=1,M                                                      
            X=0.0D0               
            IC=KK                
            DO L=1,K                                                      
               IC=IC+1          
               X=X+A(L,I)*B(IC)
            END DO
            IF (K .NE. M) THEN
               J=K+1          
               IC=IC+K       
               DO L=J,M     
                  X=X+A(L,I)*B(IC)   
                  IC=IC+L           
               END DO
            END IF
            W(K)=X                 
            KK=KK+K               
         END DO
      
         DO K=1,I                 
            KC=KC+1             
            X=0.0D0            
            DO L=1,M          
               X=X+A(L,K)*W(L)  
            END DO
            C(KC)=X            
         END DO
         
      END DO
   
   ELSE  ! MSB = 0
      !ERROR MESSAGE FROM SUBROUTINE GRMMT: ' // &
      !       'METRIC IS GENERAL RECTANGULAR
   ENDIF
   
END SUBROUTINE GRMMT

!*********************************************************************
!                                                                 
!                  **********************                         
!                  *  SUBROUTINE   ADDM *                         
!                  **********************                         
!                                                                 
!   ADD TWO MATRICES                                              
!                                                                 
!   CALL ADDM(A,B,C,M,N,MS)                                       
!                                                                 
!   A .......... INPUT MATRIX, M BY N                             
!   B .......... INPUT MATRIX, M BY N                             
!   C .......... OUTPUT MATRIX, M BY N, RESULT OF ADDITION        
!   M .......... NUMBER OF ROWS IN A, EQUAL TO NUMBER OF ROWS IN B
!   N .......... NUMBER OF COLUMNS IN A, EQUAL TO NUMBER OF       
!                COLUMNS IN B                                     
!   MS ......... STORAGE MODE OF A, EQUAL TO STORAGE MODE OF B    
!                                                                       
!*********************************************************************

SUBROUTINE ADDM(A,B,C,M,N,MS)   

   IMPLICIT NONE
   REAL(KIND=8), INTENT(IN) :: A(:), B(:)
   REAL(KIND=8), INTENT(OUT) :: C(:)
   INTEGER, INTENT(IN) :: M,N,MS
   INTEGER :: I, K
                                        
   SELECT CASE (MS)
   CASE (0)         ! rectangular
      K=M*N                                                            
   CASE (1,3,4)     ! packed: symmetric, upper or lower triangular
      K=(M*(M+1))/2                                                    
   CASE (2)         ! diagonal
      K=M                                                              
   END SELECT
   
   DO I=1,K                                                        
      C(I)=A(I)+B(I)                                                    
   END DO

   RETURN                                                            
END SUBROUTINE ADDM

!*********************************************************************
!                  **********************                         
!                  *  SUBROUTINE   SUBM *                         
!                  **********************                         
!                                                                 
!   SUBTRACT TWO MATRICES                                         
!                                                                 
!   CALL SUBM(A,B,C,M,N,MS)                                       
!                                                                 
!   A .......... INPUT MATRIX, M BY N, MINUEND                    
!   B .......... INPUT MATRIX, M BY N, SUBTRAHEND                 
!   C .......... OUTPUT MATRIX, M BY N, RESULT OF SUBTRACTION     
!   M .......... NUMBER OF ROWS IN A, EQUAL TO NUMBER OF ROWS IN B
!   N .......... NUMBER OF COLUMNS IN A, EQUAL TO NUMBER OF       
!                COLUMNS IN B                                     
!   MS ......... STORAGE MODE OF A, EQUAL TO STORAGE MODE OF B    
!                                                                       
!*********************************************************************

SUBROUTINE SUBM(A,B,C,M,N,MS)                                     
    implicit none
    real(kind=8), intent(in) :: a(:),b(:)
    real(kind=8), intent(out) :: c(:)
    integer(kind=4), intent(in) :: m,n,ms
    
    integer(kind=4) :: k,i
   
   SELECT CASE (MS)
   CASE (0)         ! rectangular
      K=M*N                                                            
   CASE (1,3,4)     ! packed: symmetric, upper or lower triangular
      K=(M*(M+1))/2                                                    
   CASE (2)         ! diagonal
      K=M                                                              
   END SELECT
   
   DO I=1,K                                                        
      C(I)=A(I)-B(I)                                                    
   END DO

   RETURN                                                            
END SUBROUTINE SUBM

!*********************************************************************
!                                                                 
!                  **********************                         
!                  *  SUBROUTINE   SCM  *                         
!                  **********************                         
!                                                                 
!   MULTIPLY A MATRIX BY A SCALAR                                 
!                                                                 
!   CALL SCM (A,X,B,M,N,MS)                                       
!                                                                 
!   A .......... INPUT MATRIX, M BY N                             
!   X .......... SCALAR CONSTANT OR VARIABLE, FACTOR IN MULTI-    
!                PLICATION                                        
!   B .......... OUTPUT MATRIX, M BY N, RESULT OF MULTIPLICATION  
!   M .......... NUMBER OF ROWS IN A                              
!   N .......... NUMBER OF COLUMNS IN A                           
!   MS ......... STORAGE MODE OF A                                
!                                                                       
!*********************************************************************

SUBROUTINE SCM(A,C,B,M,N,MS)     
    implicit none
    real(kind=8), intent(in) :: a(:),c
    real(kind=8), intent(out) :: b(:)
    integer(kind=4), intent(in) :: m,n,ms
    
    integer(kind=4) :: k,i
   
   SELECT CASE (MS)
   CASE (0)         ! rectangular
      K=M*N                                                            
   CASE (1,3,4)     ! packed: symmetric, upper or lower triangular
      K=(M*(M+1))/2                                                    
   CASE (2)         ! diagonal
      K=M                                                              
   END SELECT
   
   DO I=1,K                                                     
      B(I)=C*A(I)                                                       
   END DO
END SUBROUTINE SCM
                                                                 
!*********************************************************************
!                  **********************                         
!                  *  SUBROUTINE MPDSD  *                         
!                  **********************                         
!                                                                 
!   RESCALE A SYMMETRIC (GRAMIAN) MATRIX (MS=1) BY PRE- AND POST- 
!   MULTIPLYING IT BY A DIAGONAL MATRIX (MS=2)                    
!                                                                 
!   CALL MPDSD (A,B,C,N)                                          
!                                                                 
!   A .......... INPUT MATRIX, N BY N, DIAGONAL (MSA=2), CONTAINS 
!                SCALING FACTORS                                  
!   B .......... INPUT MATRIX, N BY N, SYMMETRIC (MSB=1)          
!   C .......... OUTPUT MATRIX, N BY N, SYMMETRIC (MSC=1), RESULT 
!                OF RESCALING                                     
!   N .......... NUMBER OF ROWS AND NUMBER OF COLUMNS IN A AND B  
!                                                                 
!*********************************************************************

SUBROUTINE MPDSD(A,B,C,N)
   IMPLICIT NONE
   REAL(KIND=8), INTENT(IN) :: A(:), B(:)
   REAL(KIND=8), INTENT(OUT) :: C(:)
   INTEGER, INTENT(IN) :: N
   INTEGER :: IC, I, J
                                
   IC=0                                                              
   DO I=1,N                                                      
      DO J=1,I                                                      
         IC=IC+1                                                           
         C(IC)=B(IC)*A(I)*A(J)                                             
      END DO
   END DO
END SUBROUTINE  MPDSD

!*********************************************************************
!                  **********************                         
!                  *  SUBROUTINE MPYTR  *                         
!                  **********************                         
!                                                                 
!   MULTIPLY TWO MATRICES, THE FIRST ONE ENTERING IN ITS TRANS-   
!                POSED FORM                                       
!                                                                 
!   CALL MPYTR (A,B,C,MA,NA,MSB,NB)                               
!                                                                 
!   A .......... INPUT MATRIX, MA BY NA, TRANSPOSED FIRST FACTOR  
!                IN MULTIPLICATION, GENERAL RECTANGULAR (MSA=0)   
!   B .......... INPUT MATRIX, MA BY NB, SECOND FACTOR IN MULTI-  
!                PLICATION                                        
!   C .......... OUTPUT MATRIX, NA BY NB, RESULT OF MULTIPLICATION
!                GENERAL RECTANGULAR (MSC=0)                      
!   MA ......... NUMBER OF ROWS IN A, EQUAL TO NUMBER OF ROWS IN B
!   NA ......... NUMBER OF COLUMNS IN A                           
!   MSB ........ STORAGE MODE OF B                                
!   NB ......... NUMBER OF COLUMNS IN B                           
!                                                                       
!*********************************************************************

SUBROUTINE MPYTR(A,B,C,MA,NA,MSB,NB)
    implicit none
    real(kind=8),intent(in) :: a(:),b(:)
    real(kind=8),intent(out) :: c(:)
    integer(kind=4),intent(in) :: ma,na,msb,nb
    
    real(kind=8) :: x
    integer(kind=4) :: k,icb,iccb,j,inb,incb,llb,lhb,jnb,i,ina,m

   K=0                                                               
   ICB=0                                                             
   ICCB=0                                                            
   DO J=1,NB                                                     
   
      SELECT CASE (MSB)
      
      CASE (0)
         INB=(J-1)*MA+1                                                    
         INCB=1                                                            
         LLB=1                                                             
         LHB=MA                                                            
      
      CASE (2)
         INB=J                                                             
         INCB=0                                                            
         LLB=J                                                             
         LHB=J                                                             
      
      CASE (3)
         INB=J*(J+1)/2                                                     
         INCB=J                                                            
         ICCB=1                                                            
         LLB=J                                                             
         LHB=MA                                                            
      
      CASE (1,4)
         INB=J*(J-1)/2+1                                                   
         INCB=1                                                            
         LLB=1                                                             
         IF(MSB .NE. 1) THEN
            LHB=J
         ELSE
            LHB=MA 
         ENDIF
      
      END SELECT
      
      JNB=INB                                                           
      
      DO I=1,NA                                                     
         INA=(I-1)*MA+1                                                    
         K=K+1                                                             
         X=0.0D0                                                           
         
         IF (LLB .GE. 1) THEN
            IF (LLB .GT. 1) INA=INA+LLB-1
            
            IF(MA .GE. LHB) THEN
               DO M=LLB,LHB             
                  X=X+A(INA)*B(INB)    
                  IF((MSB .EQ. 1) .AND. (M .EQ. J)) THEN
                     INCB=J           
                     ICCB=1          
                  ENDIF
                  INA=INA+1         
                  INB=INB+INCB+ICB 
                  ICB=ICB+ICCB    
               END DO
               IF(MSB .EQ. 1) THEN
                  INCB=1         
                  ICCB=0        
               ENDIF
               INB=JNB 
               ICB=0  
            ENDIF
         
         ENDIF
         C(K)=X                                                            
      END DO
   END DO
   RETURN                                                            
END SUBROUTINE  MPYTR

!*********************************************************************
!                                                                 
!                  **********************                        
!                  *  SUBROUTINE GRAMM  *                       
!                  **********************                      
!                                                             
!   OBTAIN THE GRAMIAN MATRIX OF GENERALIZED PRODUCTS OF ROW 
!   VECTORS OF A SPECIFIED MATRIX BY PRE- AND POSTMULTIPLYING A   
!   METRIC MATRIX (SYMMETRIC OR DIAGONAL, MS=1,2) BY THE MATRIX  
!   AND ITS TRANSPOSE                                           
!                                                              
!   CALL GRAMM (A,B,C,M,N,MSB,W)                              
!                                                            
!   A .......... INPUT MATRIX, M BY N, GENERAL RECTANGULAR (MSA=0)
!   B .......... INPUT MATRIX, N BY N, METRIC MATRIX, SYMMETRIC OR
!                DIAGONAL                                        
!   C .......... OUTPUT MATRIX, M BY M, GENERALIZED GRAMIAN,    
!                SYMMETRIC (MSC=1)                             
!   M .......... NUMBER OF ROWS IN A                          
!   N .......... NUMBER OF COLUMNS IN A, EQUAL TO NUMBER OF ROWS  
!                AND NUMBER OF COLUMNS IN B                      
!   MSB ........ STORAGE MODE OF B                              
!   W .......... WORKING VECTOR OF LENGTH N                    
!                                                             
!     Storage modes: 0 SQUARE, 1 PACKED SYMMETRIC, 2 DIAGONAL, 
!                    3 PACKED LOWER TRIANGLE
!                    4 PACKED UPPER TRIANGLE
!
!*********************************************************************

SUBROUTINE GRAMM(A,B,C,M,N,MSB,W)   
    implicit none
    integer(kind=4),intent(in) :: m,n,msb
    real(kind=8),intent(in) :: a(m,n),b(:)
    real(kind=8),intent(out),dimension(:) :: c
    real(kind=8),intent(inout),dimension(:) :: w
    
    real(kind=8) :: x
    integer(kind=4) :: kc,i,kk,ic,l,k,j
   
   IF (MSB .EQ. 0) THEN
      !'ERROR MESSAGE FROM SUBROUTINE GRAMM: METRIC IS GENERAL RECTANGULAR'
   
   ELSEIF (MSB .EQ. 1) THEN
      KC=0                                                          
      DO I=1,M                                                 
         KK=0                                                        
         DO K=1,N                                               
            X=0.0D0                                                   
            IC=KK                                                    
            DO L=1,K                                            
               IC=IC+1                                                
               X=X+A(I,L)*B(IC)                                      
            END DO
            IF (K .NE. N) THEN
               J=K+1                                               
               IC=IC+K                                            
               DO L=J,N                                      
                  X=X+A(I,L)*B(IC)                                 
                  IC=IC+L  
               END DO
            ENDIF
            W(K)=X 
            KK=KK+K 
         END DO
         
         DO K=1,I                                                   
            KC=KC+1                                                       
            X=0.0D0                                                     
            DO L=1,N                                               
               X=X+A(K,L)*W(L)                                           
            END DO
            C(KC)=X                                                  
         END DO
      END DO
   
   ELSE   ! MSB .GE. 2
      IC=0                                                 
      DO I=1,M                                        
         DO J=1,I                                       
            X=0.0D0                                           
            DO K=1,N                                     
               X=X+A(I,K)*A(J,K)*B(K)                          
            END DO
            IC=IC+1                                        
            C(IC)=X                                       
         END DO
      END DO
   ENDIF
   RETURN                                   
END SUBROUTINE GRAMM

!*********************************************************************
! SUBROUTINE HRECUR
!
!      PURPOSE: CALCULATE GAUSS-HERMITE QUADRATURE
!*********************************************************************

SUBROUTINE HRECUR(PN,DPN,PN1,X,NN)
   implicit none
   real(kind=8),intent(in)::x
   real(kind=8),intent(out)::pn,dpn,pn1
   integer(kind=4),intent(in)::nn
   
   integer(kind=4)::j
   real(kind=8)::p,dp,p1,dp1,fj,fj2,q,dq

   
   P1=1.
   P=X
   DP1=0.
   DP=1.
   
   DO J=2,NN
      FJ=J
      FJ2=(FJ-1.)/2.
      Q=X*P-FJ2*P1
      DQ=X*DP+P-FJ2*DP1
      P1=P
      P=Q
      DP1=DP
      DP=DQ
   END DO
   
   PN=P
   DPN=DP
   PN1=P1
   
END SUBROUTINE HRECUR

!*********************************************************************
! SUBROUTINE HERMIT
!
!      PURPOSE: CALCULATE GAUSS-HERMITE QUADRATURE
!*********************************************************************

SUBROUTINE HERMIT(X,A,NN,EPSQ)
   USE INTERFACE_HROOT
   implicit none
   real(kind=8),intent(in)::epsq
   real(kind=8),intent(out)::x(:),a(:)
   integer(kind=4),intent(in)::nn
   
   real(kind=8)::pi,fn,cc,s,gammas,xt,dpn,pn1
   integer(kind=4)::n1,n2,i,ni
   
   DATA PI/3.141592654D0/
   
   FN=DBLE(NN)
   N1=NN-1
   N2=(NN+1)/2
   CC=1.7724538509*GAMMAS(FN)/(2.**N1)
   S=(2.*FN+1.)**.16667
   
   DO I=1,N2
      SELECT CASE (I)
      CASE (1)
         XT=S**3-1.85575/S
      CASE (2)
         XT=XT-1.14*FN**.426/XT
      CASE (3)
         XT=1.86*XT-.86*X(1)
      CASE (4)
         XT=1.91*XT-.91*X(2)
      CASE DEFAULT
         XT=2.*XT-X(I-2)
      END SELECT
      
      CALL HROOT(XT,NN,DPN,PN1,EPSQ)
      X(I)=XT
      A(I)=CC/DPN/PN1
      NI=NN-I+1
      X(NI)=-XT
      A(NI)=A(I)
   END DO
   
   DO I=1,NN
      X(I)=-X(I)*DSQRT(2.D0)
      A(I)=A(I)*(1./DSQRT(PI))
   END DO
   
END SUBROUTINE HERMIT

!*********************************************************************
! SUBROUTINE HROOT
!
!      PURPOSE: CALCULATE GAUSS-HERMITE QUADRATURE
!*********************************************************************

SUBROUTINE HROOT(X,NN,DPN,PN1,EPSQ)
   USE INTERFACE_HRECUR
   implicit none
   real(kind=8),intent(in)::epsq
   real(kind=8),intent(inout)::x
   real(kind=8),intent(out)::dpn,pn1
   integer(kind=4),intent(in)::nn
   
   integer(kind=4)::iter
   real(kind=8)::d,p,dp
   
   DO ITER=1,10
      CALL HRECUR(P,DP,PN1,X,NN)
      D=P/DP
      X=X-D
      IF(DABS(D) .LE. EPSQ) EXIT
   END DO
   
   DPN=DP
   RETURN
END SUBROUTINE HROOT

!*********************************************************************
!                                                                 
!                  **********************                         
!                  *  SUBROUTINE  KMPY  *                         
!                  **********************                         
!                                                                 
!   FORM THE KRONECKER (DIRECT) PRODUCT OF TWO MATRICES           
!                                                                 
!   CALL KMPY (A,B,C,MA,NA,MS,MB,NB)                              
!                                                                 
!   A .......... INPUT MATRIX, MA BY NA, FIRST FACTOR IN          
!                KRONECKER MULTIPLICATION                         
!   B .......... INPUT MATRIX, MB BY NB, SECOND FACTOR IN         
!                KRONECKER MULTIPLICATION                         
!   C .......... OUTPUT MATRIX, (MA*MB) BY (NA*NB), KRONECKER     
!                PRODUCT                                          
!   MA ......... NUMBER OF ROWS IN A                              
!   NA ......... NUMBER OF COLUMNS IN A                           
!   MS ......... STORAGE MODE OF A, EQUAL TO STORAGE MODE OF B    
!   MB ......... NUMBER OF ROWS IN B                              
!   NB ......... NUMBER OF COLUMNS IN B                           
!                                                                 
!*********************************************************************

SUBROUTINE KMPY(A,B,C,P,Q,MS,R,S)                                 
    implicit none
    integer(kind=4),intent(in)::p,q,ms,r,s
    real(kind=8),intent(in)::a(p,q),b(r,s)
    real(kind=8),intent(out)::c(:)
    
   INTEGER EA,EB,ic,ii,i,j,jj,k,kk,l,ll,neb,kr,kea,kjj,nc
   
   IF (MS .LT. 1) THEN
      IC=P*Q*R*S                                                        
      II=Q+1                                                            
      
      DO I=1,Q                                                      
         II=II-1                                                           
         JJ=S+1                                                            
         
         DO J=1,S             
            JJ=JJ-1          
            KK=P+1          
            
            DO K=1,P       
               KK=KK-1    
               LL=R+1    
               
               DO L=1,R 
                  LL=LL-1  
                  C(IC)=A(KK,II)*B(LL,JJ) 
                  IC=IC-1                
               END DO
            END DO
         END DO
      END DO
      
   ELSEIF (MS .EQ. 1) THEN
   
      EA=(P*(P+1)/2)+1                                                  
      NEB=(R*(R+1)/2)+1                                                 
      IC=P*R                                                            
      IC=(IC*(IC+1))/2                                                  
      KR=R-1                                                            
      II=P+1                                                            
      
      DO I=1,P                                                      
         II=II-1                                                           
         KEA=EA                                                            
         EB=NEB                                                            
         KJJ=R+1                                                           
         
         DO J=1,R                                                      
            KJJ=KJJ-1
            EA=KEA 
            
            DO K=1,II
               EA=EA-1 
               JJ=KR  
               NC=R  
               IF (K .EQ. 1) NC=KJJ  
               
               DO L=1,NC            
                  IF (K .EQ. 1) THEN
                     EB=EB-1        
                  ELSEIF (L .EQ. 1) THEN
                     EB=NEB-J      
                  ELSEIF (L .LE. J) THEN
                     EB=EB-JJ     
                     JJ=JJ-1     
                  ELSE
                     EB=EB-1
                  ENDIF
                  ! C(IC)=A(EA,1)*B(EB,1)  
                  C(IC)=A(MOD(EA-1,P) + 1,(EA-1)/P +1) &
                        * B(MOD(EB-1,R) + 1,(EB-1)/R +1)
                  IC=IC-1               
               END DO
            END DO
         END DO
      END DO
   
   ELSE
      IC=P*R                                                            
      II=P+1                                                            
      
      DO I=1,P                                                      
         II=II-1                                                           
         JJ=R+1                                                            
         
         DO J=1,R                                                      
            JJ=JJ-1      
            C(IC)=A(II,1)*B(JJ,1)  
            IC=IC-1     
         END DO
      END DO
   ENDIF
   
END SUBROUTINE KMPY

!*********************************************************************
!                                                                 
!                  **********************                         
!                  *  SUBROUTINE   MPYM *                         
!                  **********************                         
!                                                                 
!   MULTIPLY TWO MATRICES                                         
!                                                                 
!   CALL MPYM(A,B,C,MA,NA,MSA,MSB,NB)                             
!                                                                 
!   A .......... INPUT MATRIX, MA BY NA, FIRST FACTOR IN MULTI-   
!                PLICATION                                        
!   B .......... INPUT MATRIX, NA BY NB, SECOND FACTOR IN MULTI-  
!                PLICATION                                        
!   C .......... OUTPUT MATRIX, MA BY NB, RESULT OF MULTIPLI-     
!                CATION, GENERAL RECTANGULAR (MS=0)               
!   MA ......... NUMBER OF ROWS IN A                              
!   NA ......... NUMBER OF COLUMNS IN A, EQUAL TO NUMBER OF ROWS  
!                IN B                                             
!   MSA ........ STORAGE MODE OF A                                
!   MSB ........ STORAGE MODE OF B                                
!   NB ......... NUMBER OF COLUMNS IN B                           
!                                                                       
!*********************************************************************

SUBROUTINE MPYM (A,B,C,MA,NA,MSA,MSB,NB)      
    implicit none
    real(kind=8),intent(in)::a(:),b(:)
    real(kind=8),intent(out)::c(:)
    integer(kind=4),intent(in)::ma,na,msa,msb,nb
    
    real(kind=8)::x
    integer(kind=4)::ica,icb,icca,iccb,loop,ii,jj,k,k1,k2,k3,j,m,i,lla,llb,lha,lhb,ina,inca,incb,inb,jnb,jna
   
   ICA=0                                                             
   ICB=0                                                             
   ICCA=0                                                            
   ICCB=0                                                            
   LOOP=1                                                            
   II=0                                                              
   JJ=0
                                            
   ! Abort if either array is zero-sized
   IF(MA .EQ. 0 .OR. NA .EQ. 0 .OR. NB .EQ. 0) RETURN
   
   IF((MSA .GE. 3)  .AND. (MSB .EQ. 0)) LOOP=2 
   
   IF (LOOP .EQ. 1) THEN
      CALL SYNCB()   !   DEFINITION OF PARAMETERS FOR MATRIX B
      CALL SYNCA()   !   DEFINITION OF PARAMETERS FOR MATRIX A
   ELSE
      CALL SYNCA()   !   DEFINITION OF PARAMETERS FOR MATRIX A
      CALL SYNCB()   !   DEFINITION OF PARAMETERS FOR MATRIX B
   ENDIF
   
   DO
      !     SYNCHRONIZATION OF PARAMETERS  
      K=I+(J-1)*MA  ! linear Index in destination array
      X=0.0D0                                                           
  
      IF(.NOT.(( LLA .LT. LLB) .AND. (LHA .LT. LLB)) &
       .AND. .NOT.((LLA .GT. LLB) .AND. (LHB .LT. LLA))) THEN
     
         IF (LLA .EQ. LLB) THEN
            K1=LLA  
         ELSEIF (LLA .LT. LLB) THEN
            K1=LLB 
            K3=K1-1 
            DO M=LLA,K3                                                    
               IF((MSA .EQ. 1) .AND. (M .EQ. I)) THEN
                  INCA=I 
                  ICCA=1
               ENDIF 
               INA=INA+INCA+ICA  
               ICA=ICA+ICCA     
            END DO
            
         ELSE ! lla .GT. llb
            K1=LLA 
            K3=K1-1  
            DO M=LLB,K3                                                    
               IF((MSB .EQ. 1) .AND. (M .EQ. J)) THEN
                  INCB=J  
                  ICCB=1 
               ENDIF
               INB=INB+INCB+ICB
               ICB=ICB+ICCB   
            END DO
         ENDIF
      
         IF(LHA .LT. LHB) THEN
            K2=LHA  
         ELSE
            K2=LHB 
         ENDIF
         ! 
         ! VECTOR MULTIPLICATION AND RESETTING OF PARAMETERS 
         !
         DO M=K1,K2                                                     
            X=X+A(INA)*B(INB) 
            IF((MSA .EQ. 1) .AND. (M .EQ. I)) THEN
               INCA=I
               ICCA=1 
            ENDIF
            IF((MSB .EQ. 1) .AND. (M .EQ. J)) THEN
               INCB=J
               ICCB=1
            ENDIF
            INA=INA+INCA+ICA    
            ICA=ICA+ICCA       
            INB=INB+INCB+ICB  
            ICB=ICB+ICCB     
         END DO
         
         IF(MSB .EQ. 1) THEN
            INCB=1          
            ICCB=0         
         ENDIF
         INB=JNB                                                           
         ICB=0                                                             
         IF(MSA .EQ. 1) THEN
            INCA=1        
            ICCA=0       
         ENDIF
         INA=JNA                                                           
         ICA=0                                                             
      ENDIF
  
      C(K)=X  ! store the accumulated row/column product 
              ! in the result array
      
      IF (LOOP .EQ. 1) THEN
         IF(II .LT. MA) THEN
            CALL SYNCA()
         ELSEIF (II .EQ. MA) THEN
            IF(JJ .LT. NB) THEN
               II=0   
               CALL SYNCB()
               CALL SYNCA()
            ELSE
               EXIT
            ENDIF
         ELSE
            EXIT
         ENDIF
      ELSE             ! LOOP .EQ. 2
         IF(JJ .LT. NB) THEN
            CALL SYNCB()
         ELSEIF (JJ .EQ. NB) THEN
            IF(II .LT. MA) THEN
               JJ=0
               CALL SYNCA()
               CALL SYNCB()
            ELSE
               EXIT
            ENDIF
         ELSE
            EXIT
         ENDIF
      ENDIF
   ENDDO
   
   RETURN                                                            

   CONTAINS
   
   SUBROUTINE SYNCA
      II=II+1                                                           
      IF ((MSA .EQ. 3) .AND. (MSB .EQ. 0)) THEN
         ! Storage mode of B is 0 and A is 3.  Note that this
         ! DOES NOT MATCH the equivalent test for B in SYNCB, where
         ! A is 0 and B is 4  TG 1/2/01
         I=MA-II+1                                                         
      ELSE
         I=II                                                              
      ENDIF
      
      SELECT CASE (MSA)
      CASE (0)
         INA=I                                                             
         INCA=MA                                                           
         LLA=1                                                             
         LHA=NA                                                            
      
      CASE (1,3)
         INA=I*(I-1)/2+1                                                   
         INCA=1                                                            
         LLA=1                                                             
         IF(MSA .NE. 1) THEN
            LHA=I                                                             
         ELSE
            LHA=NA                                                            
            ICCA=0                                                            
            ICA=0                                                             
         ENDIF
      
      CASE (2)
         INA=I                                                             
         INCA=0                                                            
         LLA=I                                                             
         LHA=I                                                             
      
      CASE(4)
         INA=I*(I+1)/2                                                     
         INCA=I                                                            
         ICCA=1                                                            
         ICA=0                                                             
         LLA=I                                                             
         LHA=NA                                                            
      
      END SELECT
      
      JNA=INA                                                           
   END SUBROUTINE SYNCA
   
   SUBROUTINE SYNCB
      JJ=JJ+1                                                           
      IF ((MSA .EQ. 0) .AND. (MSB .EQ. 4)) THEN
         ! Storage mode of A is 0 and B is 4
         J=NB-JJ+1                                                         
      ELSE
         J=JJ                                                              
      ENDIF
      
      SELECT CASE (MSB)
      CASE (0)
         INB=(J-1)*NA+1                                                    
         INCB=1                                                            
         LLB=1                                                             
         LHB=NA                                                            
      
      CASE (2)
         INB=J                                                             
         INCB=0                                                            
         LLB=J                                                             
         LHB=J                                                             
      
      CASE (3)
         INB=J*(J+1)/2                                                     
         INCB=J                                                            
         ICCB=1                                                            
         LLB=J                                                             
         LHB=NA                                                            
      
      CASE(1,4)
         INB=J*(J-1)/2+1                                                   
         INCB=1                                                            
         LLB=1                                                             
         IF(MSB .NE. 1) THEN
            LHB=J                                                             
         ELSE
            LHB=NA                                                            
         ENDIF
         
      END SELECT
      JNB=INB                                                           
   END SUBROUTINE SYNCB
   
END SUBROUTINE MPYM

!*********************************************************************
!                  **********************                         
!                  *  SUBROUTINE RELOC  *                         
!                  **********************                         
!                                                                 
!   DUPLICATE A MATRIX                                            
!                                                                 
!   CALL RELOC (A,B,M,N,MS)                                       
!                                                                 
!   A .......... INPUT MATRIX, M BY N                             
!   B .......... OUTPUT MATRIX, M BY N, EQUAL TO A                
!   M .......... NUMBER OF ROWS IN A                              
!   N .......... NUMBER OF COLUMNS IN A                           
!   MS ......... STORAGE MODE OF A                                
!                                                                       
!*********************************************************************

SUBROUTINE RELOC (A,B,M,N,MS)
    implicit none
    real(kind=8),intent(in)::a(:)
    real(kind=8),intent(out)::b(:)
    integer(kind=4),intent(in)::m,n,ms
    
    integer(kind=4)::l,j

   SELECT CASE (MS)
   CASE (0)       ! square matrix
      L = M*N                                                           
   CASE (1,3,4)   ! packed matrix
      L = (M*(M+1))/2                                                   
   CASE (2)       ! diagonal matrix
      L=N                                                               
   END SELECT
   DO J=1,L                                                        
      B(J) = A(J)                                                       
   END DO
   RETURN                                                            
END SUBROUTINE  RELOC

!*********************************************************************
! SUBROUTINE YSAME (YVEC,N,NSAME)                            
!                                                            
! INDICATES WHETHER ALL ELEMENTS OF YVEC ARE THE SAME        
!                                                            
! Parameters Sent                                            
! YVEC  = N x 1 vector of values                             
! N     = NUMBER OF ROWS IN YVEC                             
!                                                            
! Parameters Returned                                        
! NSAME = 0 if elements are not all the same                 
!       = 1 if elements are all the same                     
!*********************************************************************

SUBROUTINE YSAME(YVEC,N,NSAME)
   USE INTERFACE_FP_EQUAL
   implicit none
   integer(kind=4),intent(in)::n
   real(kind=8),intent(in)::yvec(n)
   integer(kind=4),intent(out)::nsame
   
   real(kind=8)::yold
   integer(kind=4)::nsum,i

   IF (N .EQ. 1) THEN 
       NSAME = 1        
   ELSE                 
       NSAME = 0
       NSUM  = 0
       YOLD  = YVEC(1)   
       DO I = 2,N    
          IF (FP_EQUAL(YVEC(I) , YOLD)) NSUM = NSUM + 1
          YOLD = YVEC(I)
       END DO
       IF (NSUM .EQ. (N-1)) NSAME = 1
   ENDIF
   RETURN
END SUBROUTINE YSAME

!*********************************************************************
!                                                                   
!                   **********************                         
!                   *  SUBROUTINE  GEND  *                          
!                   **********************                          
!                                                                  
!    GENERATE A SCALAR MATRIX ( A SPECIAL CASE OF WHICH IS THE    
!    IDENTITY MATRIX), I.E., A SQUARE MATRIX ALL DIAGONAL        
!    ELEMENTS OF WHICH ARE EQUAL TO A SPECIFIED CONSTANT, WITH THE  
!    OFF-DIAGONAL ELEMENTS EQUAL TO ZERO                           
!                                                                 
!    CALL GEND (A,X,N,MS)                                        
!                                                               
!    A .......... OUTPUT MATRIX, N BY N                        
!    X .......... INPUT CONSTANT OR VARIABLE, SPECIFIES VALUE FOR   
!                 DIAGONAL ELEMENTS IN A                           
!    N .......... NUMBER OF ROWS IN A, EQUAL TO NUMBER OF COLUMNS 
!    MS ......... DESIRED STORAGE MODE OF A                      
!                                                               
!*********************************************************************

SUBROUTINE GEND(A,C,N,MS)
    implicit none
    real(kind=8),intent(out)::a(:)
    real(kind=8),intent(in)::c
    integer(kind=4),intent(in)::n,ms
    
    integer(kind=4)::l,j,k
   
   SELECT CASE(MS)
   CASE (0)        ! rectangular
      L = N*N                                                  
      DO J=1,L                                                   
         A(J) = 0                                                    
      END DO
      K = N + 1                                                  
      DO J=1,L,K                                              
         A(J) = C                                                 
      END DO
   
   CASE (1,3,4)    ! packed symm, upper or lower triangle
      L = (N*(N+1))/2                                        
      DO J=1,L                                            
         A(J) = 0                                             
      END DO
      K = 1                                               
      DO J=1,N                                         
         A(K) = C                                          
         K = K + J + 1                                    
      END DO
      
   CASE (2)        ! diagonal
      DO J=1,N                                     
         A(J) = C                                      
      END DO
   END SELECT

   RETURN                               
END SUBROUTINE GEND

!*********************************************************************
!                                      
!                   **********************                          
!                   *  SUBROUTINE  ADJR  *                         
!                   **********************                        
!                                                                
!    ADJOIN THE ROWS OF TWO MATRICES TO PRODUCE A RESULTANT MATRIX  
!                                                                  
!    CALL ADJR (A,B,C,MA,N,MB)                                      
!                                                                  
!    A .......... INPUT MATRIX, MA BY N, GENERAL RECTANGULAR (MS=0) 
!    B .......... INPUT MATRIX, MB BY N, GENERAL RECTANGULAR (MS=0)
!    C .......... OUTPUT MATRIX, (MA+MB) BY N, RESULT OF ADJOINING, 
!                 GENERAL RECTANGULAR (MS=0)                       
!    MA ......... NUMBER OF ROWS IN A                             
!    N .......... NUMBER OF COLUMNS IN A, EQUAL TO NUMBER OF     
!                 COLUMNS IN B                                  
!    MB ......... NUMBER OF ROWS IN B                               
!                                                                  
!*********************************************************************

SUBROUTINE ADJR(A,B,C,MA,N,MB)
    implicit none
    integer(kind=4),intent(in)::n,ma,mb
    real(kind=8),intent(in)::a(ma,n),b(mb,n)
    real(kind=8),intent(out)::c(:)

    integer(kind=4)::j,jj,lk,ii,i,ll,l,jz,mab
   
   MAB = MA + MB                                                 
   LL = N + 1                                                   
   LK = MA + 1                                                 
   
   DO JJ=1,N                                                
      J = LL-JJ                                                 
      L = MAB*J-MB                                             
      
      DO II=1,MA                                            
         I = LK-II                                              
         C(L) = A(I,J)                                         
         L = L - 1                                            
      END DO
   END DO
   
   DO J=1,N                                          
      L = MAB*J-MB                                       
      DO I=1,MB                                       
         L = L + 1                                        
         C(L) = B(I,J)                                   
      END DO
   END DO
   
   RETURN                                    
END SUBROUTINE ADJR

!*********************************************************************
!                                        
!                   **********************                          
!                   *  SUBROUTINE  ADJC  *                         
!                   **********************                        
!                                                                
!    ADJOIN THE COLUMNS OF TWO MATRICES TO PRODUCE A RESULTANT  
!    MATRIX                                                    
!                                                             
!    CALL ADJC (A,B,C,M,NA,NB)                               
!                                                                   
!    A .......... INPUT MATRIX, M BY NA, GENERAL RECTANGULAR (MS=0)
!    B .......... INPUT MATRIX, M BY NB, GENERAL RECTANGULAR (MS=0) 
!    C .......... OUTPUT MATRIX, M BY (NA+NB), RESULT OF ADJOINING,
!                 GENERAL RECTANGULAR (MS=0)                      
!    M .......... NUMBER OF ROWS IN A, EQUAL TO NUMBER OF ROWS IN B 
!    NA ......... NUMBER OF COLUMNS IN A               
!    NB ......... NUMBER OF COLUMNS IN B              
!                                                    
!*********************************************************************

SUBROUTINE ADJC(A,B,C,M,NA,NB)
    implicit none
    real(kind=8),intent(in)::a(:),b(:)
    real(kind=8),intent(out)::c(:)
    integer(kind=4),intent(in)::m,na,nb
    
    integer(kind=4)::k,kk,l,jz
   
   K = M*NA                                        
   KK = M*NB                                      
   
   DO L=1,K                                    
      C(L) = A(L)                                  
   END DO
   
   DO L=1,KK                                 
      JZ = L + K                                 
      C(JZ) = B(L)                              
   END DO

END SUBROUTINE ADJC

!*********************************************************************
!                                                                 
!                  *********************                         
!                  *  SUBROUTINE  INVS  *                         
!                  **********************                         
!                                                                 
!   INVERT A SYMMETRIC MATRIX (MS=1) IN PLACE AND CALCULATE THE   
!   DETERMINANT                                                   
!                                                                 
!   CALL INVS (A,N,DET,W)                                         
!                                                                 
!   A .......... INPUT-OUTPUT MATRIX, N BY N,SYMMETRIC (MS=1)     
!   N .......... NUMBER OF ROWS IN A, EQUAL TO NUMBER OF COLUMNS  
!   DET ........ OUTPUT SCALAR, DETERMINANT OF A                  
!   W .......... WORKING VECTOR OF LENGTH N                       
!   IER......... OPTIONAL ERROR FLAG
!                                                                       
!*********************************************************************

SUBROUTINE INVS(A,N,C,W,IER,VERBOSE)
    implicit none
    real(kind=8),intent(inout)::a(:)
    real(kind=8),intent(out)::w(:),c
    integer(kind=4),intent(in)::n
   INTEGER, INTENT(inout), OPTIONAL :: IER
   LOGICAL, INTENT(in), OPTIONAL :: VERBOSE

   real(kind=8)::U,X,Y,Z,D
   integer(kind=4)::diagmk,diag,diag2,rowno,rowcol,colno,k,km1,i,j

   IF (PRESENT(IER)) THEN
      IER = 0
   END IF 

   D=A(1)                                                            
   
   IF(D .NE. 0)  THEN
   
      A(1)=1.0D0/D                                                      
      
      IF(N .GT. 1) THEN
         
         DIAG=1                                                            
         
         DO K=2,N                                                     
            KM1=K-1             
            DIAGMK=DIAG        
            DIAG=DIAG+K       
            U=A(DIAG)        
            COLNO=DIAGMK    
            DIAG2=0        
            
            DO I=1,KM1                                                     
               X=0.0D0    
               COLNO=COLNO+1         
               ROWNO=DIAGMK         
               J=1                 
               ROWCOL=DIAG2       
               DO WHILE (J .LT. I)
                  ROWCOL=ROWCOL+1    
                  ROWNO=ROWNO+1     
                  Y=A(ROWCOL)      
                  Z=A(ROWNO)      
                  X=X+Y*Z        
                  J=J+1         
               END DO
               
               ROWCOL=ROWCOL+1   
               
               DO WHILE (J .LT. K)
                  ROWNO=ROWNO+1        
                  Y=A(ROWCOL)         
                  Z=A(ROWNO)         
                  X=X+Y*Z           
                  ROWCOL=ROWCOL+J  
                  J=J+1           
               END DO
               
               W(I)=-X          
               Y=A(COLNO)      
               U = U-X*Y      
               DIAG2=DIAG2+I 
            END DO
            D=D*U
            
            IF(U .NE. 0) THEN
               A(DIAG)=1.0D0/U   
               ROWNO=DIAGMK     
               DIAG2=0         
               
               DO I=1,KM1     
                  ROWNO=ROWNO+1  
                  DIAG2=DIAG2+I 
                  X=W(I)       
                  X=X/U       
                  A(ROWNO)=X 
                  ROWCOL=DIAG2      
                  DO J=I,KM1     
                     Y=W(J)             
                     Z=A(ROWCOL)       
                     A(ROWCOL)=Z+X*Y  
                     ROWCOL=ROWCOL+J 
                  END DO
               END DO
            ENDIF
         END DO
      ENDIF
   ENDIF
   
   C = D
   IF(D .EQ. 0) THEN
      !IF(PRESENT(VERBOSE)) THEN
      !   ERROR MESSAGE FROM SUBROUTINE INVS: MATRIX IS SINGULAR'
      !ELSE
      !   ERROR MESSAGE FROM SUBROUTINE INVS: MATRIX IS SINGULAR'
      !ENDIF
      IF (PRESENT(IER)) THEN
         IER = 1
      END IF 
   ENDIF
   RETURN                                                            
END SUBROUTINE INVS

!*********************************************************************
!                                      
!                   **********************                          
!                   *  SUBROUTINE  INVT  *                         
!                   **********************                        
!                                                                
!    INVERT A TRUE TRIANGULAR MATRIX (MS=3,4) IN PLACE AND      
!    CALCULATE ITS DETERMINANT                                 
!                                                             
!    CALL INVT (A,N,DET)                                     
!                                                           
!    A .......... INPUT-OUTPUT MATRIX, N BY N, TRUE TRIANGULAR      
!                 (MS=3,4)                                         
!    N .......... NUMBER OF ROWS IN A, EQUAL TO NUMBER OF COLUMNS 
!    DET ........ OUTPUT SCALAR, DETERMINANT OF A                
!                                                               
!*********************************************************************

SUBROUTINE INVT(A,N,C)
    implicit none
    real(kind=8),intent(inout) :: a(:)
    real(kind=8),intent(out) :: c
    integer(kind=4),intent(in) :: n
    
    real(kind=8) :: d,x,y,z,u
    integer(kind=4) :: DIAG,ROWNO,COLNO,ROWCOL,i,k,j                           
   
   D=1.0D0                                                   
   DIAG=0                                                   
   
   DO I=1, N
   
      DIAG=DIAG+I                                           
      X=A(DIAG)                                                          
      D=D*X                                                             
      IF(X .EQ. 0)  EXIT    ! escape out of  loop
      A(DIAG)=1.0D0/X                                                 
      COLNO=DIAG                                                     
      
      DO K=1,I-1
         COLNO=COLNO-1                                               
         Y=A(COLNO)                                                 
         Y=-Y/X                                                    
         A(COLNO)=Y                                               
         ROWNO=DIAG                                              
         ROWCOL=COLNO                                           
         DO J=I,N-1
            ROWNO=ROWNO+J                                       
            ROWCOL=ROWCOL+J                                    
            Z=A(ROWCOL)                                       
            U=A(ROWNO)                                       
            A(ROWCOL)=Z+Y*U                                 
         END DO    ! loop j = I,N-1
      END DO   ! loop k=1,i-1
      ROWNO=DIAG                                 
      
      DO J=I,N-1
         ROWNO=ROWNO+J                           
         Y=A(ROWNO)                             
         A(ROWNO)=Y/X                          
      END DO    ! loop j = I,N-1
      
   END DO  ! loop I=1,N
   
   C=D                              
   
   !IF(C .EQ. 0) THEN
   !   'ERROR MESSAGE FROM SUBROUTINE INVT: MATRIX IS SINGULAR'
   !ENDIF
   
   RETURN                                                      
END SUBROUTINE INVT

!*********************************************************************
!                   **********************                         
!                   *  SUBROUTINE GRAMT  *                        
!                   **********************                       
!
!*********************************************************************

SUBROUTINE GRAMT(A,C,M,N)
    implicit none
    integer(kind=4),intent(in)::m,n
    REAL(kind=8),intent(in)::A(m,n)
    real(kind=8),intent(out)::C(:)

    real(kind=8)::x
    integer(kind=4)::ic,i,j,k
   
   IC=0
   
   DO I=1,N
      DO J=1,I
         X=0.0D0
         DO K=1,M
            X=X+A(K,I)*A(K,J)
         END DO
         IC=IC+1
         C(IC)=X
      END DO
   END DO
   
   RETURN
END SUBROUTINE GRAMT

!*********************************************************************
!   SUBROUTINE QUADP (B,B1,A,NQ1,NQ,NDIM,IUNIF,WA1,WA2,sca) **  
!                                                         
!   DEFINES QUADRATURE POINTS AND WEIGHTS                 
!                                                         
!   Parameters Sent                                       
!   NQ1    = Number of Quadrature Points per Dimension    
!   NDIM   = Number of Dimensions                         
!   IUNIF  = 0 for Gaussian Distribution                  
!          = 1 for Uniform  Distribution                  
!          = 2 for log gamma Distribution                 
!   SCA    = value of scale parameter (only for log gamma)
!                                                         
!   Parameters Returned                                   
!   NQ     = Total Number of Quadrature Points            
!   B1     = NQ1 quadrature points (1 dimension)          
!   B      = NDIM*(NQ1**NDIM) quadrature points           
!   A      = NQ1**NDIM quadrature weights                 
!                                                         
!   Work Vectors                                          
!   WA1    = NQ1 work vector                              
!   WA2    = NDIM*(NQ1**NDIM) work vector                 
!                                                         
!*********************************************************************

SUBROUTINE QUADP(B,B1,A,NQ1,NQ,NDIM,IUNIF,WA1,WA2,SCA)
   USE INTERFACE_HERMIT
   IMPLICIT NONE
   INTEGER, INTENT(IN)                                    :: NQ1,NDIM,IUNIF
   INTEGER, INTENT(OUT)                                   :: NQ
   REAL (KIND=8), INTENT(INOUT), DIMENSION(NQ1)           :: B1,WA1
   REAL (KIND=8), INTENT(INOUT), DIMENSION(NQ1**NDIM*NDIM) :: B,WA2
   REAL (KIND=8), INTENT(INOUT), DIMENSION(NQ1**NDIM)      :: A
   REAL (KIND=8), INTENT(INOUT), OPTIONAL                 :: SCA
   INTEGER :: I,J,J1,J2,J3,K,M2,NDIV
   REAL (KIND=8) :: EPSQ,SUMW,AQT

   ! number of quadrature points, quadrature nodes & weights  

   IF (NDIM .EQ. 0) then
      NQ   = 1                           
      A(1) = 1.0d0
   ELSE
      NQ    = NQ1
      DO I=1,NQ
         B(I)=0.0D0
         A(I)=0.0D0
      END DO

      EPSQ=1.0D-8
      CALL HERMIT(B,A,NQ,EPSQ)
      IF (iunif .eq. 1) then
         DO i = 1,NQ
            A(I) = 1.0d0 / DBLE(NQ)
         END DO
      ELSEIF (iunif .eq. 2) then 
         IF (PRESENT(SCA)) THEN
            !CALL QLGAM(B,A,nq,SCA)
         ENDIF
      ENDIF
   ENDIF

   IF (NDIM .GE. 2 .AND. NDIM .LE. 8) then

      NQ  = NQ1**NDIM
      DO J=1,NQ1
         B1(J)  = B(J)
         WA1(J) = A(J)
      END DO

      J3=0
      DO J1 = 1,NDIM
         NDIV = NQ1**(J1-1)
         M2   = NQ/NDIV
         DO J2 = 1,NDIV
            DO J  = 1,M2
               K    = ((J-1)/(NQ1**(NDIM-J1)))+1
               J3   = J3 + 1
               B(J3)   = B1(K)
               WA2(J3) = WA1(K)
            END DO
         END DO
      END DO

      SUMW=0.0D0
      DO J = 1,NQ
         AQT = 1.0d0
         DO J1 = 2,NDIM
            J2  = NQ*(j1-1)+J
            AQT = AQT * WA2(J2)
         END DO
         A(J)  = WA2(J) * AQT
         SUMW  = SUMW   + A(J)
      END DO
      DO J = 1,NQ
         A(J) = A(J)/SUMW
      END DO
   ELSEIF (NDIM .gt. 8) then
      !write(6,*)' Program can only have 8 random effects'
   ELSE
   ENDIF

   RETURN
END SUBROUTINE QUADP

!*********************************************************************
! SUBROUTINE GAMMAS
!     Implement the Gamma function, used by HERMIT
!     when calculating Gauss-Hermite quadrature. 
!*********************************************************************
real(kind=8) function gam(Y)
    implicit none
    real(kind=8),intent(in)::y
    
    GAM=(((((((.035868343*Y-.193527818)*Y+.482199394)*Y- &
   .756704078)*Y+.918206857)*Y-.897056937)*Y+.988205891)*Y &
   -.577191652)*Y+1.0
    return
end function gam

REAL(kind=8) FUNCTION GAMMAS(X)
   USE INTERFACE_FP_EQUAL
   implicit none
   real(kind=8),intent(in)::x
   real(kind=8)::z,za,gam
   
   Z=X
   IF(Z .LE. 0.0D0 .OR. Z .GE. 70.0D0) THEN
      GAMMAS=0.
      !WRITE(6,*) 'Function GAMMAS(X), X outside 1-70 range', Z
        return
      !STOP
   ENDIF
   
   IF(Z .LT. 1.0D0) THEN
      GAMMAS=GAM(Z)/Z
   ELSE IF (FP_EQUAL(Z , 1.0D0)) THEN
      GAMMAS=1.
   ELSE
      ZA=1.
      DO WHILE(Z > 1)
         Z=Z-1.
         ZA=ZA*Z
      END DO
      IF(FRACTION(Z) .EQ. 0.00D0) THEN
         GAMMAS=ZA
      ELSE
         GAMMAS=ZA*GAM(Z)
      ENDIF
   END IF
   RETURN
   
END FUNCTION GAMMAS

!*********************************************************************
! SUBROUTINE PHIFN
! Calculate the probability distribution function (Intercept) for
! various distributions:
! NORMAL, LOGISTIC, Complementary Log-Log, OR Log-Log 
!   = 0       = 1                    = 2          =3 
!*********************************************************************

REAL(kind=8) FUNCTION PHIFN(Z,nfn)
    implicit none
    real(kind=8), intent(in)::Z
    integer(kind=4),intent(in)::nfn
    
    real(kind=8)::z2,ord,e,g
   
   SELECT CASE (nfn)
   
   CASE (0)    ! Normal distribution
     IF(Z.LT.-8.0D0) THEN
        PHIFN=0.000000000000001D0
     ELSEIF(Z.GT.8.0D0) THEN
        PHIFN=0.999999999999999D0
     ELSE
        Z2 = 0.0D0 - ((Z*Z)/2.0D0)
        ORD=DEXP(Z2)/2.506628275D0
        E=1.0D0/(1.0D0+0.2316418D0*DABS(Z))
        G=((((1.330274429D0*E-1.821255978D0)* &
           E+1.781477937D0)*E-0.356563782D0)*E+0.319381530D0)*E
        G=G*ORD
        IF(Z.LE.0.0D0)PHIFN=G
        IF(Z.GT.0.0D0)PHIFN=1.0D0-G
     ENDIF
     
   CASE (1)    ! Logistic distribution
     IF(Z.LT.-34.0D0) THEN
        PHIFN=0.000000000000001D0
        RETURN
     ELSEIF(Z.GT. 34.0D0) THEN
        PHIFN=0.999999999999999D0
        RETURN
     ELSE
        PHIFN = 1.0D0 / ( 1.0D0 + DEXP(0.0D0 - Z))
     ENDIF
     
   CASE (2)    ! Complementary Log-Log distribution
     PHIFN = 1.0D0 - DEXP(0.0D0 - DEXP(Z))
     
   CASE (3)    ! Log-Log distribution
     PHIFN = DEXP(0.0D0 - DEXP(Z))
   END SELECT
   
END FUNCTION PHIFN
   
!*********************************************************************
! SUBROUTINE PHIY
! Calculate the probability distribution function (ordinant) for
! various distributions:
! NORMAL, LOGISTIC, Complementary Log-Log, OR Log-Log 
!   = 0       = 1                    = 2          =3 
!*********************************************************************

real(kind=8) FUNCTION PHIY(Z,nfn)
    implicit none
    real(kind=8),intent(in)::Z
    integer(kind=4),intent(in)::nfn
    
    real(kind=8)::az,fn
   
   SELECT CASE (nfn)
   
   CASE (0)    ! Normal distribution
     AZ=Z*Z
     IF(AZ.GT.360.0D0) THEN
        PHIY=0.0D0
   !       ELSEIF(Z.LT.-10.0D0) THEN
   !          PHIY=0.0D0
     ELSE
        PHIY=(DEXP(-Z*Z/2.0D0))/2.506628275D0
     ENDIF
     
   CASE (1)    ! Logistic distribution
     FN    = 1.0D0 / ( 1.0D0 + DEXP(0.0D0 - Z))
     PHIY  = FN * (1.0D0 - FN)
     
   CASE (2)    ! Complementary Log-Log distribution
     FN    = 1.0D0 - DEXP(0.0D0 - DEXP(Z))
     PHIY  = (FN - 1.0D0 ) * (0.0D0 - DEXP(Z))
     
   CASE (3)    ! Log-Log distribution
     FN    = DEXP(0.0D0 - DEXP(Z))
     PHIY  = FN * (0.0D0 - DEXP(Z))
     
   END SELECT
   
END FUNCTION PHIY

!*********************************************************************
!     FUNCTION FP_EQUAL(A,B)
!*********************************************************************
!     This short routine does a 'safe compare' between two floating
!     point numbers.  This will get around the compiler message:
! 
! "Floating-point comparisons for equality may produce inconsistent results."
! 
!     which IS a LEGITIMATE message - it warns of a HIGH DEGREE of 
!     susceptibility to floating point roundoff errors, which should
!     be fixed!  For a quick introduction to the background, read 
!     http://www.lahey.com/float.htm   Tony Gray 12/18/00
!*********************************************************************

LOGICAL FUNCTION FP_EQUAL(A,B)
   DOUBLE PRECISION A,B
   DOUBLE PRECISION Epsilon
   PARAMETER (EPSILON = .0000005) ! required closeness of comparands
   IF (ABS(B - A) .LE. (ABS(B+A)*EPSILON)) THEN
      FP_EQUAL = .True.
   ELSE
      FP_EQUAL = .False.
   ENDIF
   
END FUNCTION FP_EQUAL

!*********************************************************************
! *****************************************************************
! ****  SUBROUTINE GETDNPLUS(DNPLUS,N,N4)                      ****
! ****                                                         ****
! ****  BUILDS D_n+ which is THE N*N x N*(N+1)/2               ****
! ****  INVERTED DUPLICATION MATRIX FROM MAGNUS (1988) page 56 ****
! ****  WITH THE PROPERTY D_n+ vec(A) = v(A)                   ****
! ****                                                         ****
! ****  where A is n x n and symmetric                         ****
! ****                                                         ****
! ****  Parameters Sent                                        ****
! ****     N = number of rows ( = # of columns) of A           ****
! ****     N4 = n*n*n*(n+1)/2                                  ****
! ****  DNPLUS = N4  INVERTED DUPLICATION MATRIX               ****
! *****************************************************************
SUBROUTINE getdnplus(dnplus,n,n4)
    implicit none
    integer,intent(in)::n,n4
    integer::i,j,row,col1,col2,n2
    real(kind=8)::dnplus(n4)
    
    n2 = n*(n+1)/2
    dnplus = 0.0
    do j=1,n
        row = n*(j-1)+j-j*(j-1)/2
        col1 = j+n*(j-1)
        dnplus(row+n2*(col1-1)) = 1    
        do i=j+1,n
            row = n*(j-1)+i-j*(j-1)/2
            col1 = i+n*(j-1)
            col2 = j+n*(i-1)
            dnplus(row+n2*(col1-1)) = .5
            dnplus(row+n2*(col2-1)) = .5
        end do
    end do
        
END SUBROUTINE getdnplus

! *****************************************************************
! ****  SUBROUTINE GETUNP(UNP,N,N4)                            ****
! ****                                                         ****
! ****  BUILDS U_n' which is THE revised N*(N+1)/2 x N*N       ****
! ****  ELIMINATION MATRIX FROM MAGNUS (1988) page 76          ****
! ****  WITH THE PROPERTY U_n' v(A) = vec A                    ****
! ****                                                         ****
! ****  where A is n x n and upper triangular                  ****
! ****                                                         ****
! ****  Parameters Sent                                        ****
! ****     N = number of rows ( = # of columns) of A           ****
! ****     N4 = n*n*n*(n+1)/2                                  ****
! ****  UNP = N4  ELIMINATION MATRIX                           ****
! *****************************************************************
SUBROUTINE getUnp(unp,n,n4)
    implicit none
    integer,intent(in)::n,n4
    integer::i,j,row,col
    real(kind=8)::unp(n4)
    
    unp = 0
    do i=1,n
        do j=i,n
            col = i+j*(j-1)/2
            row = i+n*(j-1)
            unp(row+n*n*(col-1)) = 1
        end do
    end do
END SUBROUTINE getunp

subroutine getInKSprime(InKS,S,n,nStar,n4)
    implicit none
    integer,intent(in)::n,n4,nStar
    integer::row,n2,num,indent,sstart,ksstart
    real(kind=8)::InKS(n4)
    real(kind=8),intent(in)::S(nStar)
    
    InKS = 0
    n2 = n*n
    do row = 1, n2
        num = mod(row-1, n)+1
        indent = (row - 1) / n
        sstart = num*(num-1)/2
        ksstart = n2*(row-1)+indent*n
        InKS(ksstart+1:ksstart+num) = S(sstart+1:sstart+num)
    end do
end subroutine getInKSprime

subroutine getSStar(vechS, n, nStar, sStar)
    USE INTERFACE_MPYM
    implicit none
    integer,intent(in)::n,nStar
    integer::n2,n2nStar,n4!,i
    real(kind=8),intent(in)::vechS(nStar)
    real(kind=8)::sStar(nStar*nStar)
    real(kind=8),allocatable::dnplus(:),unp(:),InKS(:),InKSprime(:),work(:)

    n2 = n*n
    n2nStar = n2 * nStar
    n4 = n2 * n2
    allocate(dnplus(n2nStar))
    allocate(unp(n2nStar))
    allocate(InKS(n4))
    allocate(InKSprime(n4))
    allocate(work(n2nStar))

    call getdnplus(dnplus,n,n2nStar)
    call getInKSprime(InKSprime,vechS,n,nStar,n4)
    call trp(InKSprime,InKS,n2,n2)
    call mpym(dnplus,InKS,work,nStar,n2,0,0,n2)
    call getunp(unp,n,n2nStar)
    call mpym(work,unp,sStar,nStar,n2,0,0,nStar)

    deallocate(dnplus)
    deallocate(unp)
    deallocate(InKS)
    deallocate(InKSprime)
    deallocate(work)
end subroutine getSStar

SUBROUTINE TRP(A,B,MA,NA)   
    implicit none
    integer::ma,na,i,j
    real(kind=8)::a(ma,na),b(na,ma)
    do i=1,na
        do j=1,ma
            b(i,j) = a(j, i)
        end do
    end do
end subroutine trp                                                          

!Calculates ratio of Standard Normal pdfs PHI(newB)/PHI(origB)
subroutine GET_PHI_RATIO(newB,origB, phiRatio)
    implicit none
    REAL(KIND=8)::newB,origB,phiratio
    PHIRATIO = exp((origB*origB-newB*newB)/2)
END subroutine GET_PHI_RATIO

!Pulls necessary Standard Deviation from Packed Variance matrix
subroutine getSDev(n, length, Var, sdev)
    implicit none
    integer::n, length, i, j, place
    real(kind=8)::Var(length*(length+1)/2), sdev
    place = 1
    do i=1,length
        do j=1,i
            if(i==j .and. i==n) sdev = dsqrt(Var(place))
            place = place + 1
        end do
    end do
end subroutine getSDev