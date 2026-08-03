!==============================================================================
! op_frame_wrapper.f90
!
! f2py-compatible interface to the order_parameters Fortran module.
! Exposes two subroutines to Python:
!
!   init_op_calc(n_mol)
!       Allocates module arrays and defines all 383 OP types.
!       Call once per process before any compute_frame calls.
!
!   compute_frame(r_central, r_list, mol_list, lat_vecs, rlat_vecs,
!                 rcutsq, n_mol, n_list, local_ops, avg_ops)
!       Compute all 383 local and averaged OPs for one trajectory frame.
!       Thread-safe per MPI rank (module variables are per-process).
!
! Build with (see Makefile.f2py):
!   python -m numpy.f2py -c consts.f90 spherical_harmonics.f90 \
!          order_parameters.f90 op_frame_wrapper.f90 \
!          -m _op_fortran --f90exec=mpiifx --f90flags="-O3"
!==============================================================================

subroutine dealloc_op_calc()
  use order_parameters, only: OrderParameterType, OrderParameterValue, N, N2, &
       q2m, q3m, q4m, q5m, q6m, q7m, q8m, q9m, q10m, q12m, q14m, q16m, &
       q2m_avg, q3m_avg, q4m_avg, q5m_avg, q6m_avg, q7m_avg, q8m_avg, &
       q9m_avg, q10m_avg, q12m_avg, q14m_avg, q16m_avg, &
       lq2m, lq3m, lq4m, lq5m, lq6m, lq7m, lq8m, lq9m, lq10m, lq12m, lq14m, lq16m, &
       q2, q3, q4, q5, q6, q7, q8, q9, q10, q12, q14, q16, &
       q2_avg, q3_avg, q4_avg, q5_avg, q6_avg, q7_avg, q8_avg, &
       q9_avg, q10_avg, q12_avg, q14_avg, q16_avg, &
       lq2, lq3, lq4, lq5, lq6, lq7, lq8, lq9, lq10, lq12, lq14, lq16
  implicit none

  if(allocated(OrderParameterType))  deallocate(OrderParameterType)
  if(allocated(OrderParameterValue)) deallocate(OrderParameterValue)
  if(allocated(N))  deallocate(N)
  if(allocated(N2)) deallocate(N2)
  if(allocated(q2m))  deallocate(q2m);  if(allocated(q3m))  deallocate(q3m)
  if(allocated(q4m))  deallocate(q4m);  if(allocated(q5m))  deallocate(q5m)
  if(allocated(q6m))  deallocate(q6m);  if(allocated(q7m))  deallocate(q7m)
  if(allocated(q8m))  deallocate(q8m);  if(allocated(q9m))  deallocate(q9m)
  if(allocated(q10m)) deallocate(q10m); if(allocated(q12m)) deallocate(q12m)
  if(allocated(q14m)) deallocate(q14m); if(allocated(q16m)) deallocate(q16m)
  if(allocated(q2m_avg))  deallocate(q2m_avg);  if(allocated(q3m_avg))  deallocate(q3m_avg)
  if(allocated(q4m_avg))  deallocate(q4m_avg);  if(allocated(q5m_avg))  deallocate(q5m_avg)
  if(allocated(q6m_avg))  deallocate(q6m_avg);  if(allocated(q7m_avg))  deallocate(q7m_avg)
  if(allocated(q8m_avg))  deallocate(q8m_avg);  if(allocated(q9m_avg))  deallocate(q9m_avg)
  if(allocated(q10m_avg)) deallocate(q10m_avg); if(allocated(q12m_avg)) deallocate(q12m_avg)
  if(allocated(q14m_avg)) deallocate(q14m_avg); if(allocated(q16m_avg)) deallocate(q16m_avg)
  if(allocated(lq2m))  deallocate(lq2m);  if(allocated(lq3m))  deallocate(lq3m)
  if(allocated(lq4m))  deallocate(lq4m);  if(allocated(lq5m))  deallocate(lq5m)
  if(allocated(lq6m))  deallocate(lq6m);  if(allocated(lq7m))  deallocate(lq7m)
  if(allocated(lq8m))  deallocate(lq8m);  if(allocated(lq9m))  deallocate(lq9m)
  if(allocated(lq10m)) deallocate(lq10m); if(allocated(lq12m)) deallocate(lq12m)
  if(allocated(lq14m)) deallocate(lq14m); if(allocated(lq16m)) deallocate(lq16m)
  if(allocated(q2))  deallocate(q2);  if(allocated(q3))  deallocate(q3)
  if(allocated(q4))  deallocate(q4);  if(allocated(q5))  deallocate(q5)
  if(allocated(q6))  deallocate(q6);  if(allocated(q7))  deallocate(q7)
  if(allocated(q8))  deallocate(q8);  if(allocated(q9))  deallocate(q9)
  if(allocated(q10)) deallocate(q10); if(allocated(q12)) deallocate(q12)
  if(allocated(q14)) deallocate(q14); if(allocated(q16)) deallocate(q16)
  if(allocated(q2_avg))  deallocate(q2_avg);  if(allocated(q3_avg))  deallocate(q3_avg)
  if(allocated(q4_avg))  deallocate(q4_avg);  if(allocated(q5_avg))  deallocate(q5_avg)
  if(allocated(q6_avg))  deallocate(q6_avg);  if(allocated(q7_avg))  deallocate(q7_avg)
  if(allocated(q8_avg))  deallocate(q8_avg);  if(allocated(q9_avg))  deallocate(q9_avg)
  if(allocated(q10_avg)) deallocate(q10_avg); if(allocated(q12_avg)) deallocate(q12_avg)
  if(allocated(q14_avg)) deallocate(q14_avg); if(allocated(q16_avg)) deallocate(q16_avg)
  if(allocated(lq2))  deallocate(lq2);  if(allocated(lq3))  deallocate(lq3)
  if(allocated(lq4))  deallocate(lq4);  if(allocated(lq5))  deallocate(lq5)
  if(allocated(lq6))  deallocate(lq6);  if(allocated(lq7))  deallocate(lq7)
  if(allocated(lq8))  deallocate(lq8);  if(allocated(lq9))  deallocate(lq9)
  if(allocated(lq10)) deallocate(lq10); if(allocated(lq12)) deallocate(lq12)
  if(allocated(lq14)) deallocate(lq14); if(allocated(lq16)) deallocate(lq16)

end subroutine dealloc_op_calc

!==============================================================================

subroutine init_op_calc(n_mol_in)
  use consts, only: PR, PI
  use order_parameters, only: AllocateMemoryToOrderParameters, &
                               NumberOfOrderParameters, OrderParameterType

  implicit none
  integer, intent(in) :: n_mol_in
  !f2py intent(in) n_mol_in

  integer :: OpNum, n1, n2
  real(PR) :: phi0

  ! Deallocate first in case of re-initialization with different n_mol
  call dealloc_op_calc()

  call AllocateMemoryToOrderParameters(n_mol_in, 383)

  phi0  = 180._PR / 109.5_PR
  OpNum = 0

  !-- B parameters: 7 phi values x n1(1..2) x n2(1..3) = 42 total
  do n1 = 1, 2
    do n2 = 1, 3
      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='B'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=0._PR

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='B'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=2._PR/3._PR*PI

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='B'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=PI/2._PR

      ! NOTE: 4th B angle is PI/4 (NOT PI/3) to reproduce the reference
      ! features_SC / original code/hydrate_*.f90, which define this slot as a
      ! DUPLICATE PI/4 (the column header is still labelled "1_05" there, an
      ! upstream mislabel that we reproduce verbatim). Change to PI/3 only if
      ! you intentionally want 7 distinct B angles instead of matching the data.
      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='B'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=PI/4._PR

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='B'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=PI/4._PR

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='B'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=PI/5._PR

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='B'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=PI/6._PR
    end do
  end do

  !-- D parameters: na=nb=n1(1..5), nc=n2(1..5) = 25 total
  do n1 = 1, 5
    do n2 = 1, 5
      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='D'
      OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n1
      OrderParameterType(OpNum)%iarg(3)=n2
    end do
  end do

  !-- F parameters: na,nb(1..5), a(11 values) = 275 total
  do n1 = 1, 5
    do n2 = 1, 5
      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='F'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=1._PR

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='F'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=2._PR

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='F'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=3._PR

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='F'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=4._PR

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='F'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=6._PR

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='F'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=8._PR

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='F'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=phi0

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='F'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=2._PR*phi0

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='F'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=3._PR*phi0

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='F'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=4._PR*phi0

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='F'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=6._PR*phi0
    end do
  end do

  !-- I parameter (1 total)
  OpNum=OpNum+1
  OrderParameterType(OpNum)%name='I'

  !-- Q, W, LQ, LW for l = 2..6 (20 total)
  do n1 = 2, 6
    OpNum=OpNum+1; OrderParameterType(OpNum)%name='Q';  OrderParameterType(OpNum)%iarg(1)=n1
    OpNum=OpNum+1; OrderParameterType(OpNum)%name='W';  OrderParameterType(OpNum)%iarg(1)=n1
    OpNum=OpNum+1; OrderParameterType(OpNum)%name='LQ'; OrderParameterType(OpNum)%iarg(1)=n1
    OpNum=OpNum+1; OrderParameterType(OpNum)%name='LW'; OrderParameterType(OpNum)%iarg(1)=n1
  end do

  !-- Q, W, LQ, LW for l = 8,10,12,14,16 (20 total)
  do n1 = 8, 16, 2
    OpNum=OpNum+1; OrderParameterType(OpNum)%name='Q';  OrderParameterType(OpNum)%iarg(1)=n1
    OpNum=OpNum+1; OrderParameterType(OpNum)%name='W';  OrderParameterType(OpNum)%iarg(1)=n1
    OpNum=OpNum+1; OrderParameterType(OpNum)%name='LQ'; OrderParameterType(OpNum)%iarg(1)=n1
    OpNum=OpNum+1; OrderParameterType(OpNum)%name='LW'; OrderParameterType(OpNum)%iarg(1)=n1
  end do

end subroutine init_op_calc

!==============================================================================

subroutine compute_frame(r_central, r_list, mol_list_in, lat_vecs, rlat_vecs, &
                         rcutsq, n_mol, n_list, local_ops, avg_ops)
  use consts, only: PR
  use order_parameters, only: ResetOPAccumulators, &
                               ComputeConnectionMatrix, ComputeOrderParameters, &
                               ComputeUnaveragedOrderParameters, &
                               ComputeAveragedOrderParameters, &
                               OrderParameterValue, NumberOfMolecules

  implicit none

  integer,  intent(in)  :: n_mol, n_list
  real(PR), intent(in)  :: r_central(3, n_mol)
  real(PR), intent(in)  :: r_list(3, n_list)
  integer,  intent(in)  :: mol_list_in(n_list)
  real(PR), intent(in)  :: lat_vecs(3, 3)
  real(PR), intent(in)  :: rlat_vecs(3, 3)
  real(PR), intent(in)  :: rcutsq
  real(PR), intent(out) :: local_ops(383, n_mol)
  real(PR), intent(out) :: avg_ops(383, n_mol)

  !f2py intent(in)  :: r_central, r_list, mol_list_in, lat_vecs, rlat_vecs, rcutsq
  !f2py intent(hide) :: n_mol, n_list
  !f2py intent(out)  :: local_ops, avg_ops

  integer, dimension(10) :: SteinhardtQList
  logical, allocatable   :: conn(:,:)
  integer :: i, mol

  SteinhardtQList = (/2, 3, 4, 5, 6, 8, 10, 12, 14, 16/)

  allocate(conn(n_mol, n_mol))

  call ResetOPAccumulators
  call ComputeConnectionMatrix(r_central, lat_vecs, rlat_vecs, rcutsq, conn)
  call ComputeOrderParameters(r_central, r_list, mol_list_in, conn, &
                               lat_vecs, rlat_vecs, SteinhardtQList)
  call ComputeUnaveragedOrderParameters(SteinhardtQList)
  call ComputeAveragedOrderParameters(SteinhardtQList, SteinhardtQList, conn)

  do mol = 1, n_mol
    do i = 1, 383
      local_ops(i, mol) = OrderParameterValue(i, mol)%local
      avg_ops(i, mol)   = OrderParameterValue(i, mol)%avg
    end do
  end do

  deallocate(conn)

end subroutine compute_frame

!==============================================================================
! compute_frame_partial
!
! Like compute_frame, but the connection matrix is built over a larger set of
! "centres" (n_mol) than the number of molecules whose OPs are computed and
! returned (n_cen, with n_cen <= n_mol).  This reproduces the gas-hydrate
! convention of code/hydrate_OHM.f90:
!
!   - r_mol      : centre of every molecule (water O + guest M), size n_mol;
!                  used ONLY to build the n_mol x n_mol connection matrix, so a
!                  central water O "sees" a guest molecule as a neighbour.
!   - r_central  : central-atom positions of the FIRST n_cen molecules (water
!                  O); used as r1 in ComputeOrderParameters.
!   - r_list     : neighbour atoms (O,H of waters + M of guests) with mol_list
!                  mapping each to its molecule index in 1..n_mol.
!
! init_op_calc(n_cen) must be called first, so the module NumberOfMolecules =
! n_cen.  The (un)averaged loops then run over 1..n_cen only - guest molecules
! contribute to a water O's LOCAL Steinhardt (as neighbours via the connection
! matrix) but are excluded from the averaging set, exactly as in the reference.
!==============================================================================
subroutine compute_frame_partial(r_central, r_mol, r_list, mol_list_in, &
                                  lat_vecs, rlat_vecs, rcutsq, &
                                  n_cen, n_mol, n_list, local_ops, avg_ops)
  use consts, only: PR
  use order_parameters, only: ResetOPAccumulators, &
                               ComputeConnectionMatrix, ComputeOrderParameters, &
                               ComputeUnaveragedOrderParameters, &
                               ComputeAveragedOrderParameters, OrderParameterValue

  implicit none

  integer,  intent(in)  :: n_cen, n_mol, n_list
  real(PR), intent(in)  :: r_central(3, n_cen)
  real(PR), intent(in)  :: r_mol(3, n_mol)
  real(PR), intent(in)  :: r_list(3, n_list)
  integer,  intent(in)  :: mol_list_in(n_list)
  real(PR), intent(in)  :: lat_vecs(3, 3)
  real(PR), intent(in)  :: rlat_vecs(3, 3)
  real(PR), intent(in)  :: rcutsq
  real(PR), intent(out) :: local_ops(383, n_cen)
  real(PR), intent(out) :: avg_ops(383, n_cen)

  !f2py intent(in)  :: r_central, r_mol, r_list, mol_list_in, lat_vecs, rlat_vecs, rcutsq
  !f2py intent(hide) :: n_cen, n_mol, n_list
  !f2py intent(out)  :: local_ops, avg_ops

  integer, dimension(10) :: SteinhardtQList
  logical, allocatable   :: conn(:,:)
  integer :: i, mol

  SteinhardtQList = (/2, 3, 4, 5, 6, 8, 10, 12, 14, 16/)

  allocate(conn(n_mol, n_mol))

  call ResetOPAccumulators
  call ComputeConnectionMatrix(r_mol, lat_vecs, rlat_vecs, rcutsq, conn)
  call ComputeOrderParameters(r_central, r_list, mol_list_in, conn, &
                               lat_vecs, rlat_vecs, SteinhardtQList)
  call ComputeUnaveragedOrderParameters(SteinhardtQList)
  call ComputeAveragedOrderParameters(SteinhardtQList, SteinhardtQList, conn)

  do mol = 1, n_cen
    do i = 1, 383
      local_ops(i, mol) = OrderParameterValue(i, mol)%local
      avg_ops(i, mol)   = OrderParameterValue(i, mol)%avg
    end do
  end do

  deallocate(conn)

end subroutine compute_frame_partial

!==============================================================================
! 2D (frame x atom/molecule) MPI decomposition entry points.
!
! One trajectory frame is computed cooperatively by P MPI ranks that share a
! sub-communicator.  Each rank owns a disjoint slice [mol_lo,mol_hi] of the
! n_cen central molecules.  The work is split into two stages that bracket a
! single MPI all-reduce in Python:
!
!   Stage 1  compute_local_range : build the FULL connection matrix, then
!            compute the LOCAL order parameters and the per-molecule Steinhardt
!            q_lm (already normalised) for molecules in the rank's slice only.
!            Out-of-slice columns are returned as ZERO so that an element-wise
!            MPI_SUM all-reduce across the sub-communicator reconstructs the
!            full per-molecule arrays EXACTLY (x + 0 + 0 ... = x in IEEE).
!
!   (Python all-reduces local_ops, qlm, qnorm over the sub-communicator.)
!
!   Stage 2  compute_avg_range   : inject the full per-molecule local/q_lm/qnorm
!            arrays back into the module, rebuild the connection matrix, then
!            compute the AVERAGED (Lechner-Dellago) order parameters for the
!            rank's slice only.  Each mol1 averages over its own neighbour set
!            in the same fixed order as the serial code, so the slice result is
!            bit-identical to the full serial computation.
!
! Both OO (n_mol == n_cen, r_mol == r_central) and the hydrate "partial" case
! (n_mol > n_cen) are handled by the same routines.  init_op_calc(n_cen) must
! have been called first.  The packed q_lm layout (170 complex rows) is:
!   l = 2 :  1.. 5    l = 6 : 33.. 45    l =12 : 84..108
!   l = 3 :  6.. 12   l = 8 : 46.. 62    l =14 :109..137
!   l = 4 : 13.. 21   l =10 : 63.. 83    l =16 :138..170
!   l = 5 : 22.. 32
! qnorm rows 1..10 correspond to l = 2,3,4,5,6,8,10,12,14,16.
!==============================================================================
subroutine compute_local_range(mol_lo, mol_hi, r_central, r_mol, r_list, &
                                mol_list_in, lat_vecs, rlat_vecs, rcutsq, &
                                n_cen, n_mol, n_list, local_ops, qlm_out, qnorm_out)
  use consts, only: PR
  use order_parameters, only: ResetOPAccumulators, ComputeConnectionMatrix, &
       ComputeOrderParameters, ComputeUnaveragedOrderParameters, OrderParameterValue, &
       q2m, q3m, q4m, q5m, q6m, q8m, q10m, q12m, q14m, q16m, &
       q2, q3, q4, q5, q6, q8, q10, q12, q14, q16

  implicit none

  integer,  intent(in)  :: mol_lo, mol_hi, n_cen, n_mol, n_list
  real(PR), intent(in)  :: r_central(3, n_cen)
  real(PR), intent(in)  :: r_mol(3, n_mol)
  real(PR), intent(in)  :: r_list(3, n_list)
  integer,  intent(in)  :: mol_list_in(n_list)
  real(PR), intent(in)  :: lat_vecs(3, 3)
  real(PR), intent(in)  :: rlat_vecs(3, 3)
  real(PR), intent(in)  :: rcutsq
  real(PR),    intent(out) :: local_ops(383, n_cen)
  complex(PR), intent(out) :: qlm_out(170, n_cen)
  real(PR),    intent(out) :: qnorm_out(10, n_cen)

  !f2py intent(in)   :: mol_lo, mol_hi, r_central, r_mol, r_list, mol_list_in
  !f2py intent(in)   :: lat_vecs, rlat_vecs, rcutsq
  !f2py intent(hide) :: n_cen, n_mol, n_list
  !f2py intent(out)  :: local_ops, qlm_out, qnorm_out

  integer, dimension(10) :: SteinhardtQList
  logical, allocatable   :: conn(:,:)
  integer :: i, mol

  SteinhardtQList = (/2, 3, 4, 5, 6, 8, 10, 12, 14, 16/)

  allocate(conn(n_mol, n_mol))

  call ResetOPAccumulators
  call ComputeConnectionMatrix(r_mol, lat_vecs, rlat_vecs, rcutsq, conn)
  call ComputeOrderParameters(r_central, r_list, mol_list_in, conn, &
                              lat_vecs, rlat_vecs, SteinhardtQList, mol_lo, mol_hi)
  call ComputeUnaveragedOrderParameters(SteinhardtQList, mol_lo, mol_hi)

  local_ops = 0._PR
  qlm_out   = (0._PR, 0._PR)
  qnorm_out = 0._PR

  do mol = mol_lo, mol_hi
    do i = 1, 383
      local_ops(i, mol) = OrderParameterValue(i, mol)%local
    end do
    qlm_out(  1:  5, mol) = q2m(:,  mol)
    qlm_out(  6: 12, mol) = q3m(:,  mol)
    qlm_out( 13: 21, mol) = q4m(:,  mol)
    qlm_out( 22: 32, mol) = q5m(:,  mol)
    qlm_out( 33: 45, mol) = q6m(:,  mol)
    qlm_out( 46: 62, mol) = q8m(:,  mol)
    qlm_out( 63: 83, mol) = q10m(:, mol)
    qlm_out( 84:108, mol) = q12m(:, mol)
    qlm_out(109:137, mol) = q14m(:, mol)
    qlm_out(138:170, mol) = q16m(:, mol)
    qnorm_out(1, mol)  = q2(mol);  qnorm_out(2, mol)  = q3(mol)
    qnorm_out(3, mol)  = q4(mol);  qnorm_out(4, mol)  = q5(mol)
    qnorm_out(5, mol)  = q6(mol);  qnorm_out(6, mol)  = q8(mol)
    qnorm_out(7, mol)  = q10(mol); qnorm_out(8, mol)  = q12(mol)
    qnorm_out(9, mol)  = q14(mol); qnorm_out(10, mol) = q16(mol)
  end do

  deallocate(conn)

end subroutine compute_local_range

!==============================================================================
subroutine compute_avg_range(mol_lo, mol_hi, r_mol, lat_vecs, rlat_vecs, rcutsq, &
                              local_in, qlm_in, qnorm_in, n_cen, n_mol, &
                              local_ops, avg_ops)
  use consts, only: PR
  use order_parameters, only: ComputeConnectionMatrix, ComputeAveragedOrderParameters, &
       OrderParameterValue, &
       q2m, q3m, q4m, q5m, q6m, q8m, q10m, q12m, q14m, q16m, &
       q2, q3, q4, q5, q6, q8, q10, q12, q14, q16, &
       q2m_avg, q3m_avg, q4m_avg, q5m_avg, q6m_avg, q8m_avg, q10m_avg, q12m_avg, q14m_avg, q16m_avg, &
       lq2m, lq3m, lq4m, lq5m, lq6m, lq8m, lq10m, lq12m, lq14m, lq16m

  implicit none

  integer,  intent(in)  :: mol_lo, mol_hi, n_cen, n_mol
  real(PR), intent(in)  :: r_mol(3, n_mol)
  real(PR), intent(in)  :: lat_vecs(3, 3)
  real(PR), intent(in)  :: rlat_vecs(3, 3)
  real(PR), intent(in)  :: rcutsq
  real(PR),    intent(in)  :: local_in(383, n_cen)
  complex(PR), intent(in)  :: qlm_in(170, n_cen)
  real(PR),    intent(in)  :: qnorm_in(10, n_cen)
  real(PR),    intent(out) :: local_ops(383, n_cen)
  real(PR),    intent(out) :: avg_ops(383, n_cen)

  !f2py intent(in)   :: mol_lo, mol_hi, r_mol, lat_vecs, rlat_vecs, rcutsq
  !f2py intent(in)   :: local_in, qlm_in, qnorm_in
  !f2py intent(hide) :: n_cen, n_mol
  !f2py intent(out)  :: local_ops, avg_ops

  integer, dimension(10) :: SteinhardtQList
  logical, allocatable   :: conn(:,:)
  integer :: i, mol

  SteinhardtQList = (/2, 3, 4, 5, 6, 8, 10, 12, 14, 16/)

  allocate(conn(n_mol, n_mol))
  call ComputeConnectionMatrix(r_mol, lat_vecs, rlat_vecs, rcutsq, conn)

  ! Inject the FULL (already all-reduced) per-molecule state for every central
  ! molecule, since averaging mol1 reads its neighbours' local/q_lm/qnorm.
  ! Zero the averaging accumulators (q_lm_avg, lq_lm, %avg) WITHOUT touching the
  ! injected q_lm (so we must NOT call ResetOPAccumulators here).
  do mol = 1, n_cen
    do i = 1, 383
      OrderParameterValue(i, mol)%local = local_in(i, mol)
      OrderParameterValue(i, mol)%avg   = 0._PR
    end do
    q2m(:,  mol) = qlm_in(  1:  5, mol)
    q3m(:,  mol) = qlm_in(  6: 12, mol)
    q4m(:,  mol) = qlm_in( 13: 21, mol)
    q5m(:,  mol) = qlm_in( 22: 32, mol)
    q6m(:,  mol) = qlm_in( 33: 45, mol)
    q8m(:,  mol) = qlm_in( 46: 62, mol)
    q10m(:, mol) = qlm_in( 63: 83, mol)
    q12m(:, mol) = qlm_in( 84:108, mol)
    q14m(:, mol) = qlm_in(109:137, mol)
    q16m(:, mol) = qlm_in(138:170, mol)
    q2(mol)  = qnorm_in(1, mol);  q3(mol)  = qnorm_in(2, mol)
    q4(mol)  = qnorm_in(3, mol);  q5(mol)  = qnorm_in(4, mol)
    q6(mol)  = qnorm_in(5, mol);  q8(mol)  = qnorm_in(6, mol)
    q10(mol) = qnorm_in(7, mol);  q12(mol) = qnorm_in(8, mol)
    q14(mol) = qnorm_in(9, mol);  q16(mol) = qnorm_in(10, mol)
    q2m_avg(:,  mol) = (0._PR, 0._PR); q3m_avg(:,  mol) = (0._PR, 0._PR)
    q4m_avg(:,  mol) = (0._PR, 0._PR); q5m_avg(:,  mol) = (0._PR, 0._PR)
    q6m_avg(:,  mol) = (0._PR, 0._PR); q8m_avg(:,  mol) = (0._PR, 0._PR)
    q10m_avg(:, mol) = (0._PR, 0._PR); q12m_avg(:, mol) = (0._PR, 0._PR)
    q14m_avg(:, mol) = (0._PR, 0._PR); q16m_avg(:, mol) = (0._PR, 0._PR)
    lq2m(:,  mol) = (0._PR, 0._PR); lq3m(:,  mol) = (0._PR, 0._PR)
    lq4m(:,  mol) = (0._PR, 0._PR); lq5m(:,  mol) = (0._PR, 0._PR)
    lq6m(:,  mol) = (0._PR, 0._PR); lq8m(:,  mol) = (0._PR, 0._PR)
    lq10m(:, mol) = (0._PR, 0._PR); lq12m(:, mol) = (0._PR, 0._PR)
    lq14m(:, mol) = (0._PR, 0._PR); lq16m(:, mol) = (0._PR, 0._PR)
  end do

  call ComputeAveragedOrderParameters(SteinhardtQList, SteinhardtQList, conn, mol_lo, mol_hi)

  local_ops = 0._PR
  avg_ops   = 0._PR
  do mol = mol_lo, mol_hi
    do i = 1, 383
      local_ops(i, mol) = OrderParameterValue(i, mol)%local
      avg_ops(i, mol)   = OrderParameterValue(i, mol)%avg
    end do
  end do

  deallocate(conn)

end subroutine compute_avg_range
