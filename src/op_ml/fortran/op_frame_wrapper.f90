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

      OpNum=OpNum+1
      OrderParameterType(OpNum)%name='B'; OrderParameterType(OpNum)%iarg(1)=n1; OrderParameterType(OpNum)%iarg(2)=n2
      OrderParameterType(OpNum)%arg(1)=PI/3._PR

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
