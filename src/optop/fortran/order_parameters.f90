module order_parameters
  use consts, only: PR, PI
  use spherical_harmonics, only: sp_hrmcs, SteinhardtW
  implicit none
  public
  save

  type :: OrderParamterInfo
    character(len=10) :: name
    integer, dimension(5) :: iarg
    real(PR), dimension(5) :: arg
  end type OrderParamterInfo

  type :: OrderParameter
    real(PR) :: local, avg
  end type OrderParameter

  type(OrderParamterInfo), dimension(:), allocatable :: OrderParameterType
  type(OrderParameter), dimension(:,:), allocatable :: OrderParameterValue
  complex(PR), dimension(:,:), allocatable :: q2m, q3m, q4m, q5m, q6m, q7m, q8m, q9m, q10m, q12m, q14m, q16m
  complex(PR), dimension(:,:), allocatable :: q2m_avg, q3m_avg, q4m_avg, q5m_avg, q6m_avg, q7m_avg, q8m_avg, q9m_avg, q10m_avg, q12m_avg, q14m_avg, q16m_avg
  complex(PR), dimension(:,:), allocatable :: lq2m, lq3m, lq4m, lq5m, lq6m, lq7m, lq8m, lq9m, lq10m, lq12m, lq14m, lq16m
  real(PR), dimension(:), allocatable :: q2, q3, q4, q5, q6, q7, q8, q9, q10, q12, q14, q16
  real(PR), dimension(:), allocatable :: q2_avg, q3_avg, q4_avg, q5_avg, q6_avg, q7_avg, q8_avg, q9_avg, q10_avg, q12_avg, q14_avg, q16_avg
  real(PR), dimension(:), allocatable :: lq2, lq3, lq4, lq5, lq6, lq7, lq8, lq9, lq10, lq12, lq14, lq16

  integer, dimension(:), allocatable :: N,N2
  integer :: NumberOfMolecules,NumberOfOrderParameters

contains
  !=========================================================================================================================
  subroutine AllocateMemoryToOrderParameters(n_1,n_2)
    integer, intent(in) :: n_1,n_2

    NumberOfMolecules=n_1
    NumberOfOrderParameters=n_2

    !** Allocate Memory to arrays
    allocate(OrderParameterType(n_2),OrderParameterValue(n_2,n_1),N(n_1),N2(n_1))
    allocate(q2m(-2:2,n_1), q3m(-3:3,n_1), q4m(-4:4,n_1), &
             q5m(-5:5,n_1), q6m(-6:6,n_1), q7m(-7:7,n_1), &
             q8m(-8:8,n_1), q9m(-9:9,n_1), q10m(-10:10,n_1), &
             q12m(-12:12,n_1), q14m(-14:14,n_1), q16m(-16:16,n_1))
    allocate(q2m_avg(-2:2,n_1), q3m_avg(-3:3,n_1), q4m_avg(-4:4,n_1), &
             q5m_avg(-5:5,n_1), q6m_avg(-6:6,n_1), q7m_avg(-7:7,n_1), &
             q8m_avg(-8:8,n_1), q9m_avg(-9:9,n_1), q10m_avg(-10:10,n_1), &
             q12m_avg(-12:12,n_1), q14m_avg(-14:14,n_1), q16m_avg(-16:16,n_1))
    allocate(lq2m(-2:2,n_1), lq3m(-3:3,n_1), lq4m(-4:4,n_1), &
             lq5m(-5:5,n_1), lq6m(-6:6,n_1), lq7m(-7:7,n_1), &
             lq8m(-8:8,n_1), lq9m(-9:9,n_1), lq10m(-10:10,n_1), &
             lq12m(-12:12,n_1), lq14m(-14:14,n_1), lq16m(-16:16,n_1))
    allocate(q2(n_1), q3(n_1), q4(n_1), &
             q5(n_1), q6(n_1), q7(n_1), &
             q8(n_1), q9(n_1), q10(n_1), &
             q12(n_1), q14(n_1), q16(n_1))
    allocate(q2_avg(n_1), q3_avg(n_1), q4_avg(n_1), &
             q5_avg(n_1), q6_avg(n_1), q7_avg(n_1), &
             q8_avg(n_1), q9_avg(n_1), q10_avg(n_1), &
             q12_avg(n_1), q14_avg(n_1), q16_avg(n_1))
    allocate(lq2(n_1), lq3(n_1), lq4(n_1), &
             lq5(n_1), lq6(n_1), lq7(n_1), &
             lq8(n_1), lq9(n_1), lq10(n_1), &
             lq12(n_1), lq14(n_1), lq16(n_1))

  end subroutine AllocateMemoryToOrderParameters

  subroutine ResetOPAccumulators
    !** Initialize values
    OrderParameterValue%local=0._PR
    OrderParameterValue%avg=0._PR
    N=0
    N2=0
    q2m=(0._PR , 0._PR)
    q3m=(0._PR , 0._PR)
    q4m=(0._PR , 0._PR)
    q5m=(0._PR , 0._PR)
    q6m=(0._PR , 0._PR)
    q7m=(0._PR , 0._PR)
    q8m=(0._PR , 0._PR)
    q9m=(0._PR , 0._PR)
    q10m=(0._PR , 0._PR)
    q12m=(0._PR , 0._PR)
    q14m=(0._PR , 0._PR)
    q16m=(0._PR , 0._PR)
    q2m_avg=(0._PR , 0._PR)
    q3m_avg=(0._PR , 0._PR)
    q4m_avg=(0._PR , 0._PR)
    q5m_avg=(0._PR , 0._PR)
    q6m_avg=(0._PR , 0._PR)
    q7m_avg=(0._PR , 0._PR)
    q8m_avg=(0._PR , 0._PR)
    q9m_avg=(0._PR , 0._PR)
    q10m_avg=(0._PR , 0._PR)
    q12m_avg=(0._PR , 0._PR)
    q14m_avg=(0._PR , 0._PR)
    q16m_avg=(0._PR , 0._PR)
    lq2m=(0._PR , 0._PR)
    lq3m=(0._PR , 0._PR)
    lq4m=(0._PR , 0._PR)
    lq5m=(0._PR , 0._PR)
    lq6m=(0._PR , 0._PR)
    lq7m=(0._PR , 0._PR)
    lq8m=(0._PR , 0._PR)
    lq9m=(0._PR , 0._PR)
    lq10m=(0._PR , 0._PR)
    lq12m=(0._PR , 0._PR)
    lq14m=(0._PR , 0._PR)
    lq16m=(0._PR , 0._PR)
             
  end subroutine ResetOPAccumulators

  !=========================================================================================================================
  subroutine ComputeConnectionMatrix(r,LatVecs,RLatVecs,rcutsq,ConnectionMatrix)
    real(PR), dimension(:,:), intent(in) :: r
    real(PR), dimension(3,3), intent(in) :: LatVecs, RLatVecs
    real(PR), intent(in) :: rcutsq
    logical, dimension(:,:), intent(out) :: ConnectionMatrix

    integer :: mol1, mol2
    real(PR) :: rij(3), rijsq

    !** Compute Connection Matrix
    ConnectionMatrix=.false.
    do mol1=1,size(r,2)
      !** Connection Matrix between same species
      ConnectionMatrix(mol1,mol1)=.true.
      do mol2=mol1+1,size(r,2)
        rij=r(:,mol1)-r(:,mol2)
        call ApplyPBC(rij,LatVecs,RLatVecs)
        rijsq=rij(1)**2+rij(2)**2+rij(3)**2
        if(rijsq < rcutsq)then
          ConnectionMatrix(mol1,mol2)=.true.
          ConnectionMatrix(mol2,mol1)=.true.
        end if
      end do
    end do

  end subroutine ComputeConnectionMatrix

  !=========================================================================================================================
  subroutine ComputeOrderParameters(r1,r2,mol_list,ConnectionMatrix,LatVecs,RLatVecs,SteinhardtQList)
    real(PR), dimension(:,:), intent(in) :: r1,r2
    integer, dimension(:), intent(in) :: mol_list
    logical, dimension(:,:), intent(in) :: ConnectionMatrix
    real(PR), dimension(3,3), intent(in) :: LatVecs, RLatVecs
    integer, dimension(:), intent(in) :: SteinhardtQList
    
    integer :: atm2, atm3
    integer :: mol1, mol2, mol3
    real(PR) :: rij(3), rik(3), rjk(3), rij_m, rik_m, rjk_m, theta_kij
    integer :: n_1, n_2, n_alpha, n_beta, n_gamma
    real(PR) :: phi, a
    integer :: OpNum

    do mol1=1,size(r1,2)
      atom2:do atm2=1,size(r2,2)
        mol2=mol_list(atm2)
        if(ConnectionMatrix(mol1,mol2))then
          rij=r1(:,mol1)-r2(:,atm2)
          call ApplyPBC(rij,LatVecs,RLatVecs)
          rij_m=sqrt(rij(1)**2+rij(2)**2+rij(3)**2)
          if(rij_m < 0.1_PR)cycle atom2 !** Reject same atom
          N(mol1)=N(mol1)+1
        else
          cycle atom2
        end if
        
        call ComputeSteinhardt_qlm(rij,mol1,SteinhardtQList)
          
        atom3:do atm3=atm2+1,size(r2,2)
          mol3=mol_list(atm3)
          if(ConnectionMatrix(mol1,mol3))then
            rik=r1(:,mol1)-r2(:,atm3)
            call ApplyPBC(rik,LatVecs,RLatVecs)
            rik_m=sqrt(rik(1)**2+rik(2)**2+rik(3)**2)
            if(rik_m < 0.1_PR)cycle atom3
            N2(mol1)=N2(mol1)+1
          else
            cycle atom3
          end if
          rjk=r2(:,atm2)-r2(:,atm3)
          call ApplyPBC(rjk,LatVecs,RLatVecs)
          rjk_m=sqrt(rjk(1)**2+rjk(2)**2+rjk(3)**2)
          theta_kij=acos(dot_product(rik,rij)/(rij_m*rik_m))
          do OpNum=1,NumberOfOrderParameters
            select case(OrderParameterType(OpNum)%name)
            case ('B')
              n_1=OrderParameterType(OpNum)%iarg(1)
              n_2=OrderParameterType(OpNum)%iarg(2)
              phi=OrderParameterType(OpNum)%arg(1)
              OrderParameterValue(OpNum,mol1)%local=OrderParameterValue(OpNum,mol1)%local+(cos(n_2*theta_kij+phi))**n_1
            case ('D')
              n_alpha=OrderParameterType(OpNum)%iarg(1)
              n_beta=OrderParameterType(OpNum)%iarg(2)
              n_gamma=OrderParameterType(OpNum)%iarg(3)
              OrderParameterValue(OpNum,mol1)%local=OrderParameterValue(OpNum,mol1)%local+f(rij_m,n_alpha)*f(rik_m,n_beta)*f(rjk_m,n_gamma)
            case ('F')
              n_alpha=OrderParameterType(OpNum)%iarg(1)
              n_beta=OrderParameterType(OpNum)%iarg(2)
              a=OrderParameterType(OpNum)%arg(1)
              OrderParameterValue(OpNum,mol1)%local=OrderParameterValue(OpNum,mol1)%local+f(min(rij_m,rik_m),n_alpha)*f(max(rij_m,rik_m),n_beta)*cos(a*theta_kij)
            case ('I')
              OrderParameterValue(OpNum,mol1)%local=OrderParameterValue(OpNum,mol1)%local+(cos(theta_kij)+1._PR/3._PR)**2
            end select 
          end do
        end do atom3
      end do atom2
    end do
  end subroutine ComputeOrderParameters

  subroutine ComputeUnaveragedOrderParameters(SteinhardtQList)
    integer, dimension(:), intent(in) :: SteinhardtQList

    integer :: mol1, l, OpNum

    do mol1=1,NumberOfMolecules
      call ComputeSteinhardt_q(mol1,N(mol1),SteinhardtQList)

      do OpNum=1,NumberOfOrderParameters
        select case(OrderParameterType(OpNum)%name)
        case ('B','D','F')
          OrderParameterValue(OpNum,mol1)%local=OrderParameterValue(OpNum,mol1)%local/real(N2(mol1),PR)
        case ('I')
          OrderParameterValue(OpNum,mol1)%local=1._PR-0.375_PR*OrderParameterValue(OpNum,mol1)%local
        case ('Q')
          l=OrderParameterType(OpNum)%iarg(1)
          select case(l)
          case (2)
            OrderParameterValue(OpNum,mol1)%local=sqrt(4._PR*PI/real(2*l+1,PR))*q2(mol1)
          case (3)
            OrderParameterValue(OpNum,mol1)%local=sqrt(4._PR*PI/real(2*l+1,PR))*q3(mol1)
          case (4)
            OrderParameterValue(OpNum,mol1)%local=sqrt(4._PR*PI/real(2*l+1,PR))*q4(mol1)
          case (5)
            OrderParameterValue(OpNum,mol1)%local=sqrt(4._PR*PI/real(2*l+1,PR))*q5(mol1)
          case (6)
            OrderParameterValue(OpNum,mol1)%local=sqrt(4._PR*PI/real(2*l+1,PR))*q6(mol1)
          case (7)
            OrderParameterValue(OpNum,mol1)%local=sqrt(4._PR*PI/real(2*l+1,PR))*q7(mol1)
          case (8)
            OrderParameterValue(OpNum,mol1)%local=sqrt(4._PR*PI/real(2*l+1,PR))*q8(mol1)
          case (9)
            OrderParameterValue(OpNum,mol1)%local=sqrt(4._PR*PI/real(2*l+1,PR))*q9(mol1)
          case (10)
            OrderParameterValue(OpNum,mol1)%local=sqrt(4._PR*PI/real(2*l+1,PR))*q10(mol1)
          case (12)
            OrderParameterValue(OpNum,mol1)%local=sqrt(4._PR*PI/real(2*l+1,PR))*q12(mol1)
          case (14)
            OrderParameterValue(OpNum,mol1)%local=sqrt(4._PR*PI/real(2*l+1,PR))*q14(mol1)
          case (16)
            OrderParameterValue(OpNum,mol1)%local=sqrt(4._PR*PI/real(2*l+1,PR))*q16(mol1)
          end select
        case ('W')
          l=OrderParameterType(OpNum)%iarg(1)
          select case(l)
          case (2)
            OrderParameterValue(OpNum,mol1)%local=SteinhardtW(l,q2m(:,mol1),q2(mol1))
          case (3)
            OrderParameterValue(OpNum,mol1)%local=SteinhardtW(l,q3m(:,mol1),q3(mol1))
          case (4)
            OrderParameterValue(OpNum,mol1)%local=SteinhardtW(l,q4m(:,mol1),q4(mol1))
          case (5)
            OrderParameterValue(OpNum,mol1)%local=SteinhardtW(l,q5m(:,mol1),q5(mol1))
          case (6)
            OrderParameterValue(OpNum,mol1)%local=SteinhardtW(l,q6m(:,mol1),q6(mol1))
          case (7)
            OrderParameterValue(OpNum,mol1)%local=SteinhardtW(l,q7m(:,mol1),q7(mol1))
          case (8)
            OrderParameterValue(OpNum,mol1)%local=SteinhardtW(l,q8m(:,mol1),q8(mol1))
          case (9)
            OrderParameterValue(OpNum,mol1)%local=SteinhardtW(l,q9m(:,mol1),q9(mol1))
          case (10)
            OrderParameterValue(OpNum,mol1)%local=SteinhardtW(l,q10m(:,mol1),q10(mol1))
          case (12)
            OrderParameterValue(OpNum,mol1)%local=SteinhardtW(l,q12m(:,mol1),q12(mol1))
          case (14)
            OrderParameterValue(OpNum,mol1)%local=SteinhardtW(l,q14m(:,mol1),q14(mol1))
          case (16)
            OrderParameterValue(OpNum,mol1)%local=SteinhardtW(l,q16m(:,mol1),q16(mol1))
          end select
        end select
      end do
    end do
  end subroutine ComputeUnaveragedOrderParameters

  !=========================================================================================================================
  subroutine ComputeAveragedOrderParameters(SteinhardtQavgList, SteinhardtLQList,ConnectionMatrix)
    integer, dimension(:), intent(in) :: SteinhardtQavgList, SteinhardtLQList
    logical, dimension(:,:), intent(in) :: ConnectionMatrix
    
    integer :: mol1, mol2
    integer :: NumberOfNeighbors, i, l, OpNum

    do mol1=1,NumberOfMolecules
      !******   Sum over local OPs   ****************
      do mol2=1,NumberOfMolecules
        if(ConnectionMatrix(mol1,mol2))then
          do OpNum=1,NumberOfOrderParameters
            select case(OrderParameterType(OpNum)%name)
            case ('B','D','F','I')
              OrderParameterValue(OpNum,mol1)%avg=OrderParameterValue(OpNum,mol1)%avg+OrderParameterValue(OpNum,mol2)%local
            end select
          end do
          do i=1,size(SteinhardtQavgList)
            l=SteinhardtQavgList(i)
            select case(l)
              case(2)
                q2m_avg(:,mol1)=q2m_avg(:,mol1)+q2m(:,mol2)
              case(3)
                q3m_avg(:,mol1)=q3m_avg(:,mol1)+q3m(:,mol2)
              case(4)
                q4m_avg(:,mol1)=q4m_avg(:,mol1)+q4m(:,mol2)
              case(5)
                q5m_avg(:,mol1)=q5m_avg(:,mol1)+q5m(:,mol2)
              case(6)
                q6m_avg(:,mol1)=q6m_avg(:,mol1)+q6m(:,mol2)
              case(7)
                q7m_avg(:,mol1)=q7m_avg(:,mol1)+q7m(:,mol2)
              case(8)
                q8m_avg(:,mol1)=q8m_avg(:,mol1)+q8m(:,mol2)
              case(9)
                q9m_avg(:,mol1)=q9m_avg(:,mol1)+q9m(:,mol2)
              case(10)
                q10m_avg(:,mol1)=q10m_avg(:,mol1)+q10m(:,mol2)
              case(12)
                q12m_avg(:,mol1)=q12m_avg(:,mol1)+q12m(:,mol2)
              case(14)
                q14m_avg(:,mol1)=q14m_avg(:,mol1)+q14m(:,mol2)
              case(16)
                q16m_avg(:,mol1)=q16m_avg(:,mol1)+q16m(:,mol2)
            end select
          end do
          do i=1,size(SteinhardtLQList)
            if(mol1 == mol2)cycle
            l=SteinhardtLQList(i)
            select case(l)
              case(2)
                lq2m(:,mol1)=lq2m(:,mol1)+q2m(:,mol1)*conjg(q2m(:,mol2))/(q2(mol1)*q2(mol2))
              case(3)
                lq3m(:,mol1)=lq3m(:,mol1)+q3m(:,mol1)*conjg(q3m(:,mol2))/(q3(mol1)*q3(mol2))
              case(4)
                lq4m(:,mol1)=lq4m(:,mol1)+q4m(:,mol1)*conjg(q4m(:,mol2))/(q4(mol1)*q4(mol2))
              case(5)
                lq5m(:,mol1)=lq5m(:,mol1)+q5m(:,mol1)*conjg(q5m(:,mol2))/(q5(mol1)*q5(mol2))
              case(6)
                lq6m(:,mol1)=lq6m(:,mol1)+q6m(:,mol1)*conjg(q6m(:,mol2))/(q6(mol1)*q6(mol2))
              case(7)
                lq7m(:,mol1)=lq7m(:,mol1)+q7m(:,mol1)*conjg(q7m(:,mol2))/(q7(mol1)*q7(mol2))
              case(8)
                lq8m(:,mol1)=lq8m(:,mol1)+q8m(:,mol1)*conjg(q8m(:,mol2))/(q8(mol1)*q8(mol2))
              case(9)
                lq9m(:,mol1)=lq9m(:,mol1)+q9m(:,mol1)*conjg(q9m(:,mol2))/(q9(mol1)*q9(mol2))
              case(10)
                lq10m(:,mol1)=lq10m(:,mol1)+q10m(:,mol1)*conjg(q10m(:,mol2))/(q10(mol1)*q10(mol2))
              case(12)
                lq12m(:,mol1)=lq12m(:,mol1)+q12m(:,mol1)*conjg(q12m(:,mol2))/(q12(mol1)*q12(mol2))
              case(14)
                lq14m(:,mol1)=lq14m(:,mol1)+q14m(:,mol1)*conjg(q14m(:,mol2))/(q14(mol1)*q14(mol2))
              case(16)
                lq16m(:,mol1)=lq16m(:,mol1)+q16m(:,mol1)*conjg(q16m(:,mol2))/(q16(mol1)*q16(mol2))
            end select
          end do
        end if
      end do

      !***** Compute Average    ********
      NumberOfNeighbors=count(ConnectionMatrix(mol1,1:NumberOfMolecules))-1
      !**** Assign values to B, D, F, I if any   ******
      do OpNum=1,NumberOfOrderParameters
        select case(OrderParameterType(OpNum)%name)
        case ('B','D','F','I')
          OrderParameterValue(OpNum,mol1)%avg=OrderParameterValue(OpNum,mol1)%avg/real(NumberOfNeighbors+1,PR)
        end select
      end do
      do i=1,size(SteinhardtQavgList)
        l=SteinhardtQavgList(i)
        select case(l)
        case (2)
          q2m_avg(:,mol1)=q2m_avg(:,mol1)/real(NumberOfNeighbors+1,PR)
          q2_avg(mol1)=sqrt(real(dot_product(q2m_avg(:,mol1),q2m_avg(:,mol1))))
        case (3)
          q3m_avg(:,mol1)=q3m_avg(:,mol1)/real(NumberOfNeighbors+1,PR)
          q3_avg(mol1)=sqrt(real(dot_product(q3m_avg(:,mol1),q3m_avg(:,mol1))))
        case (4)
          q4m_avg(:,mol1)=q4m_avg(:,mol1)/real(NumberOfNeighbors+1,PR)
          q4_avg(mol1)=sqrt(real(dot_product(q4m_avg(:,mol1),q4m_avg(:,mol1))))
        case (5)
          q5m_avg(:,mol1)=q5m_avg(:,mol1)/real(NumberOfNeighbors+1,PR)
          q5_avg(mol1)=sqrt(real(dot_product(q5m_avg(:,mol1),q5m_avg(:,mol1))))
        case (6)
          q6m_avg(:,mol1)=q6m_avg(:,mol1)/real(NumberOfNeighbors+1,PR)
          q6_avg(mol1)=sqrt(real(dot_product(q6m_avg(:,mol1),q6m_avg(:,mol1))))
        case (7)
          q7m_avg(:,mol1)=q7m_avg(:,mol1)/real(NumberOfNeighbors+1,PR)
          q7_avg(mol1)=sqrt(real(dot_product(q7m_avg(:,mol1),q7m_avg(:,mol1))))
        case (8)
          q8m_avg(:,mol1)=q8m_avg(:,mol1)/real(NumberOfNeighbors+1,PR)
          q8_avg(mol1)=sqrt(real(dot_product(q8m_avg(:,mol1),q8m_avg(:,mol1))))
        case (9)
          q9m_avg(:,mol1)=q9m_avg(:,mol1)/real(NumberOfNeighbors+1,PR)
          q9_avg(mol1)=sqrt(real(dot_product(q9m_avg(:,mol1),q9m_avg(:,mol1))))
        case (10)
          q10m_avg(:,mol1)=q10m_avg(:,mol1)/real(NumberOfNeighbors+1,PR)
          q10_avg(mol1)=sqrt(real(dot_product(q10m_avg(:,mol1),q10m_avg(:,mol1))))
        case (12)
          q12m_avg(:,mol1)=q12m_avg(:,mol1)/real(NumberOfNeighbors+1,PR)
          q12_avg(mol1)=sqrt(real(dot_product(q12m_avg(:,mol1),q12m_avg(:,mol1))))
        case (14)
          q14m_avg(:,mol1)=q14m_avg(:,mol1)/real(NumberOfNeighbors+1,PR)
          q14_avg(mol1)=sqrt(real(dot_product(q14m_avg(:,mol1),q14m_avg(:,mol1))))
        case (16)
          q16m_avg(:,mol1)=q16m_avg(:,mol1)/real(NumberOfNeighbors+1,PR)
          q16_avg(mol1)=sqrt(real(dot_product(q16m_avg(:,mol1),q16m_avg(:,mol1))))
        end select
      end do
      do i=1,size(SteinhardtLQList)
        l=SteinhardtLQList(i)
        select case(l)
        case (2)
          lq2m(:,mol1)=lq2m(:,mol1)/real(NumberOfNeighbors,PR)
          lq2(mol1)=sqrt(real(dot_product(lq2m(:,mol1),lq2m(:,mol1))))
        case (3)
          lq3m(:,mol1)=lq3m(:,mol1)/real(NumberOfNeighbors,PR)
          lq3(mol1)=sqrt(real(dot_product(lq3m(:,mol1),lq3m(:,mol1))))
        case (4)
          lq4m(:,mol1)=lq4m(:,mol1)/real(NumberOfNeighbors,PR)
          lq4(mol1)=sqrt(real(dot_product(lq4m(:,mol1),lq4m(:,mol1))))
        case (5)
          lq5m(:,mol1)=lq5m(:,mol1)/real(NumberOfNeighbors,PR)
          lq5(mol1)=sqrt(real(dot_product(lq5m(:,mol1),lq5m(:,mol1))))
        case (6)
          lq6m(:,mol1)=lq6m(:,mol1)/real(NumberOfNeighbors,PR)
          lq6(mol1)=sqrt(real(dot_product(lq6m(:,mol1),lq6m(:,mol1))))
        case (7)
          lq7m(:,mol1)=lq7m(:,mol1)/real(NumberOfNeighbors,PR)
          lq7(mol1)=sqrt(real(dot_product(lq7m(:,mol1),lq7m(:,mol1))))
        case (8)
          lq8m(:,mol1)=lq8m(:,mol1)/real(NumberOfNeighbors,PR)
          lq8(mol1)=sqrt(real(dot_product(lq8m(:,mol1),lq8m(:,mol1))))
        case (9)
          lq9m(:,mol1)=lq9m(:,mol1)/real(NumberOfNeighbors,PR)
          lq9(mol1)=sqrt(real(dot_product(lq9m(:,mol1),lq9m(:,mol1))))
        case (10)
          lq10m(:,mol1)=lq10m(:,mol1)/real(NumberOfNeighbors,PR)
          lq10(mol1)=sqrt(real(dot_product(lq10m(:,mol1),lq10m(:,mol1))))
        case (12)
          lq12m(:,mol1)=lq12m(:,mol1)/real(NumberOfNeighbors,PR)
          lq12(mol1)=sqrt(real(dot_product(lq12m(:,mol1),lq12m(:,mol1))))
        case (14)
          lq14m(:,mol1)=lq14m(:,mol1)/real(NumberOfNeighbors,PR)
          lq14(mol1)=sqrt(real(dot_product(lq14m(:,mol1),lq14m(:,mol1))))
        case (16)
          lq16m(:,mol1)=lq16m(:,mol1)/real(NumberOfNeighbors,PR)
          lq16(mol1)=sqrt(real(dot_product(lq16m(:,mol1),lq16m(:,mol1))))
        end select
      end do


      !***** Assign values to Q, W, LQ and LW if any      **********
      do OpNum=1,NumberOfOrderParameters
        l=OrderParameterType(OpNum)%iarg(1)
        select case(OrderParameterType(OpNum)%name)
        case ('Q')
          select case(l)
          case (2)
            OrderParameterValue(OpNum,mol1)%avg=sqrt(4._PR*PI/real(2*l+1,PR))*q2_avg(mol1)
          case (3)
            OrderParameterValue(OpNum,mol1)%avg=sqrt(4._PR*PI/real(2*l+1,PR))*q3_avg(mol1)
          case (4)
            OrderParameterValue(OpNum,mol1)%avg=sqrt(4._PR*PI/real(2*l+1,PR))*q4_avg(mol1)
          case (5)
            OrderParameterValue(OpNum,mol1)%avg=sqrt(4._PR*PI/real(2*l+1,PR))*q5_avg(mol1)
          case (6)
            OrderParameterValue(OpNum,mol1)%avg=sqrt(4._PR*PI/real(2*l+1,PR))*q6_avg(mol1)
          case (7)
            OrderParameterValue(OpNum,mol1)%avg=sqrt(4._PR*PI/real(2*l+1,PR))*q7_avg(mol1)
          case (8)
            OrderParameterValue(OpNum,mol1)%avg=sqrt(4._PR*PI/real(2*l+1,PR))*q8_avg(mol1)
          case (9)
            OrderParameterValue(OpNum,mol1)%avg=sqrt(4._PR*PI/real(2*l+1,PR))*q9_avg(mol1)
          case (10)
            OrderParameterValue(OpNum,mol1)%avg=sqrt(4._PR*PI/real(2*l+1,PR))*q10_avg(mol1)
          case (12)
            OrderParameterValue(OpNum,mol1)%avg=sqrt(4._PR*PI/real(2*l+1,PR))*q12_avg(mol1)
          case (14)
            OrderParameterValue(OpNum,mol1)%avg=sqrt(4._PR*PI/real(2*l+1,PR))*q14_avg(mol1)
          case (16)
            OrderParameterValue(OpNum,mol1)%avg=sqrt(4._PR*PI/real(2*l+1,PR))*q16_avg(mol1)
          end select
        case ('W')
          select case(l)
          case (2)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,q2m_avg(:,mol1),q2_avg(mol1))
          case (3)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,q3m_avg(:,mol1),q3_avg(mol1))
          case (4)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,q4m_avg(:,mol1),q4_avg(mol1))
          case (5)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,q5m_avg(:,mol1),q5_avg(mol1))
          case (6)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,q6m_avg(:,mol1),q6_avg(mol1))
          case (7)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,q7m_avg(:,mol1),q7_avg(mol1))
          case (8)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,q8m_avg(:,mol1),q8_avg(mol1))
          case (9)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,q9m_avg(:,mol1),q9_avg(mol1))
          case (10)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,q10m_avg(:,mol1),q10_avg(mol1))
          case (12)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,q12m_avg(:,mol1),q12_avg(mol1))
          case (14)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,q14m_avg(:,mol1),q14_avg(mol1))
          case (16)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,q16m_avg(:,mol1),q16_avg(mol1))
          end select
        case ('LQ')
          select case(l)
          case (2)
            OrderParameterValue(OpNum,mol1)%avg=sum(real(lq2m(:,mol1)))
          case (3)
            OrderParameterValue(OpNum,mol1)%avg=sum(real(lq3m(:,mol1)))
          case (4)
            OrderParameterValue(OpNum,mol1)%avg=sum(real(lq4m(:,mol1)))
          case (5)
            OrderParameterValue(OpNum,mol1)%avg=sum(real(lq5m(:,mol1)))
          case (6)
            OrderParameterValue(OpNum,mol1)%avg=sum(real(lq6m(:,mol1)))
          case (7)
            OrderParameterValue(OpNum,mol1)%avg=sum(real(lq7m(:,mol1)))
          case (8)
            OrderParameterValue(OpNum,mol1)%avg=sum(real(lq8m(:,mol1)))
          case (9)
            OrderParameterValue(OpNum,mol1)%avg=sum(real(lq9m(:,mol1)))
          case (10)
            OrderParameterValue(OpNum,mol1)%avg=sum(real(lq10m(:,mol1)))
          case (12)
            OrderParameterValue(OpNum,mol1)%avg=sum(real(lq12m(:,mol1)))
          case (14)
            OrderParameterValue(OpNum,mol1)%avg=sum(real(lq14m(:,mol1)))
          case (16)
            OrderParameterValue(OpNum,mol1)%avg=sum(real(lq16m(:,mol1)))
          end select
        case ('LW')
          select case(l)
          case (2)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,lq2m(:,mol1),lq2(mol1))
          case (3)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,lq3m(:,mol1),lq3(mol1))
          case (4)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,lq4m(:,mol1),lq4(mol1))
          case (5)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,lq5m(:,mol1),lq5(mol1))
          case (6)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,lq6m(:,mol1),lq6(mol1))
          case (7)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,lq7m(:,mol1),lq7(mol1))
          case (8)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,lq8m(:,mol1),lq8(mol1))
          case (9)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,lq9m(:,mol1),lq9(mol1))
          case (10)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,lq10m(:,mol1),lq10(mol1))
          case (12)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,lq12m(:,mol1),lq12(mol1))
          case (14)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,lq14m(:,mol1),lq14(mol1))
          case (16)
            OrderParameterValue(OpNum,mol1)%avg=SteinhardtW(l,lq16m(:,mol1),lq16(mol1))
          end select
        end select
      end do
    end do
  end subroutine ComputeAveragedOrderParameters

  !=====================================================================================================
  subroutine ComputeSteinhardt_qlm(rij,mol,SteinhardtQList)
    real(PR), dimension(3), intent(in) :: rij
    integer, intent(in) :: mol
    integer, dimension(:), intent(in) :: SteinhardtQList

    real(PR) :: theta,phi
    complex(PR) :: sp
    integer :: i,l,m,N

    theta=atan2(sqrt(rij(1)**2+rij(2)**2),rij(3))
    phi=atan2(rij(2),rij(1))
    N=size(SteinhardtQList,1)
    do i=1,N
      l=SteinhardtQList(i)
      select case(l)
      case (2)
        do m=-l,l
          sp=sp_hrmcs(l,m,theta,phi)
          q2m(m,mol)=q2m(m,mol)+sp
        end do
      case (3)
        do m=-l,l
          sp=sp_hrmcs(l,m,theta,phi)
          q3m(m,mol)=q3m(m,mol)+sp
        end do
      case (4)
        do m=-l,l
          sp=sp_hrmcs(l,m,theta,phi)
          q4m(m,mol)=q4m(m,mol)+sp
        end do
      case (5)
        do m=-l,l
          sp=sp_hrmcs(l,m,theta,phi)
          q5m(m,mol)=q5m(m,mol)+sp
        end do
      case (6)
        do m=-l,l
          sp=sp_hrmcs(l,m,theta,phi)
          q6m(m,mol)=q6m(m,mol)+sp
        end do
      case (7)
        do m=-l,l
          sp=sp_hrmcs(l,m,theta,phi)
          q7m(m,mol)=q7m(m,mol)+sp
        end do
      case (8)
        do m=-l,l
          sp=sp_hrmcs(l,m,theta,phi)
          q8m(m,mol)=q8m(m,mol)+sp
        end do
      case (9)
        do m=-l,l
          sp=sp_hrmcs(l,m,theta,phi)
          q9m(m,mol)=q9m(m,mol)+sp
        end do
      case (10)
        do m=-l,l
          sp=sp_hrmcs(l,m,theta,phi)
          q10m(m,mol)=q10m(m,mol)+sp
        end do
      case (12)
        do m=-l,l
          sp=sp_hrmcs(l,m,theta,phi)
          q12m(m,mol)=q12m(m,mol)+sp
        end do
      case (14)
        do m=-l,l
          sp=sp_hrmcs(l,m,theta,phi)
          q14m(m,mol)=q14m(m,mol)+sp
        end do
      case (16)
        do m=-l,l
          sp=sp_hrmcs(l,m,theta,phi)
          q16m(m,mol)=q16m(m,mol)+sp
        end do
      end select
    end do
  end subroutine ComputeSteinhardt_qlm

  !=====================================================================================================
  subroutine ComputeSteinhardt_q(mol1,N,SteinhardtQList)
    integer, intent(in) :: mol1, N
    integer, dimension(:), intent(in) :: SteinhardtQList

    integer :: i, l

    do i=1,size(SteinhardtQList)
      l=SteinhardtQList(i)
      select case(l)
      case (2)
        q2m(:,mol1)=q2m(:,mol1)/real(N,PR)
        q2(mol1)=sqrt(real(dot_product(q2m(:,mol1),q2m(:,mol1))))
      case (3)
        q3m(:,mol1)=q3m(:,mol1)/real(N,PR)
        q3(mol1)=sqrt(real(dot_product(q3m(:,mol1),q3m(:,mol1))))
      case (4)
        q4m(:,mol1)=q4m(:,mol1)/real(N,PR)
        q4(mol1)=sqrt(real(dot_product(q4m(:,mol1),q4m(:,mol1))))
      case (5)
        q5m(:,mol1)=q5m(:,mol1)/real(N,PR)
        q5(mol1)=sqrt(real(dot_product(q5m(:,mol1),q5m(:,mol1))))
      case (6)
        q6m(:,mol1)=q6m(:,mol1)/real(N,PR)
        q6(mol1)=sqrt(real(dot_product(q6m(:,mol1),q6m(:,mol1))))
      case (7)
        q7m(:,mol1)=q7m(:,mol1)/real(N,PR)
        q7(mol1)=sqrt(real(dot_product(q7m(:,mol1),q7m(:,mol1))))
      case (8)
        q8m(:,mol1)=q8m(:,mol1)/real(N,PR)
        q8(mol1)=sqrt(real(dot_product(q8m(:,mol1),q8m(:,mol1))))
      case (9)
        q9m(:,mol1)=q9m(:,mol1)/real(N,PR)
        q9(mol1)=sqrt(real(dot_product(q9m(:,mol1),q9m(:,mol1))))
      case (10)
        q10m(:,mol1)=q10m(:,mol1)/real(N,PR)
        q10(mol1)=sqrt(real(dot_product(q10m(:,mol1),q10m(:,mol1))))
      case (12)
        q12m(:,mol1)=q12m(:,mol1)/real(N,PR)
        q12(mol1)=sqrt(real(dot_product(q12m(:,mol1),q12m(:,mol1))))
      case (14)
        q14m(:,mol1)=q14m(:,mol1)/real(N,PR)
        q14(mol1)=sqrt(real(dot_product(q14m(:,mol1),q14m(:,mol1))))
      case (16)
        q16m(:,mol1)=q16m(:,mol1)/real(N,PR)
        q16(mol1)=sqrt(real(dot_product(q16m(:,mol1),q16m(:,mol1))))
      end select
    end do

  end subroutine ComputeSteinhardt_q
  !=====================================================================================================
  function f(r,n)
    real(PR), intent(in) :: r
    integer, intent(in) :: n
    real(PR) :: f

    select case (n)
    case (1)
      f=sqrt(r)
    case (2)
      f=r
    case (3)
      f=r**2
    case (4)
      f=1._PR-exp(-2._PR*(r-3._PR)**2/9._PR)
    case (5)
      f=0.5_PR+0.5_PR*exp(-2._PR*(r-3._PR)**2/9._PR)
    end select
  end function f

  !=====================================================================================================
  subroutine ApplyPBC(vec,LatVecs,RLatVecs)
    real(PR), dimension(3), intent(inout) :: vec
    real(PR), dimension(3,3), intent(in) :: LatVecs, RLatVecs

    real(PR), dimension(3) :: svec

    svec=matmul(RLatVecs,vec)
    svec=svec-anint(svec)
    vec=matmul(LatVecs,svec)

  end subroutine ApplyPBC

  !=========================================================================================================================
  ! WriteCSVHeader: writes one header row to an open CSV file unit.
  !   avg_flag = .true.  -> column names end with "avg"  (OP_avg.csv)
  !   avg_flag = .false. -> no suffix                    (OP_unavg.csv)
  !=========================================================================================================================
  subroutine WriteCSVHeader(unit, avg_flag)
    integer, intent(in) :: unit
    logical, intent(in) :: avg_flag

    integer           :: i
    character(len=3)  :: suffix
    character(len=5)  :: phi_str
    character(len=20) :: colname

    suffix = ''
    if (avg_flag) suffix = 'avg'

    write(unit, '(a)', advance='no') 'mol_id'
    do i = 1, NumberOfOrderParameters
      write(unit, '(a)', advance='no') ','
      select case(trim(OrderParameterType(i)%name))
      case('B')
        call RealToUnderscoreStr(OrderParameterType(i)%arg(1), phi_str)
        write(colname, '(a,i0,a,i0,a,a,a)') &
          'B_', OrderParameterType(i)%iarg(1), '_', &
          OrderParameterType(i)%iarg(2), '_', trim(phi_str), trim(suffix)
      case('D')
        write(colname, '(a,i0,a,i0,a,i0,a)') &
          'D_', OrderParameterType(i)%iarg(1), '_', &
          OrderParameterType(i)%iarg(2), '_', &
          OrderParameterType(i)%iarg(3), trim(suffix)
      case('F')
        call RealToUnderscoreStr(OrderParameterType(i)%arg(1), phi_str)
        write(colname, '(a,i0,a,i0,a,a,a)') &
          'F_', OrderParameterType(i)%iarg(1), '_', &
          OrderParameterType(i)%iarg(2), '_', trim(phi_str), trim(suffix)
      case('I')
        write(colname, '(a,a)') 'I', trim(suffix)
      case default   ! Q, W, LQ, LW
        write(colname, '(a,a,i0,a)') &
          trim(OrderParameterType(i)%name), '_', &
          OrderParameterType(i)%iarg(1), trim(suffix)
      end select
      write(unit, '(a)', advance='no') trim(colname)
    end do
    write(unit, *)
  end subroutine WriteCSVHeader

  !-------------------------------------------------------------------------------------------------------------------------
  subroutine RealToUnderscoreStr(val, str)
    real(PR), intent(in)          :: val
    character(len=5), intent(out) :: str
    integer :: k
    write(str, '(f4.2)') val
    do k = 1, 5
      if (str(k:k) == '.') then
        str(k:k) = '_'
        exit
      end if
    end do
  end subroutine RealToUnderscoreStr

end module order_parameters
