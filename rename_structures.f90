! Read in ML_ABN and change the system name of the configurations in the
! section(s) of configuration numbers entered by the user. The result is
! written to ML_ABN_rename, everything else is copied unchanged.
! The total number of configurations is printed first; based on it, enter the
! first and last configuration of the section and the new name (e.g.
! "30 100 hBN with C" renames structures 30 to 100 to 'hBN with C').
! Sections can be entered repeatedly; enter "0 0" to finish.
! NB: -The name may contain blanks, it is everything that follows the two
!      numbers (up to 40 characters, as in the ML_AB format)
!     -Sections may overlap, the last entry wins
!     -Copy/rename the ML_ABN* files as needed
!
! Compile: gfortran -o rename_structures rename_structures.f90
!
  PROGRAM RENAME_STRUCTURES
    IMPLICIT NONE
    INTEGER, PARAMETER :: LLIN = 1024  ! max. length of a line of ML_ABN
    INTEGER, PARAMETER :: LNAM = 40    ! max. length of a system name
    INTEGER                         :: NCONFM   ! total no. of configurations
    INTEGER                         :: ICONF    ! configuration being copied
    INTEGER                         :: NREN     ! no. of configurations renamed
    INTEGER                         :: IOS, I, IPOS
    INTEGER                         :: I1, I2   ! section to rename
    CHARACTER(LEN=LNAM), ALLOCATABLE :: CNAME(:) ! new name (blank = keep old)
    CHARACTER(LEN=LLIN)             :: LINE
    CHARACTER(LEN=LNAM)             :: CNEW
    !
    ! read the total number of configurations (line 5 of ML_ABN)
    OPEN(10,FILE='ML_ABN',STATUS='OLD',IOSTAT=IOS)
    IF (IOS /= 0) ERROR STOP 'Error: could not open ML_ABN'
    DO I=1,4
      READ(10,'(a)',IOSTAT=IOS) LINE
      IF (IOS /= 0) ERROR STOP 'Error: unexpected end of ML_ABN in header'
    ENDDO
    READ(10,*,IOSTAT=IOS) NCONFM
    IF (IOS /= 0 .OR. NCONFM < 1) ERROR STOP 'Error: could not read the number of configurations'
    ALLOCATE(CNAME(NCONFM))
    CNAME = ' '
    !
    ! choose the sections to rename
    WRITE(*,*) 'Total number of configurations = ', NCONFM
    WRITE(*,*) 'Enter the first and last configuration of the section and the new name'
    WRITE(*,*) '(e.g. "30 100 hBN with C" renames structures 30 to 100).'
    WRITE(*,*) 'Sections can be entered repeatedly; enter "0 0" when done:'
    DO
      READ(*,'(a)',IOSTAT=IOS) LINE
      IF (IOS /= 0) ERROR STOP 'Error: unexpected end of input'
      IF (LEN_TRIM(LINE) == 0) CYCLE
      READ(LINE,*,IOSTAT=IOS) I1, I2
      IF (IOS /= 0) THEN
        WRITE(*,*) 'Invalid input: enter "first last name" ("0 0" to finish)'
        CYCLE
      ENDIF
      IF (I1 == 0 .AND. I2 == 0) EXIT
      IF (I1 > I2) THEN
        ! swap so that I1 <= I2
        I  = I1
        I1 = I2
        I2 = I
      ENDIF
      IF (I1 < 1 .OR. I2 > NCONFM) THEN
        WRITE(*,*) 'Invalid section: configurations must lie between 1 and ', NCONFM
        CYCLE
      ENDIF
      ! the new name is everything after the two numbers
      CNEW = AFTER_TWO(LINE)
      IF (LEN_TRIM(CNEW) == 0) THEN
        WRITE(*,*) 'Invalid input: no name given after the two numbers'
        CYCLE
      ENDIF
      CNAME(I1:I2) = CNEW
      WRITE(*,*) 'Renaming configurations ', I1, ' to ', I2, ' to "', TRIM(CNEW), '"'
    ENDDO
    !
    ! copy ML_ABN, replacing the name of the selected configurations
    REWIND(10)
    OPEN(11,FILE='ML_ABN_rename',STATUS='REPLACE')
    ICONF = 0
    NREN  = 0
    DO
      READ(10,'(a)',IOSTAT=IOS) LINE
      IF (IOS /= 0) EXIT
      WRITE(11,'(a)') TRIM(LINE)
      IPOS = INDEX(LINE,'Configuration num.')
      IF (IPOS > 0) THEN
        READ(LINE(IPOS+18:),*,IOSTAT=IOS) ICONF
        IF (IOS /= 0 .OR. ICONF < 1 .OR. ICONF > NCONFM) &
          ERROR STOP 'Error: could not read a configuration number'
      ELSEIF (INDEX(LINE,'System name') > 0 .AND. ICONF > 0) THEN
        IF (CNAME(ICONF) /= ' ') THEN
          ! copy the '----' separator, then write the new name
          READ(10,'(a)',IOSTAT=IOS) LINE
          IF (IOS /= 0) ERROR STOP 'Error: unexpected end of ML_ABN in a system name'
          WRITE(11,'(a)') TRIM(LINE)
          READ(10,'(a)',IOSTAT=IOS) LINE   ! old name, dropped
          IF (IOS /= 0) ERROR STOP 'Error: unexpected end of ML_ABN in a system name'
          WRITE(11,'(5x,a)') CNAME(ICONF)
          NREN = NREN+1
        ENDIF
      ENDIF
    ENDDO
    !
    WRITE(*,*) 'Total number of configurations = ', NCONFM
    WRITE(*,*) 'Configurations renamed         = ', NREN
    IF (NREN /= COUNT(CNAME /= ' ')) THEN
      WRITE(*,*) 'WARNING: ', COUNT(CNAME /= ' '), ' configurations were selected but ', &
                 NREN, ' name(s) were found'
    ENDIF
    WRITE(*,*) 'Written to ML_ABN_rename'
    !
    CLOSE(10)
    CLOSE(11)
    !
  CONTAINS
    !
    ! everything that follows the first two blank-separated tokens of a line
    FUNCTION AFTER_TWO(LINE) RESULT(S)
      CHARACTER(LEN=*), INTENT(IN) :: LINE
      CHARACTER(LEN=LNAM) :: S
      INTEGER :: IP, IT, L
      L  = LEN(LINE)
      IP = 1
      DO IT=1,2
        DO WHILE (IP <= L)                  ! skip blanks
          IF (LINE(IP:IP) /= ' ') EXIT
          IP = IP+1
        ENDDO
        DO WHILE (IP <= L)                  ! skip the token
          IF (LINE(IP:IP) == ' ') EXIT
          IP = IP+1
        ENDDO
      ENDDO
      S = ADJUSTL(LINE(MIN(IP,L):))
    END FUNCTION AFTER_TWO
    !
  END PROGRAM RENAME_STRUCTURES
