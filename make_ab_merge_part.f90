! Read in and merge ML_ABN files from different directories. Configurations and
! basis sets are renumbered accordingly. The result is written to ML_AB_merge,
! ready for refitting (no need to use ML_MODE=select unless that is what you
! want to do).
!
! Usage:
!   make_ab_merge_part [dir1 dir2 ...]
!   -each argument is either a directory (then DIR/ML_ABN, or DIR/ML_AB if the
!    former does not exist, is read) or the ML_AB file itself
!   -without arguments the directories in DIRDEF below are used
!   -the merge order is the order of the arguments
!   -for every file the program prints its number of configurations and asks
!    which of them to keep ("0 0" or an empty line = all of them), so any
!    subset of any file can be merged, not only of the last one
!   -the output is named ML_AB_merge, copy/rename it as needed
!
! What the program checks/does for you:
!   -the merged header is rebuilt from scratch: the number of configurations
!    and the numbers of basis sets per atom type are counted, never guessed,
!    and the maxima (atoms per system, atoms per atom type) are taken over all
!    the files
!   -the atom types of all the files are merged by NAME, so the files may
!    contain different species and different numbers of species. The species
!    of a file no longer have to be a subset of the species of the last file
!   -the declared number of configurations of every input file is verified
!    against the configurations actually present, and the basis sets are
!    checked to point to existing configurations
!   -atomic masses and reference energies of a same species are compared
!    between files, and a warning is issued if they differ
!
! NB: the configurations are copied verbatim, so lines of any length and any
!     number of atoms per configuration are handled.
!
! Compile: gfortran -o make_ab_merge_part make_ab_merge_part.f90
!
  PROGRAM MAKE_AB_MERGE_PART
    IMPLICIT NONE
    INTEGER, PARAMETER :: DP = KIND(1D0)
    INTEGER, PARAMETER :: LLIN = 1024   ! max. length of a line of the ML_AB files
    INTEGER, PARAMETER :: LNAM = 8      ! max. length of a species name
    INTEGER, PARAMETER :: IUIN = 10, IUOUT = 11, IUBOD = 12
    CHARACTER(LEN=*), PARAMETER :: OUTFILE = 'ML_AB_merge'
    ! directories used when the program is called without arguments
    CHARACTER(LEN=*), PARAMETER :: DIRDEF(*) = [CHARACTER(LEN=64) :: 'data1', 'data2']
    !
    ! everything that is read from one ML_AB file
    TYPE :: FILE_T
      CHARACTER(LEN=256)            :: PATH        ! file actually read
      CHARACTER(LEN=64)             :: VERS        ! version line
      INTEGER                       :: NCONF       ! no. of configurations
      INTEGER                       :: NTYP        ! no. of species
      INTEGER                       :: MAT         ! max. no. of atoms per system
      INTEGER                       :: MATT        ! max. no. of atoms per atom type
      CHARACTER(LEN=LNAM), ALLOCATABLE :: CTYP(:)  ! species names
      REAL(DP),            ALLOCATABLE :: EATOM(:) ! reference atomic energies
      REAL(DP),            ALLOCATABLE :: POMASS(:)! atomic masses
      INTEGER,             ALLOCATABLE :: NB(:)    ! no. of basis sets per species
      INTEGER,             ALLOCATABLE :: ISTR(:,:)! configuration of a basis set
      INTEGER,             ALLOCATABLE :: IB(:,:)  ! atom index of a basis set
      INTEGER,             ALLOCATABLE :: ITG(:)   ! local species -> global species
      INTEGER,             ALLOCATABLE :: MAP(:)   ! old config. -> new config. (0=dropped)
      INTEGER                       :: I1, I2      ! range of configurations kept
      INTEGER                       :: NKEEP       ! no. of configurations kept
    END TYPE FILE_T
    TYPE(FILE_T), ALLOCATABLE :: F(:)
    !
    ! merged (global) data
    INTEGER                          :: NFILE, NTYPG, MATG, MATTG, NCONFG
    CHARACTER(LEN=LNAM), ALLOCATABLE :: CTYPG(:)
    REAL(DP),            ALLOCATABLE :: EATOMG(:), POMASSG(:)
    INTEGER,             ALLOCATABLE :: NBTOT(:)
    !
    INTEGER            :: IFILE, ITYP, JTYP, I, J, IOS, IWRITTEN, NARG
    CHARACTER(LEN=256) :: ARG
    CHARACTER(LEN=LLIN):: LINE
    !
    ! -----------------------------------------------------------------------
    ! 1) which files to merge
    ! -----------------------------------------------------------------------
    NARG = COMMAND_ARGUMENT_COUNT()
    IF (NARG > 0) THEN
      NFILE = NARG
    ELSE
      NFILE = SIZE(DIRDEF)
    ENDIF
    IF (NFILE < 1) ERROR STOP 'Error: no ML_AB file to merge'
    ALLOCATE(F(NFILE))
    DO IFILE=1,NFILE
      IF (NARG > 0) THEN
        CALL GET_COMMAND_ARGUMENT(IFILE, ARG)
      ELSE
        ARG = DIRDEF(IFILE)
      ENDIF
      F(IFILE)%PATH = RESOLVE_PATH(ARG)
    ENDDO
    IF (NFILE == 1) WRITE(*,*) 'Warning: only one file given, ML_AB_merge will be a copy of it'
    !
    ! -----------------------------------------------------------------------
    ! 2) read the header and the basis sets of every file, check the file
    ! -----------------------------------------------------------------------
    DO IFILE=1,NFILE
      WRITE(*,*)
      WRITE(*,*) 'Reading ', TRIM(F(IFILE)%PATH)
      CALL READ_HEADER(F(IFILE))
      CALL CHECK_FILE(F(IFILE))
      WRITE(*,'(1x,a,i8,a,i4,a,999(1x,a))') 'configurations = ', F(IFILE)%NCONF, &
            ', species = ', F(IFILE)%NTYP, ':', (TRIM(F(IFILE)%CTYP(I)), I=1,F(IFILE)%NTYP)
      WRITE(*,*) '   basis sets per species = ', F(IFILE)%NB(1:F(IFILE)%NTYP)
      IF (ALL(F(IFILE)%NB(1:F(IFILE)%NTYP) == 1) .AND. &
          ALL(F(IFILE)%ISTR(1:F(IFILE)%NTYP,1) == 1) .AND. &
          ALL(F(IFILE)%IB(1:F(IFILE)%NTYP,1) == 1)) THEN
        WRITE(*,*) '   Warning: this file has placeholder basis sets (one "1 1" per species).'
        WRITE(*,*) '            They are merged as they are; use ML_MODE=select if you want'
        WRITE(*,*) '            VASP to pick the local reference configurations itself.'
      ENDIF
    ENDDO
    !
    ! -----------------------------------------------------------------------
    ! 3) merge the species lists by name
    ! -----------------------------------------------------------------------
    ALLOCATE(CTYPG(SUM(F(:)%NTYP)), EATOMG(SUM(F(:)%NTYP)), POMASSG(SUM(F(:)%NTYP)))
    NTYPG = 0
    DO IFILE=1,NFILE
      ALLOCATE(F(IFILE)%ITG(F(IFILE)%NTYP))
      DO ITYP=1,F(IFILE)%NTYP
        JTYP = TYPE_INDEX(F(IFILE)%CTYP(ITYP))
        IF (JTYP == 0) THEN
          NTYPG = NTYPG+1
          JTYP  = NTYPG
          CTYPG(JTYP)   = F(IFILE)%CTYP(ITYP)
          EATOMG(JTYP)  = F(IFILE)%EATOM(ITYP)
          POMASSG(JTYP) = F(IFILE)%POMASS(ITYP)
        ELSE
          IF (ABS(POMASSG(JTYP)-F(IFILE)%POMASS(ITYP)) > 1.D-3) &
            WRITE(*,*) 'Warning: different mass for ', TRIM(CTYPG(JTYP)), ' in ', &
                       TRIM(F(IFILE)%PATH), POMASSG(JTYP), F(IFILE)%POMASS(ITYP)
          IF (ABS(EATOMG(JTYP)-F(IFILE)%EATOM(ITYP)) > 1.D-6) &
            WRITE(*,*) 'Warning: different reference energy for ', TRIM(CTYPG(JTYP)), &
                       ' in ', TRIM(F(IFILE)%PATH), EATOMG(JTYP), F(IFILE)%EATOM(ITYP)
        ENDIF
        F(IFILE)%ITG(ITYP) = JTYP
      ENDDO
      IF (ADJUSTL(F(IFILE)%VERS) /= ADJUSTL(F(1)%VERS)) &
        WRITE(*,*) 'Warning: different version line in ', TRIM(F(IFILE)%PATH)
    ENDDO
    MATG  = MAXVAL(F(:)%MAT)
    MATTG = MAXVAL(F(:)%MATT)
    WRITE(*,*)
    WRITE(*,'(1x,a,i4,a,999(1x,a))') 'Merged species list: ', NTYPG, ':', &
          (TRIM(CTYPG(I)), I=1,NTYPG)
    DO IFILE=1,NFILE
      IF (F(IFILE)%NTYP /= NTYPG) &
        WRITE(*,*) 'Note: ', TRIM(F(IFILE)%PATH), ' does not contain all the species'
    ENDDO
    !
    ! -----------------------------------------------------------------------
    ! 4) which configurations to keep from every file
    ! -----------------------------------------------------------------------
    NCONFG = 0
    DO IFILE=1,NFILE
      WRITE(*,*)
      WRITE(*,'(1x,a,a,a,i8)') 'File ', TRIM(F(IFILE)%PATH), ': configurations = ', &
            F(IFILE)%NCONF
      DO
        WRITE(*,'(1x,a)',ADVANCE='no') 'Enter the first and last configuration to keep &
                                       &("0 0" or empty = all): '
        READ(*,'(a)',IOSTAT=IOS) LINE
        IF (IOS /= 0) ERROR STOP 'Error: unexpected end of input'
        IF (LEN_TRIM(LINE) == 0) THEN
          F(IFILE)%I1 = 1
          F(IFILE)%I2 = F(IFILE)%NCONF
          EXIT
        ENDIF
        READ(LINE,*,IOSTAT=IOS) F(IFILE)%I1, F(IFILE)%I2
        IF (IOS /= 0) THEN
          WRITE(*,*) 'Invalid input: enter two integers'
          CYCLE
        ENDIF
        IF (F(IFILE)%I1 == 0 .AND. F(IFILE)%I2 == 0) THEN
          F(IFILE)%I1 = 1
          F(IFILE)%I2 = F(IFILE)%NCONF
          EXIT
        ENDIF
        IF (F(IFILE)%I1 > F(IFILE)%I2) THEN
          I = F(IFILE)%I1
          F(IFILE)%I1 = F(IFILE)%I2
          F(IFILE)%I2 = I
        ENDIF
        IF (F(IFILE)%I1 < 1 .OR. F(IFILE)%I2 > F(IFILE)%NCONF) THEN
          WRITE(*,*) 'Invalid range: the configurations must lie between 1 and ', F(IFILE)%NCONF
          CYCLE
        ENDIF
        EXIT
      ENDDO
      ! old configuration index -> new (merged) configuration index
      ALLOCATE(F(IFILE)%MAP(F(IFILE)%NCONF))
      F(IFILE)%MAP   = 0
      F(IFILE)%NKEEP = 0
      DO I=F(IFILE)%I1,F(IFILE)%I2
        F(IFILE)%NKEEP = F(IFILE)%NKEEP+1
        F(IFILE)%MAP(I) = NCONFG + F(IFILE)%NKEEP
      ENDDO
      NCONFG = NCONFG + F(IFILE)%NKEEP
      WRITE(*,*) '   keeping ', F(IFILE)%NKEEP, ' configurations'
    ENDDO
    IF (NCONFG == 0) ERROR STOP 'Error: no configuration selected'
    !
    ! -----------------------------------------------------------------------
    ! 5) copy the selected configurations to a scratch file (renumbered)
    ! -----------------------------------------------------------------------
    OPEN(IUBOD,STATUS='SCRATCH')
    IWRITTEN = 0
    DO IFILE=1,NFILE
      CALL COPY_CONF(F(IFILE), IWRITTEN)
    ENDDO
    IF (IWRITTEN /= NCONFG) THEN
      WRITE(*,*) 'Error: expected ', NCONFG, ' configurations, wrote ', IWRITTEN
      ERROR STOP 'Error: inconsistent number of configurations'
    ENDIF
    !
    ! -----------------------------------------------------------------------
    ! 6) write ML_AB_merge: header, basis sets, configurations
    ! -----------------------------------------------------------------------
    ALLOCATE(NBTOT(NTYPG))
    NBTOT = 0
    DO IFILE=1,NFILE
      DO ITYP=1,F(IFILE)%NTYP
        JTYP = F(IFILE)%ITG(ITYP)
        DO I=1,F(IFILE)%NB(ITYP)
          IF (F(IFILE)%MAP(F(IFILE)%ISTR(ITYP,I)) > 0) NBTOT(JTYP) = NBTOT(JTYP)+1
        ENDDO
      ENDDO
    ENDDO
    !
    OPEN(IUOUT,FILE=OUTFILE,STATUS='REPLACE')
    WRITE(IUOUT,'(a)') TRIM(F(1)%VERS)
    CALL WHEAD('The number of configurations')
    WRITE(IUOUT,'(i12)') NCONFG
    CALL WHEAD('The maximum number of atom type')
    WRITE(IUOUT,'(i12)') NTYPG
    CALL WHEAD('The atom types in the data file')
    DO ITYP=1,NTYPG,3
      WRITE(IUOUT,'(3(2x,a))') (TRIM(CTYPG(I)), I=ITYP, MIN(ITYP+2,NTYPG))
    ENDDO
    CALL WHEAD('The maximum number of atoms per system')
    WRITE(IUOUT,'(i12)') MATG
    CALL WHEAD('The maximum number of atoms per atom type')
    WRITE(IUOUT,'(i12)') MATTG
    CALL WHEAD('Reference atomic energy (eV)')
    DO ITYP=1,NTYPG,3
      WRITE(IUOUT,*) (EATOMG(I), I=ITYP, MIN(ITYP+2,NTYPG))
    ENDDO
    CALL WHEAD('Atomic mass')
    DO ITYP=1,NTYPG,3
      WRITE(IUOUT,*) (POMASSG(I), I=ITYP, MIN(ITYP+2,NTYPG))
    ENDDO
    CALL WHEAD('The numbers of basis sets per atom type')
    DO ITYP=1,NTYPG,3
      WRITE(IUOUT,'(3i12)') (NBTOT(I), I=ITYP, MIN(ITYP+2,NTYPG))
    ENDDO
    ! basis sets, renumbered, in file order
    DO JTYP=1,NTYPG
      CALL WHEAD('Basis set for '//TRIM(CTYPG(JTYP)))
      DO IFILE=1,NFILE
        DO ITYP=1,F(IFILE)%NTYP
          IF (F(IFILE)%ITG(ITYP) /= JTYP) CYCLE
          DO I=1,F(IFILE)%NB(ITYP)
            J = F(IFILE)%MAP(F(IFILE)%ISTR(ITYP,I))
            IF (J > 0) WRITE(IUOUT,'(2i12)') J, F(IFILE)%IB(ITYP,I)
          ENDDO
        ENDDO
      ENDDO
    ENDDO
    ! configurations
    REWIND(IUBOD)
    DO
      READ(IUBOD,'(a)',IOSTAT=IOS) LINE
      IF (IOS /= 0) EXIT
      WRITE(IUOUT,'(a)') TRIM(LINE)
    ENDDO
    CLOSE(IUBOD)
    CLOSE(IUOUT)
    !
    WRITE(*,*)
    DO IFILE=1,NFILE
      WRITE(*,'(1x,a,a,a,i8,a,i8,a,i8,a,i8)') 'From ', TRIM(F(IFILE)%PATH), ': kept ', &
            F(IFILE)%NKEEP, ' of ', F(IFILE)%NCONF, ', configurations ', F(IFILE)%I1, &
            ' to ', F(IFILE)%I2
    ENDDO
    WRITE(*,*) 'The number of configurations            = ', NCONFG
    WRITE(*,*) 'The numbers of basis sets per atom type = ', (NBTOT(I), I=1,NTYPG)
    WRITE(*,*) 'Written to ', OUTFILE
    !
  CONTAINS
    !
    ! -----------------------------------------------------------------------
    ! Index of a species in the merged list (0 if not there yet).
    ! -----------------------------------------------------------------------
    FUNCTION TYPE_INDEX(CT) RESULT(IND)
      CHARACTER(LEN=*), INTENT(IN) :: CT
      INTEGER :: IND, K
      IND = 0
      DO K=1,NTYPG
        IF (CTYPG(K) == CT) THEN
          IND = K
          EXIT
        ENDIF
      ENDDO
    END FUNCTION TYPE_INDEX
    !
    ! -----------------------------------------------------------------------
    ! Turn a command-line argument into the ML_AB file to read: the argument
    ! itself if it is a file, otherwise ARG/ML_ABN or ARG/ML_AB.
    ! -----------------------------------------------------------------------
    FUNCTION RESOLVE_PATH(ARGIN) RESULT(PATH)
      CHARACTER(LEN=*), INTENT(IN) :: ARGIN
      CHARACTER(LEN=256) :: PATH, DIRN
      DIRN = ADJUSTL(ARGIN)
      ! a directory is tried first, so that "data1" finds data1/ML_ABN even
      ! though INQUIRE(EXIST=) is true for the directory itself
      PATH = TRIM(DIRN)//'/ML_ABN'
      IF (READABLE(PATH)) RETURN
      PATH = TRIM(DIRN)//'/ML_AB'
      IF (READABLE(PATH)) RETURN
      PATH = DIRN
      IF (READABLE(PATH)) RETURN
      WRITE(*,*) 'Error: found none of ', TRIM(DIRN)//'/ML_ABN', ', ', &
                 TRIM(DIRN)//'/ML_AB', ', ', TRIM(DIRN)
      ERROR STOP 'Error: input file not found'
    END FUNCTION RESOLVE_PATH
    !
    ! -----------------------------------------------------------------------
    ! .TRUE. if the path is a file from which a line can be read (INQUIRE
    ! alone is not enough: it is also true for a directory).
    ! -----------------------------------------------------------------------
    FUNCTION READABLE(PATH) RESULT(LOK)
      CHARACTER(LEN=*), INTENT(IN) :: PATH
      LOGICAL :: LOK, LEX
      INTEGER :: IOS2
      CHARACTER(LEN=LLIN) :: LIN
      LOK = .FALSE.
      INQUIRE(FILE=TRIM(PATH), EXIST=LEX)
      IF (.NOT.LEX) RETURN
      OPEN(IUIN,FILE=TRIM(PATH),STATUS='OLD',ACTION='READ',IOSTAT=IOS2)
      IF (IOS2 /= 0) RETURN
      READ(IUIN,'(a)',IOSTAT=IOS2) LIN
      CLOSE(IUIN)
      LOK = (IOS2 == 0)
    END FUNCTION READABLE
    !
    ! -----------------------------------------------------------------------
    ! Write a '****' / title / '----' header block to the output.
    ! -----------------------------------------------------------------------
    SUBROUTINE WHEAD(TITLE)
      CHARACTER(LEN=*), INTENT(IN) :: TITLE
      WRITE(IUOUT,'(a)') REPEAT('*', 50)
      WRITE(IUOUT,'(5x,a)') TITLE
      WRITE(IUOUT,'(a)') REPEAT('-', 50)
    END SUBROUTINE WHEAD
    !
    ! -----------------------------------------------------------------------
    ! Read the header and the basis sets of one ML_AB file. The blocks are
    ! located by their title, not by their line number, and the values are
    ! read list-directed so that they may be spread over several lines.
    ! -----------------------------------------------------------------------
    SUBROUTINE READ_HEADER(FF)
      TYPE(FILE_T), INTENT(INOUT) :: FF
      CHARACTER(LEN=LLIN) :: LIN
      INTEGER :: IOS2, K, ITY, MXB
      LOGICAL :: LF(8)
      !
      LF = .FALSE.
      OPEN(IUIN,FILE=TRIM(FF%PATH),STATUS='OLD',IOSTAT=IOS2)
      IF (IOS2 /= 0) THEN
        WRITE(*,*) 'Error: could not open ', TRIM(FF%PATH)
        ERROR STOP 'Error: could not open an ML_AB file'
      ENDIF
      READ(IUIN,'(a)',IOSTAT=IOS2) LIN
      IF (IOS2 /= 0) ERROR STOP 'Error: empty ML_AB file'
      FF%VERS = LIN(1:LEN(FF%VERS))   ! kept verbatim, it is copied to the output
      DO
        READ(IUIN,'(a)',IOSTAT=IOS2) LIN
        IF (IOS2 /= 0) EXIT
        IF (INDEX(LIN,'The number of configurations') > 0) THEN
          CALL SKIP_DASH()
          READ(IUIN,*,IOSTAT=IOS2) FF%NCONF
          CALL CHK(IOS2,'the number of configurations')
          LF(1) = .TRUE.
        ELSEIF (INDEX(LIN,'The maximum number of atom type') > 0) THEN
          CALL SKIP_DASH()
          READ(IUIN,*,IOSTAT=IOS2) FF%NTYP
          CALL CHK(IOS2,'the number of atom types')
          IF (FF%NTYP < 1) ERROR STOP 'Error: non-positive number of atom types'
          ALLOCATE(FF%CTYP(FF%NTYP), FF%EATOM(FF%NTYP), FF%POMASS(FF%NTYP), FF%NB(FF%NTYP))
          FF%EATOM  = 0._DP
          FF%POMASS = 0._DP
          FF%NB     = 0
          LF(2) = .TRUE.
        ELSEIF (INDEX(LIN,'The atom types in the data file') > 0) THEN
          CALL NEEDTYP(LF(2))
          CALL SKIP_DASH()
          READ(IUIN,*,IOSTAT=IOS2) (FF%CTYP(K), K=1,FF%NTYP)
          CALL CHK(IOS2,'the atom types')
          LF(3) = .TRUE.
        ELSEIF (INDEX(LIN,'The maximum number of atoms per system') > 0) THEN
          CALL SKIP_DASH()
          READ(IUIN,*,IOSTAT=IOS2) FF%MAT
          CALL CHK(IOS2,'the max. number of atoms per system')
          LF(4) = .TRUE.
        ELSEIF (INDEX(LIN,'The maximum number of atoms per atom type') > 0) THEN
          CALL SKIP_DASH()
          READ(IUIN,*,IOSTAT=IOS2) FF%MATT
          CALL CHK(IOS2,'the max. number of atoms per atom type')
          LF(5) = .TRUE.
        ELSEIF (INDEX(LIN,'Reference atomic energy') > 0) THEN
          CALL NEEDTYP(LF(2))
          CALL SKIP_DASH()
          READ(IUIN,*,IOSTAT=IOS2) (FF%EATOM(K), K=1,FF%NTYP)
          CALL CHK(IOS2,'the reference atomic energies')
          LF(6) = .TRUE.
        ELSEIF (INDEX(LIN,'Atomic mass') > 0) THEN
          CALL NEEDTYP(LF(2))
          CALL SKIP_DASH()
          READ(IUIN,*,IOSTAT=IOS2) (FF%POMASS(K), K=1,FF%NTYP)
          CALL CHK(IOS2,'the atomic masses')
          LF(7) = .TRUE.
        ELSEIF (INDEX(LIN,'The numbers of basis sets per atom type') > 0) THEN
          CALL NEEDTYP(LF(2))
          CALL SKIP_DASH()
          READ(IUIN,*,IOSTAT=IOS2) (FF%NB(K), K=1,FF%NTYP)
          CALL CHK(IOS2,'the numbers of basis sets')
          IF (ANY(FF%NB < 0)) ERROR STOP 'Error: negative number of basis sets'
          MXB = MAX(1, MAXVAL(FF%NB))
          ALLOCATE(FF%ISTR(FF%NTYP,MXB), FF%IB(FF%NTYP,MXB))
          FF%ISTR = 0
          FF%IB   = 0
          DO ITY=1,FF%NTYP
            READ(IUIN,'(a)',IOSTAT=IOS2) LIN   ! '****' separator
            READ(IUIN,'(a)',IOSTAT=IOS2) LIN   ! 'Basis set for X'
            IF (IOS2 == 0 .AND. INDEX(LIN,'Basis set for') == 0) THEN
              WRITE(*,*) 'Error: expected a "Basis set for" line, found: ', TRIM(LIN)
              ERROR STOP 'Error: unexpected basis set section'
            ENDIF
            READ(IUIN,'(a)',IOSTAT=IOS2) LIN   ! '----' separator
            CALL CHK(IOS2,'the basis set section')
            DO K=1,FF%NB(ITY)
              READ(IUIN,*,IOSTAT=IOS2) FF%ISTR(ITY,K), FF%IB(ITY,K)
              CALL CHK(IOS2,'a basis set')
            ENDDO
          ENDDO
          LF(8) = .TRUE.
          EXIT
        ENDIF
      ENDDO
      CLOSE(IUIN)
      IF (.NOT.LF(1)) ERROR STOP 'Error: "The number of configurations" not found'
      IF (.NOT.LF(2)) ERROR STOP 'Error: "The maximum number of atom type" not found'
      IF (.NOT.LF(3)) ERROR STOP 'Error: "The atom types in the data file" not found'
      IF (.NOT.LF(4)) ERROR STOP 'Error: "The maximum number of atoms per system" not found'
      IF (.NOT.LF(5)) ERROR STOP 'Error: "The maximum number of atoms per atom type" not found'
      IF (.NOT.LF(6)) WRITE(*,*) 'Warning: no reference atomic energy found, using 0'
      IF (.NOT.LF(7)) WRITE(*,*) 'Warning: no atomic mass found, using 0'
      IF (.NOT.LF(8)) ERROR STOP 'Error: basis set section not found'
      IF (FF%NCONF < 1) ERROR STOP 'Error: non-positive number of configurations'
    END SUBROUTINE READ_HEADER
    !
    SUBROUTINE SKIP_DASH()
      CHARACTER(LEN=LLIN) :: LIN
      INTEGER :: IOS2
      READ(IUIN,'(a)',IOSTAT=IOS2) LIN
      IF (IOS2 /= 0) ERROR STOP 'Error: unexpected end of an ML_AB file'
    END SUBROUTINE SKIP_DASH
    !
    SUBROUTINE CHK(IOS2, WHAT)
      INTEGER,          INTENT(IN) :: IOS2
      CHARACTER(LEN=*), INTENT(IN) :: WHAT
      IF (IOS2 /= 0) THEN
        WRITE(*,*) 'Error: could not read ', WHAT
        ERROR STOP 'Error while reading an ML_AB file'
      ENDIF
    END SUBROUTINE CHK
    !
    SUBROUTINE NEEDTYP(LSET)
      LOGICAL, INTENT(IN) :: LSET
      IF (.NOT.LSET) ERROR STOP 'Error: the number of atom types must come first in ML_AB'
    END SUBROUTINE NEEDTYP
    !
    ! -----------------------------------------------------------------------
    ! Check that the configurations announced in the header are really there
    ! and that the basis sets point to existing configurations.
    ! -----------------------------------------------------------------------
    SUBROUTINE CHECK_FILE(FF)
      TYPE(FILE_T), INTENT(IN) :: FF
      CHARACTER(LEN=LLIN) :: LIN
      INTEGER :: IOS2, IP, IC, NC, ITY, K
      !
      DO ITY=1,FF%NTYP
        DO K=1,FF%NB(ITY)
          IF (FF%ISTR(ITY,K) < 1 .OR. FF%ISTR(ITY,K) > FF%NCONF) THEN
            WRITE(*,*) 'Error: basis set ', K, ' of species ', ITY, ' points to configuration ', &
                       FF%ISTR(ITY,K), ' but the file has ', FF%NCONF
            ERROR STOP 'Error: basis set out of range'
          ENDIF
        ENDDO
      ENDDO
      NC = 0
      OPEN(IUIN,FILE=TRIM(FF%PATH),STATUS='OLD',IOSTAT=IOS2)
      IF (IOS2 /= 0) ERROR STOP 'Error: could not reopen an ML_AB file'
      DO
        READ(IUIN,'(a)',IOSTAT=IOS2) LIN
        IF (IOS2 /= 0) EXIT
        IP = INDEX(LIN,'Configuration num.')
        IF (IP > 0) THEN
          READ(LIN(IP+18:),*,IOSTAT=IOS2) IC
          CALL CHK(IOS2,'a configuration number')
          NC = NC+1
          IF (IC /= NC) THEN
            WRITE(*,*) 'Error: configuration ', NC, ' is numbered ', IC, ' in ', TRIM(FF%PATH)
            ERROR STOP 'Error: configurations are not numbered 1,2,3,...'
          ENDIF
        ENDIF
      ENDDO
      CLOSE(IUIN)
      IF (NC /= FF%NCONF) THEN
        WRITE(*,*) 'Error: ', TRIM(FF%PATH), ' announces ', FF%NCONF, &
                   ' configurations but contains ', NC
        ERROR STOP 'Error: wrong number of configurations in an ML_AB file'
      ENDIF
    END SUBROUTINE CHECK_FILE
    !
    ! -----------------------------------------------------------------------
    ! Copy the selected configurations of one file to the scratch unit, with
    ! their new configuration numbers. Everything else is copied verbatim.
    ! The '****' separators are held back and written in front of the next
    ! kept configuration, so that no separator is lost or left over.
    ! -----------------------------------------------------------------------
    SUBROUTINE COPY_CONF(FF, NW)
      TYPE(FILE_T), INTENT(IN)    :: FF
      INTEGER,      INTENT(INOUT) :: NW
      CHARACTER(LEN=LLIN) :: LIN, LADJ
      INTEGER :: IOS2, IP, IC
      LOGICAL :: LWRITE, LPEND
      !
      OPEN(IUIN,FILE=TRIM(FF%PATH),STATUS='OLD',IOSTAT=IOS2)
      IF (IOS2 /= 0) ERROR STOP 'Error: could not reopen an ML_AB file'
      LWRITE = .FALSE.
      LPEND  = .FALSE.
      DO
        READ(IUIN,'(a)',IOSTAT=IOS2) LIN
        IF (IOS2 /= 0) EXIT
        IP = INDEX(LIN,'Configuration num.')
        IF (IP > 0) THEN
          READ(LIN(IP+18:),*,IOSTAT=IOS2) IC
          CALL CHK(IOS2,'a configuration number')
          IF (FF%MAP(IC) > 0) THEN
            IF (LPEND) WRITE(IUBOD,'(a)') REPEAT('*', 50)
            WRITE(IUBOD,'(a,i7)') '     Configuration num.', FF%MAP(IC)
            NW = NW+1
            LWRITE = .TRUE.
          ELSE
            LWRITE = .FALSE.
          ENDIF
          LPEND = .FALSE.
        ELSE
          ! a '****' line may be the separator in front of the next
          ! configuration or a separator inside the current one
          IF (LPEND .AND. LWRITE) WRITE(IUBOD,'(a)') REPEAT('*', 50)
          LPEND = .FALSE.
          LADJ = ADJUSTL(LIN)
          IF (LADJ(1:4) == '****') THEN
            LPEND = .TRUE.
          ELSEIF (LWRITE) THEN
            WRITE(IUBOD,'(a)') TRIM(LIN)
          ENDIF
        ENDIF
      ENDDO
      CLOSE(IUIN)
    END SUBROUTINE COPY_CONF
    !
  END PROGRAM MAKE_AB_MERGE_PART
