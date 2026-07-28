! Create a new ML_AB file with structures, energies, forces and stress tensors
! read from CONF/OUTCAR* files (OUTCAR1, OUTCAR2, ...). Use to train a MLFF
! with ML_MODE=select.
!
! What the program does:
!   -the atom types, the number of atom types, the number of atoms per type,
!    the atomic masses and the number of ions are DETECTED from the OUTCAR
!    files themselves; nothing is hardcoded. Different OUTCAR files may
!    contain different systems/compositions: the header of ML_AB gets the
!    union of all atom types found, and each configuration gets the types and
!    the atom numbers of its own OUTCAR
!   -the number of configurations (line 5 of ML_AB) is written automatically,
!    so no manual editing is needed. The configurations are first written to a
!    scratch file and copied to ML_AB once the final count is known
!   -from each OUTCAR only the LAST structure plus NBACK structures further
!    back are stored (NBACK=0 -> last structure only, NBACK=2 -> the last 3
!    ionic steps, etc.). If an OUTCAR contains fewer steps, all of them are
!    taken
!   -the system name of each configuration is chosen in one of two ways:
!      1) from a names file (default CONF/names.dat): line i is the name of
!         CONF/OUTCARi (blank lines and lines starting with '#' are skipped;
!         names may contain blanks)
!      2) from ranges of OUTCAR numbers entered interactively, e.g.
!         "1 100 graphite", one range per line, "0 0 end" to finish
!         (names entered this way must not contain blanks)
!    Configurations are written GROUPED by name, in the order in which the
!    names first appear.
!
! NB: hardcoded CTIFOR (see below), the folder is CONF and the files are
!     CONF/OUTCAR1, CONF/OUTCAR2, ... (consecutive numbering, the scan stops
!     at the first missing file)
!
! Compile: gfortran -o make_ab_outcars make_ab_outcars.f90
!
      PROGRAM MAKE_AB
        IMPLICIT NONE
! Parameters.
        INTEGER, PARAMETER :: DP   = KIND(1D0)
        INTEGER, PARAMETER :: MXT  = 50   ! max. no. of atom types per OUTCAR
        INTEGER, PARAMETER :: LNM  = 40   ! length of a system name
        INTEGER, PARAMETER :: NWRITE = 3  ! entries per line in the ML_AB header
        INTEGER, PARAMETER :: IUAB = 10, IUOUT = 11, IUENE = 12, IUBOD = 13, &
                              IUNAM = 14
        REAL(DP), PARAMETER :: CTIFOR = 0.002_DP  ! CTIFOR value to write
! Formats (kept as named strings so that the internal routines can use them).
        CHARACTER(LEN=*), PARAMETER :: &
          FM_NCONF  = "(50('*')/5X,'The number of configurations'/50('-')/i6)"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_MXTYP  = "(50('*')/5X,'The maximum number of atom type'/50('-')/i6)"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_TYPES  = "(50('*')/5X,'The atom types in the data file'/50('-'))"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_MXIONS = "(50('*')/5X,'The maximum number of atoms per system'/50('-')/i6)"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_MXPTYP = "(50('*')/5X,'The maximum number of atoms per atom type'/50('-')/i6)"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_EATOM  = "(50('*')/5X,'Reference atomic energy (eV)'/50('-'))"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_MASS   = "(50('*')/5X,'Atomic mass'/50('-'))"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_NBASIS = "(50('*')/5X,'The numbers of basis sets per atom type'/50('-'))"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_BASIS  = "(50('*')/5X,'Basis set for ',a/50('-'))"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_CONF   = "(50('*')/5X,'Configuration num. ',i6)"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_SNAME  = "(50('=')/5X,'System name'/50('-')/5X,a40)"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_NATYP  = "(50('=')/5X,'The number of atom types'/50('-')/5X,i6)"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_NIONS  = "(50('=')/5X,'The number of atoms'/50('-')/i6)"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_ATNUM  = "(50('*')/5X,'Atom types and atom numbers'/50('-'))"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_CTIFOR = "(50('=')/5X,'CTIFOR'/50('-'))"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_LATT   = "(50('=')/5X,'Primitive lattice vectors (ang.)'/50('-'))"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_POS    = "(50('=')/5X,'Atomic positions (ang.)'/50('-'))"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_ENE    = "(50('=')/5X,'Total energy (eV)'/50('-'))"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_FOR    = "(50('=')/5X,'Forces (eV ang.^-1)'/50('-'))"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_STR1   = "(50('=')/5X,'Stress (kbar)'/50('-')/5X,'XX YY ZZ'/50('-'))"
        CHARACTER(LEN=*), PARAMETER :: &
          FM_STR2   = "(50('-')/5X,'XY YZ ZX'/50('-'))"
! Global data (union over all OUTCAR files).
        INTEGER                     :: NTYP        ! total no. of atom types
        CHARACTER(LEN=3)            :: CTYP(MXT)   ! atom types
        REAL(DP)                    :: POMASS(MXT) ! atomic masses
        INTEGER                     :: MITYP       ! max no. of atoms per type
        INTEGER                     :: MXIONS      ! max no. of atoms per system
! Per OUTCAR file data.
        INTEGER                          :: NFILES
        INTEGER,          ALLOCATABLE    :: FNTYP(:)     ! no. of types
        INTEGER,          ALLOCATABLE    :: FNITYP(:,:)  ! atoms per type
        INTEGER,          ALLOCATABLE    :: FNIONS(:)    ! no. of atoms
        INTEGER,          ALLOCATABLE    :: FGRP(:)      ! name group
        CHARACTER(LEN=3), ALLOCATABLE    :: FCTYP(:,:)   ! atom types
        CHARACTER(LEN=LNM), ALLOCATABLE  :: FNAM(:)      ! system name
        CHARACTER(LEN=LNM), ALLOCATABLE  :: GNAM(:)      ! distinct names
        INTEGER                          :: NGRP
! Work variables.
        INTEGER            :: NBACK   ! extra structures back from the last one
        INTEGER            :: NBUF    ! = NBACK+1 structures kept per OUTCAR
        INTEGER            :: ICOUNT  ! configurations written to ML_AB
        INTEGER            :: ICONFT  ! configurations found in all OUTCARs
        INTEGER            :: IMODE   ! naming mode
        INTEGER            :: I, ITYP, IFILE, IG, IOS, IT, NC
        INTEGER            :: NTF, NITF(MXT)
        REAL(DP)           :: PMF(MXT)
        CHARACTER(LEN=3)   :: CTF(MXT)
        CHARACTER(LEN=200) :: TMPFILE
        CHARACTER(LEN=500) :: BUF
!
! ---------------------------------------------------------------------------
! 1) how many OUTCAR files are there
! ---------------------------------------------------------------------------
        NFILES = 0
        DO
          WRITE(TMPFILE,'(a,i0)') 'CONF/OUTCAR', NFILES+1
          OPEN(IUOUT,FILE=TMPFILE,STATUS='OLD',IOSTAT=IOS)
          IF (IOS /= 0) EXIT
          CLOSE(IUOUT)
          NFILES = NFILES+1
        ENDDO
        IF (NFILES == 0) ERROR STOP 'Error: no CONF/OUTCAR1 file found'
        WRITE(*,*) 'Number of OUTCAR files found = ', NFILES
!
        ALLOCATE(FNTYP(NFILES), FNIONS(NFILES), FGRP(NFILES))
        ALLOCATE(FNITYP(MXT,NFILES), FCTYP(MXT,NFILES))
        ALLOCATE(FNAM(NFILES), GNAM(NFILES))
        FNITYP = 0
        FCTYP  = ' '
!
! ---------------------------------------------------------------------------
! 2) detect atom types, atom numbers and masses in every OUTCAR
! ---------------------------------------------------------------------------
        NTYP   = 0
        CTYP   = ' '
        POMASS = 0._DP
        MITYP  = 0
        MXIONS = 0
        DO IFILE=1,NFILES
          WRITE(TMPFILE,'(a,i0)') 'CONF/OUTCAR', IFILE
          CALL SCAN_HEADER(TMPFILE, NTF, CTF, NITF, PMF)
          FNTYP(IFILE)  = NTF
          FNIONS(IFILE) = SUM(NITF(1:NTF))
          DO ITYP=1,NTF
            FCTYP(ITYP,IFILE)  = CTF(ITYP)
            FNITYP(ITYP,IFILE) = NITF(ITYP)
            ! add the type to the global list if it is new
            IT = TYPE_INDEX(CTF(ITYP))
            IF (IT == 0) THEN
              IF (NTYP == MXT) ERROR STOP 'Error: too many atom types, increase MXT'
              NTYP = NTYP+1
              IT   = NTYP
              CTYP(IT)   = CTF(ITYP)
              POMASS(IT) = PMF(ITYP)
            ELSEIF (ABS(POMASS(IT)-PMF(ITYP)) > 1.D-3) THEN
              WRITE(*,*) 'Warning: different masses for type ', TRIM(CTF(ITYP)), &
                         ' in ', TRIM(TMPFILE), POMASS(IT), PMF(ITYP)
            ENDIF
            MITYP = MAX(MITYP, NITF(ITYP))
          ENDDO
          MXIONS = MAX(MXIONS, FNIONS(IFILE))
          WRITE(*,'(1x,a,a,i4,a,i6)') TRIM(TMPFILE), ': atom types = ', NTF, &
                                      ', atoms = ', FNIONS(IFILE)
          WRITE(*,*) '   ', (TRIM(CTF(ITYP))//' ', NITF(ITYP), ITYP=1,NTF)
        ENDDO
        WRITE(*,*) 'Total number of atom types found = ', NTYP
        WRITE(*,*) 'Atom types                       = ', (TRIM(CTYP(I))//' ', I=1,NTYP)
        WRITE(*,*) 'Max. atoms per system            = ', MXIONS
        WRITE(*,*) 'Max. atoms per atom type         = ', MITYP
!
! ---------------------------------------------------------------------------
! 3) how many structures to take from each OUTCAR
! ---------------------------------------------------------------------------
        DO
          WRITE(*,*) 'Enter the number of extra structures to take back from the last one'
          WRITE(*,*) '(0 = last structure only, n = last structure + n steps back):'
          READ(*,*,IOSTAT=IOS) NBACK
          IF (IOS == 0 .AND. NBACK >= 0) EXIT
          IF (IOS < 0) ERROR STOP 'Error: unexpected end of input'
          WRITE(*,*) 'Invalid input: enter a non-negative integer'
        ENDDO
        NBUF = NBACK+1
        WRITE(*,*) 'Structures taken per OUTCAR (at most) = ', NBUF
!
! ---------------------------------------------------------------------------
! 4) system names
! ---------------------------------------------------------------------------
        DO
          WRITE(*,*) 'Choose how the system names are assigned:'
          WRITE(*,*) '  1 = one name per OUTCAR read from a names file'
          WRITE(*,*) '  2 = names given by ranges of OUTCAR numbers'
          READ(*,*,IOSTAT=IOS) IMODE
          IF (IOS == 0 .AND. (IMODE == 1 .OR. IMODE == 2)) EXIT
          IF (IOS < 0) ERROR STOP 'Error: unexpected end of input'
          WRITE(*,*) 'Invalid input: enter 1 or 2'
        ENDDO
        IF (IMODE == 1) THEN
          CALL NAMES_FROM_FILE
        ELSE
          CALL NAMES_FROM_RANGES
        ENDIF
        ! distinct names, in order of first appearance
        NGRP = 0
        FGRP = 0
        DO IFILE=1,NFILES
          DO IG=1,NGRP
            IF (GNAM(IG) == FNAM(IFILE)) THEN
              FGRP(IFILE) = IG
              EXIT
            ENDIF
          ENDDO
          IF (FGRP(IFILE) == 0) THEN
            NGRP = NGRP+1
            GNAM(NGRP)  = FNAM(IFILE)
            FGRP(IFILE) = NGRP
          ENDIF
        ENDDO
        WRITE(*,*) 'Number of different system names = ', NGRP
        DO IG=1,NGRP
          WRITE(*,'(1x,a,i4,a,a,a,i6)') 'group ', IG, ' "', TRIM(GNAM(IG)), &
                                        '", OUTCAR files: ', COUNT(FGRP == IG)
        ENDDO
!
! ---------------------------------------------------------------------------
! 5) read the OUTCAR files and write the configurations (grouped by name)
! ---------------------------------------------------------------------------
        OPEN(IUBOD,STATUS='SCRATCH')
        OPEN(IUENE,FILE='energy.dat',STATUS='UNKNOWN')
        ICOUNT = 0
        ICONFT = 0
        DO IG=1,NGRP
          DO IFILE=1,NFILES
            IF (FGRP(IFILE) /= IG) CYCLE
            CALL PROCESS_FILE(IFILE, NC)
            ICONFT = ICONFT+NC
          ENDDO
        ENDDO
        CLOSE(IUENE)
        IF (ICOUNT == 0) ERROR STOP 'Error: no configuration could be read from the OUTCAR files'
!
! ---------------------------------------------------------------------------
! 6) write ML_AB: header (with the correct no. of configurations) + body
! ---------------------------------------------------------------------------
        OPEN(IUAB,FILE='ML_AB',STATUS='UNKNOWN')
        WRITE(IUAB,*) ' 1.0 Version'
        WRITE(IUAB,FM_NCONF) ICOUNT
        WRITE(IUAB,FM_MXTYP) NTYP
        WRITE(IUAB,FM_TYPES)
        DO ITYP=1,NTYP,NWRITE
          WRITE(IUAB,*) (CTYP(I), I=ITYP, MIN(ITYP+NWRITE-1, NTYP))
        ENDDO
        WRITE(IUAB,FM_MXIONS) MXIONS
        WRITE(IUAB,FM_MXPTYP) MITYP
        WRITE(IUAB,FM_EATOM)
        DO ITYP=1,NTYP,NWRITE
          WRITE(IUAB,*) (0.0, I=ITYP, MIN(ITYP+NWRITE-1, NTYP)) ! EATOM
        ENDDO
        WRITE(IUAB,FM_MASS)
        DO ITYP=1,NTYP,NWRITE
          WRITE(IUAB,*) (POMASS(I), I=ITYP, MIN(ITYP+NWRITE-1, NTYP))
        ENDDO
        WRITE(IUAB,FM_NBASIS)
        WRITE(IUAB,*) (1, ITYP=1,NTYP)        ! basis sets per atom type (dummy)
        DO ITYP=1,NTYP
          WRITE(IUAB,FM_BASIS) TRIM(CTYP(ITYP))
          WRITE(IUAB,*) 1, 1                  ! basis sets (dummy)
        ENDDO
        ! copy the configurations
        REWIND(IUBOD)
        DO
          READ(IUBOD,'(a)',IOSTAT=IOS) BUF
          IF (IOS /= 0) EXIT
          WRITE(IUAB,'(a)') TRIM(BUF)
        ENDDO
        CLOSE(IUBOD)
        CLOSE(IUAB)
!
        WRITE(*,*) 'Configurations found in the OUTCAR files = ', ICONFT
        WRITE(*,*) 'Configurations written to ML_AB          = ', ICOUNT
        WRITE(*,*) 'The number of configurations in ML_AB (line 5) is already set'
!
        DEALLOCATE(FNTYP, FNIONS, FGRP, FNITYP, FCTYP, FNAM, GNAM)
!
      CONTAINS
!
! ---------------------------------------------------------------------------
! Index of an atom type in the global list (0 if not present yet).
! ---------------------------------------------------------------------------
      FUNCTION TYPE_INDEX(CT) RESULT(IND)
        CHARACTER(LEN=*), INTENT(IN) :: CT
        INTEGER :: IND, J
        IND = 0
        DO J=1,NTYP
          IF (CTYP(J) == CT) THEN
            IND = J
            EXIT
          ENDIF
        ENDDO
      END FUNCTION TYPE_INDEX
!
! ---------------------------------------------------------------------------
! Substring of LINE after the first '=' (blank if there is none).
! ---------------------------------------------------------------------------
      FUNCTION AFTER_EQ(LINE) RESULT(S)
        CHARACTER(LEN=*), INTENT(IN) :: LINE
        CHARACTER(LEN=200) :: S
        INTEGER :: IP
        IP = INDEX(LINE,'=')
        IF (IP > 0) THEN
          S = LINE(IP+1:)
        ELSE
          S = ' '
        ENDIF
      END FUNCTION AFTER_EQ
!
! ---------------------------------------------------------------------------
! Number of integers that can be read from a string (at most MXT).
! ---------------------------------------------------------------------------
      FUNCTION COUNT_INTS(S) RESULT(N)
        CHARACTER(LEN=*), INTENT(IN) :: S
        INTEGER :: N, K, IOS2, IDUM(MXT), J
        N = 0
        DO K=1,MXT
          READ(S,*,IOSTAT=IOS2) (IDUM(J), J=1,K)
          IF (IOS2 /= 0) EXIT
          N = K
        ENDDO
      END FUNCTION COUNT_INTS
!
! ---------------------------------------------------------------------------
! Read atom types, atoms per type and masses from the header of an OUTCAR.
! The atom types are taken from the VRHFIN (or TITEL) lines of the POTCARs,
! their number and the number of atoms from the 'ions per type' line.
! ---------------------------------------------------------------------------
      SUBROUTINE SCAN_HEADER(FILEN, NTF, CTF, NITF, PMF)
        CHARACTER(LEN=*), INTENT(IN)  :: FILEN
        INTEGER,          INTENT(OUT) :: NTF
        CHARACTER(LEN=3), INTENT(OUT) :: CTF(MXT)
        INTEGER,          INTENT(OUT) :: NITF(MXT)
        REAL(DP),         INTENT(OUT) :: PMF(MXT)
        INTEGER            :: NV, NT2, IOS2, J, IP
        CHARACTER(LEN=3)   :: CV(MXT), CT2(MXT)
        CHARACTER(LEN=200) :: LINE2, SUB, SMASS
        CHARACTER(LEN=32)  :: ADUM, CEL
        LOGICAL            :: LF(2)
!
        NTF  = 0
        CTF  = ' '
        NITF = 0
        PMF  = 0._DP
        NV    = 0
        NT2   = 0
        SMASS = ' '
        LF    = .FALSE.
        OPEN(IUOUT,FILE=FILEN,STATUS='OLD',IOSTAT=IOS2)
        IF (IOS2 /= 0) ERROR STOP 'Error: could not open an OUTCAR file'
        DO
          READ(IUOUT,'(a)',IOSTAT=IOS2) LINE2
          IF (IOS2 /= 0) EXIT
          ! atom types: 'VRHFIN =B: s2p1'
          IF (INDEX(LINE2,'VRHFIN') > 0 .AND. NV < MXT) THEN
            SUB = ADJUSTL(AFTER_EQ(LINE2))
            IP  = INDEX(SUB,':')
            IF (IP > 1) SUB = SUB(1:IP-1)
            READ(SUB,*,IOSTAT=IOS2) CEL
            IF (IOS2 == 0 .AND. LEN_TRIM(CEL) > 0) THEN
              NV = NV+1
              CV(NV) = CEL(1:3)
            ENDIF
          ENDIF
          ! atom types (fallback): 'TITEL  = PAW_PBE B_h 06Sep2000'
          IF (INDEX(LINE2,'TITEL') > 0 .AND. NT2 < MXT) THEN
            SUB = AFTER_EQ(LINE2)
            READ(SUB,*,IOSTAT=IOS2) ADUM, CEL
            IF (IOS2 == 0 .AND. LEN_TRIM(CEL) > 0) THEN
              IP = INDEX(CEL,'_')
              IF (IP > 1) CEL = CEL(1:IP-1)
              NT2 = NT2+1
              CT2(NT2) = CEL(1:3)
            ENDIF
          ENDIF
          ! number of types and atoms per type
          IF (LINE2(1:16) == '   ions per type' .AND. .NOT.LF(1)) THEN
            SUB = AFTER_EQ(LINE2)
            NTF = COUNT_INTS(SUB)
            IF (NTF < 1) ERROR STOP 'Error: could not read "ions per type"'
            READ(SUB,*,IOSTAT=IOS2) (NITF(J), J=1,NTF)
            IF (IOS2 /= 0) ERROR STOP 'Error: could not read "ions per type"'
            LF(1) = .TRUE.
          ENDIF
          ! masses ('  Mass of Ions in am' followed by ' POMASS = ...'); the
          ! line is only stored here, it is parsed once NTF is known
          IF (LINE2(1:14) == '  Mass of Ions' .AND. .NOT.LF(2)) THEN
            READ(IUOUT,'(a)',IOSTAT=IOS2) LINE2
            IF (IOS2 /= 0) EXIT
            SMASS = AFTER_EQ(LINE2)
            LF(2) = .TRUE.
          ENDIF
          IF (LF(1) .AND. LF(2)) EXIT
        ENDDO
        CLOSE(IUOUT)
        IF (.NOT.LF(1)) ERROR STOP 'Error: "ions per type" not found in an OUTCAR'
        IF (.NOT.LF(2)) ERROR STOP 'Error: "Mass of Ions" not found in an OUTCAR'
        READ(SMASS,*,IOSTAT=IOS2) (PMF(J), J=1,NTF)
        IF (IOS2 /= 0) ERROR STOP 'Error: could not read the atomic masses'
        ! the first NTF species read are the ones of this system
        IF (NV >= NTF) THEN
          CTF(1:NTF) = CV(1:NTF)
        ELSEIF (NT2 >= NTF) THEN
          CTF(1:NTF) = CT2(1:NTF)
        ELSE
          ERROR STOP 'Error: could not identify the atom types in an OUTCAR'
        ENDIF
      END SUBROUTINE SCAN_HEADER
!
! ---------------------------------------------------------------------------
! Names read from a file, one name per OUTCAR (line i -> CONF/OUTCARi).
! ---------------------------------------------------------------------------
      SUBROUTINE NAMES_FROM_FILE
        CHARACTER(LEN=200) :: FNIN, LINE2, LADJ
        INTEGER            :: IOS2, N
        WRITE(*,*) 'Enter the names file (empty line = CONF/names.dat):'
        READ(*,'(a)',IOSTAT=IOS2) FNIN
        IF (IOS2 /= 0) FNIN = ' '
        IF (LEN_TRIM(FNIN) == 0) FNIN = 'CONF/names.dat'
        OPEN(IUNAM,FILE=TRIM(ADJUSTL(FNIN)),STATUS='OLD',IOSTAT=IOS2)
        IF (IOS2 /= 0) ERROR STOP 'Error: could not open the names file'
        N = 0
        DO
          READ(IUNAM,'(a)',IOSTAT=IOS2) LINE2
          IF (IOS2 /= 0) EXIT
          LADJ = ADJUSTL(LINE2)
          IF (LEN_TRIM(LADJ) == 0) CYCLE      ! skip blank lines
          IF (LADJ(1:1) == '#') CYCLE         ! skip comments
          IF (N == NFILES) THEN
            WRITE(*,*) 'Warning: the names file has more entries than OUTCAR files, ignoring the rest'
            EXIT
          ENDIF
          N = N+1
          FNAM(N) = LADJ(1:LNM)
        ENDDO
        CLOSE(IUNAM)
        IF (N < NFILES) THEN
          WRITE(*,*) 'Error: only ', N, ' names for ', NFILES, ' OUTCAR files'
          ERROR STOP 'Error: not enough names in the names file'
        ENDIF
      END SUBROUTINE NAMES_FROM_FILE
!
! ---------------------------------------------------------------------------
! Names given by ranges of OUTCAR numbers, e.g. "1 100 graphite".
! ---------------------------------------------------------------------------
      SUBROUTINE NAMES_FROM_RANGES
        CHARACTER(LEN=200)  :: LINE2
        CHARACTER(LEN=LNM)  :: CNAM
        INTEGER             :: I1, I2, IOS2, J
        FNAM = ' '
        WRITE(*,*) 'Enter the first and last OUTCAR number of a range and its name,'
        WRITE(*,*) 'e.g. "1 100 graphite" (the name must not contain blanks).'
        WRITE(*,*) 'Ranges can be entered repeatedly; enter "0 0 end" when done:'
        DO
          READ(*,'(a)',IOSTAT=IOS2) LINE2
          IF (IOS2 /= 0) ERROR STOP 'Error: unexpected end of input'
          IF (LEN_TRIM(LINE2) == 0) CYCLE
          READ(LINE2,*,IOSTAT=IOS2) I1, I2, CNAM
          IF (IOS2 /= 0) THEN
            WRITE(*,*) 'Invalid input: enter "first last name" ("0 0 end" to finish)'
            CYCLE
          ENDIF
          IF (I1 == 0 .AND. I2 == 0) EXIT
          IF (I1 > I2) THEN
            J  = I1
            I1 = I2
            I2 = J
          ENDIF
          IF (I1 < 1 .OR. I2 > NFILES) THEN
            WRITE(*,*) 'Invalid range: the OUTCAR numbers must lie between 1 and ', NFILES
            CYCLE
          ENDIF
          IF (ANY(FNAM(I1:I2) /= ' ')) &
            WRITE(*,*) 'Warning: some files of this range already had a name, overwriting'
          FNAM(I1:I2) = CNAM
          WRITE(*,*) 'OUTCAR ', I1, ' to ', I2, ' -> ', TRIM(CNAM)
        ENDDO
        IF (ANY(FNAM == ' ')) THEN
          WRITE(*,*) 'Warning: ', COUNT(FNAM == ' '), ' OUTCAR files are not covered by a range,'
          WRITE(*,*) '         they get the name "other"'
          DO J=1,NFILES
            IF (FNAM(J) == ' ') FNAM(J) = 'other'
          ENDDO
        ENDIF
      END SUBROUTINE NAMES_FROM_RANGES
!
! ---------------------------------------------------------------------------
! Read one OUTCAR file and write its last NBUF configurations to the scratch
! file. NCF returns the number of configurations found in the file.
! Only the last NBUF configurations are kept in memory (circular buffer).
! ---------------------------------------------------------------------------
      SUBROUTINE PROCESS_FILE(IFIL, NCF)
        INTEGER, INTENT(IN)  :: IFIL
        INTEGER, INTENT(OUT) :: NCF
        REAL(DP), ALLOCATABLE :: BA(:,:,:), BPOS(:,:,:), BFOR(:,:,:), BSIF(:,:)
        REAL(DP), ALLOCATABLE :: BEN(:)
        REAL(DP)              :: A(3,3), POSION(3,FNIONS(IFIL))
        REAL(DP)              :: TIFOR(3,FNIONS(IFIL)), TSIF(6), TOTEN
        INTEGER               :: NIONS, NTF, IOS2, IX, JX, IN, IS, J, JF, NKEEP
        CHARACTER(LEN=200)    :: FILEN, LINE2
        CHARACTER(LEN=32)     :: ADUM
        LOGICAL               :: LFOUND(4), LENE
!
        NIONS = FNIONS(IFIL)
        NTF   = FNTYP(IFIL)
        NCF   = 0
        WRITE(FILEN,'(a,i0)') 'CONF/OUTCAR', IFIL
        OPEN(IUOUT,FILE=FILEN,STATUS='OLD',IOSTAT=IOS2)
        IF (IOS2 /= 0) THEN
          WRITE(*,*) ' Error opening file ', TRIM(FILEN)
          RETURN
        ENDIF
        WRITE(*,*) ' Opening file ', TRIM(FILEN)
        ALLOCATE(BA(3,3,NBUF), BPOS(3,NIONS,NBUF), BFOR(3,NIONS,NBUF), &
                 BSIF(6,NBUF), BEN(NBUF))
        A      = 0._DP
        POSION = 0._DP
        TIFOR  = 0._DP
        TSIF   = 0._DP
        TOTEN  = 0._DP
        LFOUND = .FALSE.
        LENE   = .FALSE.
        DO
          ! loop over the configurations of this OUTCAR
          READ(IUOUT,'(a)',IOSTAT=IOS2) LINE2
          IF (IOS2 /= 0) EXIT
          ! lattice vectors
          IF (LINE2(1:17) == ' VOLUME and BASIS') THEN
            DO IX=1,4
              READ(IUOUT,*,IOSTAT=IOS2) ADUM
            ENDDO
            DO IX=1,3
              READ(IUOUT,*,IOSTAT=IOS2) (A(JX,IX), JX=1,3)
            ENDDO
            IF (IOS2 /= 0) EXIT
            LFOUND(4) = .TRUE.
          ENDIF
          ! atomic positions and forces
          IF (LINE2(1:9) == ' POSITION') THEN
            READ(IUOUT,*,IOSTAT=IOS2) ADUM
            DO IN=1,NIONS
              READ(IUOUT,*,IOSTAT=IOS2) (POSION(IX,IN), IX=1,3), (TIFOR(IX,IN), IX=1,3)
            ENDDO
            IF (IOS2 /= 0) EXIT
            LFOUND(2) = .TRUE.
          ENDIF
          ! stress
          IF (LINE2(1:7) == '  in kB') THEN
            READ(LINE2,*,IOSTAT=IOS2) ADUM, ADUM, (TSIF(J), J=1,6)
            IF (IOS2 == 0) LFOUND(3) = .TRUE.
          ENDIF
          ! energy (last quantity read for each configuration)
          IF (LINE2(1:14) == '  free  energy') THEN
            READ(LINE2,*,IOSTAT=IOS2) ADUM, ADUM, ADUM, ADUM, TOTEN
            IF (IOS2 == 0) THEN
              LFOUND(1) = .TRUE.
              LENE = .TRUE.
            ENDIF
          ENDIF
          ! store the configuration in the circular buffer
          IF (LENE) THEN
            LENE = .FALSE.
            NCF  = NCF+1
            IS = MOD(NCF-1,NBUF)+1
            BA(:,:,IS)   = A
            BPOS(:,:,IS) = POSION
            BFOR(:,:,IS) = TIFOR
            BSIF(:,IS)   = TSIF
            BEN(IS)      = TOTEN
          ENDIF
        ENDDO ! done with all the configurations of this OUTCAR
        CLOSE(IUOUT)
        IF (.NOT.ALL(LFOUND)) THEN
          WRITE(*,*) 'Warning: energy, positions, stress, lattice found in ', TRIM(FILEN), '?'
          WRITE(*,*) LFOUND
        ENDIF
        ! write the last NBUF configurations, in chronological order
        NKEEP = MIN(NCF, NBUF)
        IF (NKEEP == 0) THEN
          WRITE(*,*) 'Warning: no configuration found in ', TRIM(FILEN)
        ELSE
          WRITE(*,*) '   configurations in file = ', NCF, ', stored = ', NKEEP
        ENDIF
        DO JF=NCF-NKEEP+1,NCF
          IS = MOD(JF-1,NBUF)+1
          ICOUNT = ICOUNT+1
          WRITE(IUBOD,FM_CONF) ICOUNT
          WRITE(IUBOD,FM_SNAME) FNAM(IFIL)
          WRITE(IUBOD,FM_NATYP) NTF
          WRITE(IUBOD,FM_NIONS) NIONS
          WRITE(IUBOD,FM_ATNUM)
          DO J=1,NTF
            WRITE(IUBOD,*) FCTYP(J,IFIL), FNITYP(J,IFIL)
          ENDDO
          WRITE(IUBOD,FM_CTIFOR)
          WRITE(IUBOD,*) CTIFOR
          WRITE(IUBOD,FM_LATT)
          DO IX=1,3
            WRITE(IUBOD,*) (BA(JX,IX,IS), JX=1,3)
          ENDDO
          WRITE(IUBOD,FM_POS)
          DO IN=1,NIONS
            WRITE(IUBOD,*) (BPOS(IX,IN,IS), IX=1,3)
          ENDDO
          WRITE(IUBOD,FM_ENE)
          WRITE(IUBOD,*) BEN(IS)
          WRITE(IUENE,*) BEN(IS)
          WRITE(IUBOD,FM_FOR)
          DO IN=1,NIONS
            WRITE(IUBOD,*) (BFOR(IX,IN,IS), IX=1,3)
          ENDDO
          WRITE(IUBOD,FM_STR1)
          WRITE(IUBOD,*) BSIF(1,IS), BSIF(2,IS), BSIF(3,IS)
          WRITE(IUBOD,FM_STR2)
          WRITE(IUBOD,*) BSIF(4,IS), BSIF(5,IS), BSIF(6,IS)
        ENDDO
        DEALLOCATE(BA, BPOS, BFOR, BSIF, BEN)
      END SUBROUTINE PROCESS_FILE
!
      END PROGRAM MAKE_AB
