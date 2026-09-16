static char help[] = "Tests MATIS views cached after the local matrix is modified through MatISGetLocalMat().\n\n";

#include <petscmat.h>

/*
  One-dimensional domain decomposition: rank r owns the nl nodes
  r*(nl-1) ... r*(nl-1)+nl-1, so consecutive subdomains share one node.
  The local (Neumann) matrix is the elementwise assembly of

    [  2 -1 ]
    [ -1  2 ]

  hence shared nodes receive a contribution from two subdomains.
*/
static PetscErrorCode SubdomainMatISCreate(MPI_Comm comm, PetscInt nl, Mat *A)
{
  ISLocalToGlobalMapping l2g;
  PetscInt              *gidx;
  PetscInt               N;
  PetscMPIInt            rank, size;
  const PetscScalar      elem[] = {2.0, -1.0, -1.0, 2.0};

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  N = size * (nl - 1) + 1;
  PetscCall(PetscMalloc1(nl, &gidx));
  for (PetscInt i = 0; i < nl; i++) gidx[i] = rank * (nl - 1) + i;
  PetscCall(ISLocalToGlobalMappingCreate(comm, 1, nl, gidx, PETSC_OWN_POINTER, &l2g));
  PetscCall(MatCreateIS(comm, 1, PETSC_DECIDE, PETSC_DECIDE, N, N, l2g, l2g, A));
  PetscCall(ISLocalToGlobalMappingDestroy(&l2g));
  PetscCall(MatISSetPreallocation(*A, 3, NULL, 0, NULL));
  for (PetscInt e = 0; e < nl - 1; e++) {
    const PetscInt lrows[] = {e, e + 1};

    PetscCall(MatSetValuesLocal(*A, 2, lrows, 2, lrows, elem, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Wraps A as the (1,1) block of a 2x2 MATNEST whose (0,0) block is a plain MATAIJ */
static PetscErrorCode NestCreate(MPI_Comm comm, Mat A, Mat *nest)
{
  Mat         mats[4] = {NULL, NULL, NULL, A};
  IS          isr[2], isc[2];
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(MatCreateAIJ(comm, 1, 1, PETSC_DECIDE, PETSC_DECIDE, 1, NULL, 0, NULL, &mats[0]));
  PetscCall(MatSetValue(mats[0], rank, rank, 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(mats[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mats[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateNest(comm, 2, NULL, 2, NULL, mats, nest));
  PetscCall(MatDestroy(&mats[0]));
  PetscCall(MatAssemblyBegin(*nest, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*nest, MAT_FINAL_ASSEMBLY));
  /* exercise the lazy MATNEST index set setup, as a solver would */
  PetscCall(MatNestGetISs(*nest, isr, isc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Imposes Dirichlet rows the way an application does when the conditions must be
  visible in the local Neumann matrix handed to PCHPDDMSetAuxiliaryMat(): the rows
  are zeroed in the local matrix directly instead of calling MatZeroRows() on the
  MATIS. With keeppattern false, MatZeroRows() shrinks the local nonzero pattern.
*/
static PetscErrorCode ZeroRowsLocal(Mat A, PetscInt n, const PetscInt grows[], PetscScalar diag, PetscBool keeppattern)
{
  ISLocalToGlobalMapping rl2g;
  Mat                    lA;
  PetscInt              *lrows;
  PetscInt               nloc;

  PetscFunctionBeginUser;
  PetscCall(MatISGetLocalMat(A, &lA));
  PetscCall(MatGetLocalToGlobalMapping(A, &rl2g, NULL));
  PetscCall(PetscMalloc1(n, &lrows));
  PetscCall(ISGlobalToLocalMappingApply(rl2g, IS_GTOLM_DROP, n, grows, &nloc, lrows));
  PetscCall(MatSetOption(lA, MAT_KEEP_NONZERO_PATTERN, keeppattern));
  PetscCall(MatZeroRows(lA, nloc, lrows, diag, NULL, NULL));
  PetscCall(PetscFree(lrows));
  PetscCall(MatISRestoreLocalMat(A, &lA));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Prints the diagonal of A, one line per rank, and returns it */
static PetscErrorCode DiagonalGetAndView(Mat A, const char *label, Vec *d)
{
  PetscInt           n;
  const PetscScalar *vals;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(A, NULL, d));
  PetscCall(MatGetDiagonal(A, *d));
  PetscCall(VecGetLocalSize(*d, &n));
  PetscCall(VecGetArrayRead(*d, &vals));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%-44s", label));
  for (PetscInt i = 0; i < n; i++) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, " %g", (double)PetscRealPart(vals[i])));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));
  PetscCall(VecRestoreArrayRead(*d, &vals));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Reports whether two diagonals holding the same rows on each rank agree */
static PetscErrorCode DiagonalCompare(Vec d, Vec ref, const char *label)
{
  const PetscScalar *dd, *rr;
  PetscReal          err = 0.0;
  PetscInt           n, nref;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(d, &n));
  PetscCall(VecGetLocalSize(ref, &nref));
  PetscCheck(n == nref, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible diagonals %" PetscInt_FMT " != %" PetscInt_FMT, n, nref);
  PetscCall(VecGetArrayRead(d, &dd));
  PetscCall(VecGetArrayRead(ref, &rr));
  for (PetscInt i = 0; i < n; i++) err = PetscMax(err, PetscAbsScalar(dd[i] - rr[i]));
  PetscCall(VecRestoreArrayRead(d, &dd));
  PetscCall(VecRestoreArrayRead(ref, &rr));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &err, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
  /* the verdict is reported as text: petscdiff masks floating point numbers by default */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-44s %s\n", label, err < PETSC_SMALL ? "matches" : "DIFFERS"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat                A, dA, lA, ref, nest = NULL, Aij = NULL;
  Vec                d, dref, dlocref;
  PetscScalar       *locvals;
  const PetscScalar *refvals;
  PetscInt           grows[2], ngrows, rst, ren, nl = 4;
  PetscMPIInt        size;
  PetscBool          use_nest = PETSC_FALSE, keeppattern = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nl", &nl, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_nest", &use_nest, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-keep_pattern", &keeppattern, NULL));
  PetscCheck(nl >= 3, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-nl must be at least 3");

  PetscCall(SubdomainMatISCreate(PETSC_COMM_WORLD, nl, &A));
  if (use_nest) PetscCall(NestCreate(PETSC_COMM_WORLD, A, &nest));

  /* Dirichlet rows: an interior node of the first subdomain and, in parallel, an interface node */
  ngrows   = 1;
  grows[0] = 1;
  if (size > 1) grows[ngrows++] = nl - 1;

  /* assembled views built before the rows are zeroed, the way a solver or an exporter caches them */
  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &Aij));
  PetscCall(MatGetDiagonalBlock(A, &dA));
  PetscCall(DiagonalGetAndView(Aij, "before, assembled:", &d));
  PetscCall(VecDestroy(&d));

  PetscCall(ZeroRowsLocal(A, ngrows, grows, 1.0, keeppattern));

  /* the local matrices do carry the requested diagonal value */
  PetscCall(MatISGetLocalMat(A, &lA));
  PetscCall(DiagonalGetAndView(lA, "after, local (Neumann):", &d));
  PetscCall(VecDestroy(&d));
  PetscCall(MatISRestoreLocalMat(A, &lA));

  /* a matrix assembled from scratch is the reference: every other view must match it */
  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &ref));
  PetscCall(DiagonalGetAndView(ref, "after, assembled (MAT_INITIAL_MATRIX):", &dref));
  PetscCall(MatGetOwnershipRange(ref, &rst, &ren));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, ren - rst, &dlocref));
  PetscCall(VecGetArrayRead(dref, &refvals));
  PetscCall(VecGetArray(dlocref, &locvals));
  PetscCall(PetscArraycpy(locvals, refvals, ren - rst));
  PetscCall(VecRestoreArray(dlocref, &locvals));
  PetscCall(VecRestoreArrayRead(dref, &refvals));

  PetscCall(MatConvert(A, MATAIJ, MAT_REUSE_MATRIX, &Aij));
  PetscCall(DiagonalGetAndView(Aij, "after, assembled (MAT_REUSE_MATRIX):", &d));
  PetscCall(DiagonalCompare(d, dref, "MAT_REUSE_MATRIX vs MAT_INITIAL_MATRIX:"));
  PetscCall(VecDestroy(&d));

  PetscCall(MatGetDiagonalBlock(A, &dA));
  PetscCall(DiagonalGetAndView(dA, "after, diagonal block:", &d));
  PetscCall(DiagonalCompare(d, dlocref, "MatGetDiagonalBlock() vs MAT_INITIAL_MATRIX:"));
  PetscCall(VecDestroy(&d));

  PetscCall(VecDestroy(&dlocref));
  PetscCall(VecDestroy(&dref));
  PetscCall(MatDestroy(&ref));
  PetscCall(MatDestroy(&Aij));
  PetscCall(MatDestroy(&nest));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      nsize: 2
      output_file: output/ex320_1.out

      test:
         suffix: 1
         args: -keep_pattern 1

      test:
         suffix: 1_nest
         args: -keep_pattern 1 -use_nest 1

      test:
         suffix: 2
         args: -keep_pattern 0

      test:
         suffix: 2_nest
         args: -keep_pattern 0 -use_nest 1

TEST*/
