static char help[] = "Tests that the assembled views of a MATIS follow a change made through MatISGetLocalMat().\n\n";

#include <petscmat.h>

/* Prints the diagonal of A, one line per process */
static PetscErrorCode DiagonalView(Mat A, const char *label)
{
  Vec                d;
  PetscInt           n;
  const PetscScalar *vals;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(A, NULL, &d));
  PetscCall(MatGetDiagonal(A, d));
  PetscCall(VecGetLocalSize(d, &n));
  PetscCall(VecGetArrayRead(d, &vals));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%-32s", label));
  for (PetscInt i = 0; i < n; i++) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, " %g", (double)PetscRealPart(vals[i])));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));
  PetscCall(VecRestoreArrayRead(d, &vals));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  PetscCall(VecDestroy(&d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat                    A, Aij, dA, lA;
  ISLocalToGlobalMapping l2g;
  PetscInt              *gidx;
  PetscInt               row = 1, nl = 4, N;
  PetscMPIInt            rank, size;
  const PetscScalar      elem[] = {2.0, -1.0, -1.0, 2.0};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  /* subdomain r holds the nl nodes r*(nl-1) ... r*(nl-1)+nl-1, so consecutive subdomains share a node */
  N = size * (nl - 1) + 1;
  PetscCall(PetscMalloc1(nl, &gidx));
  for (PetscInt i = 0; i < nl; i++) gidx[i] = rank * (nl - 1) + i;
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, nl, gidx, PETSC_OWN_POINTER, &l2g));
  PetscCall(MatCreateIS(PETSC_COMM_WORLD, 1, PETSC_DECIDE, PETSC_DECIDE, N, N, l2g, l2g, &A));
  PetscCall(ISLocalToGlobalMappingDestroy(&l2g));
  PetscCall(MatISSetPreallocation(A, 3, NULL, 0, NULL));
  for (PetscInt e = 0; e < nl - 1; e++) {
    const PetscInt erows[] = {e, e + 1};

    PetscCall(MatSetValuesLocal(A, 2, erows, 2, erows, elem, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* assembled views built before the change, the way a solver caches them */
  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &Aij));
  PetscCall(MatGetDiagonalBlock(A, &dA));

  /* impose a unit diagonal on one row of every local matrix, keeping the nonzero pattern */
  PetscCall(MatISGetLocalMat(A, &lA));
  PetscCall(MatSetOption(lA, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  PetscCall(MatZeroRows(lA, 1, &row, 1.0, NULL, NULL));
  PetscCall(MatISRestoreLocalMat(A, &lA));

  PetscCall(MatISGetLocalMat(A, &lA));
  PetscCall(DiagonalView(lA, "local (Neumann):"));
  PetscCall(MatISRestoreLocalMat(A, &lA));
  PetscCall(MatConvert(A, MATAIJ, MAT_REUSE_MATRIX, &Aij));
  PetscCall(DiagonalView(Aij, "assembled (MAT_REUSE_MATRIX):"));
  PetscCall(MatDestroy(&Aij));
  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &Aij));
  PetscCall(DiagonalView(Aij, "assembled (MAT_INITIAL_MATRIX):"));
  PetscCall(MatGetDiagonalBlock(A, &dA));
  PetscCall(DiagonalView(dA, "diagonal block:"));

  PetscCall(MatDestroy(&Aij));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      diff_args: -j

TEST*/
