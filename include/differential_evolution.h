#pragma once

#define PARALLEL 1

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#if PARALLEL == 1
#include <mpi.h>
#endif

double rand01();
double my_max(double a, double b);
void my_min(double *a, double* min, int* index_min);
void my_differential_evolution(double(*J_fun)(double*, const int), 
	double* lb, double* ub, const int nvar,
	const int n_iter, const int np,
    const double JRelTol, const double JAbsTol,
	double* x_best, double* J_overall);
void my_differential_evolution_MPI(double (*J_fun)(double *, const int),
    double *lb, double *ub, const int nvar,
    const int n_iter, const int np,
    const double JRelTol, const double JAbsTol,
    const int rank, const int n_proc,
    double *x_best, double *J_overall);
