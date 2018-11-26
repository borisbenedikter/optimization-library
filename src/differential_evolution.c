#include "differential_evolution.h"

double rand01() {
    // Returns a random number between 0 and 1
    double drand = (double)(rand());
    double drand_max = (double)RAND_MAX;
	return  drand / drand_max;
}

double my_max(double a, double b) {
	if (a > b) return a;
	else return b;
}

void my_min(double *a, double* min, int* index_min) {
    // Find minimum element of an array of doubles

    // Number of elements
    int n = sizeof(a) / sizeof(a[0]);

    // Initialization
    *min = a[0];
    *index_min = 0;
    
    // Loop
    for(int i = 1; i < n; i++)
    {
        if (a[i] < *min) {
            *min = a[i];
            *index_min = i;
        }
    }
}

void my_differential_evolution(double (*J_fun)(double *, const int),
    double *lb, double *ub, const int nvar,
    const int n_iter, const int np,
    const double JRelTol, const double JAbsTol,
    double *x_best, double *J_overall)
{

    // Setup population
    double pop[np][nvar];
    double pop_trial[np][nvar];
    for (int i = 0; i < np; i++)
    {
        for (int j = 0; j < nvar; j++)
        {
            pop[i][j] = lb[j] + rand01() * (ub[j] - lb[j]);
            pop_trial[i][j] = pop[i][j];
        }
    }

    // Optimization
    double x[nvar];
	double J_storage[np];
	double J_best[np];

    // Print headers
    printf("%20s %20s %20s \n", "Iter", "J", "Time");

    // Initial time
    clock_t begin = clock();

    int k = 0;
    double JRelErr = JRelTol + 1.;
    double JAbsErr = JAbsTol + 1.;
    while (k < n_iter && JRelErr > JRelTol && JAbsErr > JAbsTol)
	{
        // Evaluate the objective function for all individuals
        for (int i = 0; i < np; i++)
        {
			for (int m = 0; m < nvar; m++)
			{
				x[m] = pop_trial[i][m];
			}
			double J = J_fun(x, nvar);
			J_storage[i] = J;
		}

		// At the first iteration store the best values.
        if (k == 0)
        {
            // Find minimum J and its index
            int index;
            my_min(J_storage, J_overall, &index);

            // Store (initialize) the best values
            for (int i = 0; i < nvar; i++)
            {
                x_best[i] = pop_trial[index][i];
            }

            for (int i = 0; i < np; i++)
            {
                J_best[i] = J_storage[i];
            }
        }

        // Update the overall best J if a new minimum is found.
        double J_storage_min;
        int index_J_st_min;
        my_min(J_storage, &J_storage_min, &index_J_st_min);
        if (J_storage_min < *J_overall)
        {
            JAbsErr = (*J_overall - J_storage_min);
            JRelErr = (*J_overall - J_storage_min) / J_storage_min;
            *J_overall = J_storage_min;
            for (int i = 0; i < nvar; i++)
            {
                x_best[i] = pop_trial[index_J_st_min][i];
            }
        }

        // Update the population if the mutated individual is a better
		// performer than the existing one.
        for (int i = 0; i < np; i++)
        {
            if (J_storage[i] < J_best[i])
            {
                J_best[i] = J_storage[i];
                for (int j = 0; j < nvar; j++)
                {
                    pop[i][j] = pop_trial[i][j];
                }
            }
        }

        // Population update
        for (int i = 0; i < np; i++)
        {
            double F = my_max(0.2, rand01());
            int r0 = (int)(rand01() * np);
            int r1 = (int)(rand01() * np);
            int r2 = (int)(rand01() * np);

            double v[nvar];

            for (int j = 0; j < nvar; j++)
            {
                // Mutation
                v[j] = pop[r0][j] + F * (pop[r1][j] - pop[r2][j]);

                // Bounds check
                if (v[j] > ub[j])
                    v[j] = ub[j];
                else if (v[j] < lb[j])
                    v[j] = lb[j];

                // Population update
                pop_trial[i][j] = v[j];
            }
        }

        // Elapsed time
		clock_t end = clock();
		double elapsed_secs = (double)(end - begin) / CLOCKS_PER_SEC;

        // Update counter
        k++;

		// Print iter, J and time
        printf("%20d %20.10e %20.3e \n", k, *J_overall, elapsed_secs);

        // Termination
        if (k >= n_iter)
        {
            printf("Termination due to MAX_ITER.\n");
        }
        else if (JAbsErr <= JAbsTol)
        {
            printf("Absolute tolerance met.\n");
        }
        else if (JRelErr <= JRelTol)
        {
            printf("Relative tolerance met.\n");
        }
    }
    printf("Final J: %20.10e \n", *J_overall);
    for (int i = 0; i < nvar; i++)
    {
        printf("x[%d] = %20.10e \n", i, x_best[i]);
    }
}

void my_differential_evolution_MPI(double (*J_fun)(double *, const int),
   double *lb, double *ub, const int nvar,
   const int n_iter, const int np,
   const double JRelTol, const double JAbsTol,
   const int rank, const int n_proc,
   double *x_best, double *J_overall)
{
    // Master process
    int master_rank = 0;

    // Setup population
    double pop[np][nvar];
    double pop_trial[np][nvar];

    if (rank == master_rank)
    {
        for (int i = 0; i < np; i++)
        {
            for (int j = 0; j < nvar; j++)
            {
                pop[i][j] = lb[j] + rand01() * (ub[j] - lb[j]);
                pop_trial[i][j] = pop[i][j];
            }
        }
    }

    MPI_Bcast(pop, np * nvar, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(pop_trial, np * nvar, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);

    // Optimization
    double x[nvar];
    double J_storage[np];
    double J_best[np];

    const int np_rank = np / n_proc;
    double J_storage_rank[np_rank];

    // Print headers
    if (rank == master_rank)
    {
        printf("%20s %20s %20s \n", "Iter", "J", "Time");
    }

    // Initial time
    clock_t begin = clock();

        
    // Loop
    int k = 0;
    double JRelErr = JRelTol + 1.;
    double JAbsErr = JAbsTol + 1.;
    while (k < n_iter && JRelErr > JRelTol && JAbsErr > JAbsTol)
    {
        // Evaluate the objective function for all individuals
        for (int i = 0; i < np_rank; i++)
        {
            for (int m = 0; m < nvar; m++)
            {
                int i_rk = rank * np_rank + i;
                x[m] = pop_trial[i_rk][m];
            }
            double J = J_fun(x, nvar);
            J_storage_rank[i] = J;
        }
        
        MPI_Gather(J_storage_rank, np_rank, MPI_DOUBLE,
                   J_storage, np_rank, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);

        
        if (rank == master_rank)
        {

            // At the first iteration store the best values.
            if (k == 0)
            {
                // Find minimum J and its index
                int index;
                my_min(J_storage, J_overall, &index);

                // Store (initialize) the best values
                for (int i = 0; i < nvar; i++)
                {
                    x_best[i] = pop_trial[index][i];
                }

                for (int i = 0; i < np; i++)
                {
                    J_best[i] = J_storage[i];
                }
            }

            // Update the overall best J if a new minimum is found.
            double J_storage_min;
            int index_J_st_min;
            my_min(J_storage, &J_storage_min, &index_J_st_min);
            if (J_storage_min < *J_overall)
            {
                JAbsErr = (*J_overall - J_storage_min);
                JRelErr = (*J_overall - J_storage_min) / J_storage_min;
                *J_overall = J_storage_min;
                for (int i = 0; i < nvar; i++)
                {
                    x_best[i] = pop_trial[index_J_st_min][i];
                }
            }

            // Update the population if the mutated individual is a better
            // performer than the existing one.
            for (int i = 0; i < np; i++)
            {
                if (J_storage[i] < J_best[i])
                {
                    J_best[i] = J_storage[i];
                    for (int j = 0; j < nvar; j++)
                    {
                        pop[i][j] = pop_trial[i][j];
                    }
                }
            }

            // Population update
            for (int i = 0; i < np; i++)
            {
                double F = my_max(0.2, rand01());
                int r0 = (int)(rand01() * np);
                int r1 = (int)(rand01() * np);
                int r2 = (int)(rand01() * np);

                double v[nvar];

                for (int j = 0; j < nvar; j++)
                {
                    // Mutation
                    v[j] = pop[r0][j] + F * (pop[r1][j] - pop[r2][j]);

                    // Bounds check
                    if (v[j] > ub[j])
                        v[j] = ub[j];
                    else if (v[j] < lb[j])
                        v[j] = lb[j];

                    // Population update
                    pop_trial[i][j] = v[j];
                }
            }

            // Elapsed time
            clock_t end = clock();
            double elapsed_secs = (double)(end - begin) / CLOCKS_PER_SEC;


            // Print iter, J and time
            printf("%20d %20.10e %20.3e \n", k + 1, *J_overall, elapsed_secs);

            // Termination
            if (k >= n_iter)
            {
                printf("Termination due to MAX_ITER.\n");
            }
            else if (JAbsErr <= JAbsTol)
            {
                printf("Absolute tolerance met.\n");
            }
            else if (JRelErr <= JRelTol)
            {
                printf("Relative tolerance met.\n");
            }
        } // if (rank == master_rank)
        

        MPI_Bcast(&JAbsErr, 1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
        
        MPI_Bcast(&JRelErr, 1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);

        MPI_Bcast(pop, np * nvar, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);

        MPI_Bcast(pop_trial, np * nvar, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);

        // Update counter
        k++;
    }
    if (rank == master_rank)
    {
        printf("Final J: %20.10e \n", *J_overall);
        for (int i = 0; i < nvar; i++)
        {
            printf("x[%d] = %20.10e \n", i, x_best[i]);
        }
    }
    

}