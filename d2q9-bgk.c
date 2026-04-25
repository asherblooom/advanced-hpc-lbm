/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>

#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
	int nx;			  /* no. of cells in x-direction */
	int ny;			  /* no. of cells in y-direction */
	int maxIters;	  /* no. of iterations */
	int reynolds_dim; /* dimension for Reynolds number */
	float density;	  /* density per link */
	float accel;	  /* density redistribution */
	float omega;	  /* relaxation parameter */
} t_param;

// typedef struct
// {
// 	float speeds[NSPEEDS];
// } t_speed;

/* struct to hold the 'speed' values */
// new SoA format - each speed is a 1-D array with 1 speed for every cell
// that is why it is a pointer
typedef struct {
	float* s0;
	float* s1;
	float* s2;
	float* s3;
	float* s4;
	float* s5;
	float* s6;
	float* s7;
	float* s8;
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
			   t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
			   int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow() & timestep_merged()
*/
float timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
float timestep_merged(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
			 int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[]) {
	char* paramfile = NULL;															   /* name of the input parameter file */
	char* obstaclefile = NULL;														   /* name of a the input obstacle file */
	t_param params;																	   /* struct to hold parameter values */
	t_speed cells;																	   /* grid containing fluid densities */
	t_speed tmp_cells;																   /* scratch space */
	int* obstacles = NULL;															   /* grid indicating which cells are blocked */
	float* av_vels = NULL;															   /* a record of the av. velocity computed for each timestep */
	struct timeval timstr;															   /* structure to hold elapsed time */
	double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

	/* parse the command line */
	if (argc != 3) {
		usage(argv[0]);
	} else {
		paramfile = argv[1];
		obstaclefile = argv[2];
	}

	/* Total/init time starts here: initialise our data structures and load values from file */
	gettimeofday(&timstr, NULL);
	tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
	init_tic = tot_tic;
	initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

	/* Init time stops here, compute time starts*/
	gettimeofday(&timstr, NULL);
	init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
	comp_tic = init_toc;

	for (int tt = 0; tt < params.maxIters; tt++) {
		av_vels[tt] = timestep(params, &cells, &tmp_cells, obstacles);
		// Swap cells and tmp_cells, so cells has the new values for the next timestep!
		t_speed swap = cells;
		cells = tmp_cells;
		tmp_cells = swap;
#ifdef DEBUG
		printf("==timestep: %d==\n", tt);
		printf("av velocity: %.12E\n", av_vels[tt]);
		printf("tot density: %.12E\n", total_density(params, cells));
#endif
	}

	/* Compute time stops here, collate time starts*/
	gettimeofday(&timstr, NULL);
	comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
	col_tic = comp_toc;

	// Collate data from ranks here

	/* Total/collate time stops here.*/
	gettimeofday(&timstr, NULL);
	col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
	tot_toc = col_toc;

	/* write final values and free memory */
	printf("==done==\n");
	printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, &cells, obstacles));
	printf("Elapsed Init time:\t\t\t%.6lf (s)\n", init_toc - init_tic);
	printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
	printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc - col_tic);
	printf("Elapsed Total time:\t\t\t%.6lf (s)\n", tot_toc - tot_tic);
	write_values(params, &cells, obstacles, av_vels);
	finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

	return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles) {
	accelerate_flow(params, cells, obstacles);
	return timestep_merged(params, cells, tmp_cells, obstacles);
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles) {
	float* restrict c1 = cells->s1;
	float* restrict c3 = cells->s3;
	float* restrict c5 = cells->s5;
	float* restrict c6 = cells->s6;
	float* restrict c7 = cells->s7;
	float* restrict c8 = cells->s8;

	/* compute weighting factors */
	float w1 = params.density * params.accel / 9.f;
	float w2 = params.density * params.accel / 36.f;

	/* modify the 2nd row of the grid */
	int jj = params.ny - 2;
	int jj_nx = jj * params.nx;

	__builtin_assume(params.nx % 16 == 0);
#pragma omp simd aligned(c1, c3, c5, c6, c7, c8 : 64)
	for (int ii = 0; ii < params.nx; ii++) {
		int idx = ii + jj_nx;
		/* if the cell is not occupied and
		** we don't send a negative density */
		if (!obstacles[idx] &&
			(c3[idx] - w1) > 0.f &&
			(c6[idx] - w2) > 0.f &&
			(c7[idx] - w2) > 0.f) {
			/* increase 'east-side' densities */
			c1[idx] += w1;
			c5[idx] += w2;
			c8[idx] += w2;
			/* decrease 'west-side' densities */
			c3[idx] -= w1;
			c6[idx] -= w2;
			c7[idx] -= w2;
		}
	}

	return EXIT_SUCCESS;
}

float timestep_merged(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles) {
	const float* restrict c0 = cells->s0;
	const float* restrict c1 = cells->s1;
	const float* restrict c2 = cells->s2;
	const float* restrict c3 = cells->s3;
	const float* restrict c4 = cells->s4;
	const float* restrict c5 = cells->s5;
	const float* restrict c6 = cells->s6;
	const float* restrict c7 = cells->s7;
	const float* restrict c8 = cells->s8;

	float* restrict t0 = tmp_cells->s0;
	float* restrict t1 = tmp_cells->s1;
	float* restrict t2 = tmp_cells->s2;
	float* restrict t3 = tmp_cells->s3;
	float* restrict t4 = tmp_cells->s4;
	float* restrict t5 = tmp_cells->s5;
	float* restrict t6 = tmp_cells->s6;
	float* restrict t7 = tmp_cells->s7;
	float* restrict t8 = tmp_cells->s8;

	// av_velocity variables
	int tot_cells = 0; /* no. of cells used in calculation */
	float tot_u = 0;   /* accumulated magnitudes of velocity for each cell */

	/* loop over _all_ cells */
	for (int jj = 0; jj < params.ny; jj++) {
		// these dont rely on ii, so calculate them here
		int y_n = (jj + 1) % params.ny;
		int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
		int jj_nx = jj * params.nx;
		int yn_nx = y_n * params.nx;
		int ys_nx = y_s * params.nx;

		__builtin_assume(params.nx % 16 == 0);
#pragma omp simd aligned(c0, c1, c2, c3, c4, c5, c6, c7, c8, t0, t1, t2, t3, t4, t5, t6, t7, t8 : 64) reduction(+ : tot_u, tot_cells)
		for (int ii = 0; ii < params.nx; ii++) {
			int x_e = (ii + 1) % params.nx;
			int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
			int idx = ii + jj_nx;
			const float w0 = 4.f / 9.f;	 /* weighting factor */
			const float w1 = 1.f / 9.f;	 /* weighting factor */
			const float w2 = 1.f / 36.f; /* weighting factor */
			// c_sq is 1/3, so we can hardcode the inverted fraction
			const float c_sq_inv_half = 1.5f;  // 1 / (2 * c_sq)
			// pre-calculate as much as possible
			const float one_minus_omega = 1.0f - params.omega;
			const float omega_w0 = params.omega * w0;
			const float omega_w1 = params.omega * w1;
			const float omega_w2 = params.omega * w2;

			/* propagate densities from neighbouring cells, following
	** appropriate directions of travel and writing into
	** speeds variables */
			float speeds0 = c0[idx];		 /* central cell, no movement */
			float speeds1 = c1[x_w + jj_nx]; /* east */
			float speeds2 = c2[ii + ys_nx];	 /* north */
			float speeds3 = c3[x_e + jj_nx]; /* west */
			float speeds4 = c4[ii + yn_nx];	 /* south */
			float speeds5 = c5[x_w + ys_nx]; /* north-east */
			float speeds6 = c6[x_e + ys_nx]; /* north-west */
			float speeds7 = c7[x_e + yn_nx]; /* south-west */
			float speeds8 = c8[x_w + yn_nx]; /* south-east */

			/* compute local density total */
			float local_density = speeds0 + speeds1 + speeds2 + speeds3 + speeds4 + speeds5 + speeds6 + speeds7 + speeds8;
			float inv_density = 1.0f / local_density;  // avoid division
			/* compute x velocity component */
			float u_x = (speeds1 + speeds5 + speeds8 - (speeds3 + speeds6 + speeds7)) * inv_density;
			/* compute y velocity component */
			float u_y = (speeds2 + speeds5 + speeds6 - (speeds4 + speeds7 + speeds8)) * inv_density;
			/* velocity squared */
			float u_sq = u_x * u_x + u_y * u_y;
			// Pre-calculate common terms
			float term_sq = u_sq * c_sq_inv_half;
			float one_minus_term_sq = 1.0f - term_sq;
			float w0_den_omega = omega_w0 * local_density;
			float w1_den_omega = omega_w1 * local_density;
			float w2_den_omega = omega_w2 * local_density;

			float cu1 = 3.0f * u_x;
			float cu2 = 3.0f * u_y;
			// Exploit symmetry
			float cu3 = -cu1;
			float cu4 = -cu2;
			float cu5 = cu1 + cu2;
			float cu6 = -cu1 + cu2;
			float cu7 = -cu5;
			float cu8 = -cu6;

			int is_solid = obstacles[idx];
			// Use ternary operators to avoid big branch in loop body
			t0[idx] = is_solid ? speeds0 : (speeds0 * one_minus_omega + w0_den_omega * one_minus_term_sq);
			t1[idx] = is_solid ? speeds3 : (speeds1 * one_minus_omega + w1_den_omega * (one_minus_term_sq + cu1 * (1.0f + 0.5f * cu1)));
			t2[idx] = is_solid ? speeds4 : (speeds2 * one_minus_omega + w1_den_omega * (one_minus_term_sq + cu2 * (1.0f + 0.5f * cu2)));
			t3[idx] = is_solid ? speeds1 : (speeds3 * one_minus_omega + w1_den_omega * (one_minus_term_sq + cu3 * (1.0f + 0.5f * cu3)));
			t4[idx] = is_solid ? speeds2 : (speeds4 * one_minus_omega + w1_den_omega * (one_minus_term_sq + cu4 * (1.0f + 0.5f * cu4)));
			t5[idx] = is_solid ? speeds7 : (speeds5 * one_minus_omega + w2_den_omega * (one_minus_term_sq + cu5 * (1.0f + 0.5f * cu5)));
			t6[idx] = is_solid ? speeds8 : (speeds6 * one_minus_omega + w2_den_omega * (one_minus_term_sq + cu6 * (1.0f + 0.5f * cu6)));
			t7[idx] = is_solid ? speeds5 : (speeds7 * one_minus_omega + w2_den_omega * (one_minus_term_sq + cu7 * (1.0f + 0.5f * cu7)));
			t8[idx] = is_solid ? speeds6 : (speeds8 * one_minus_omega + w2_den_omega * (one_minus_term_sq + cu8 * (1.0f + 0.5f * cu8)));

			tot_u += is_solid ? 0.0f : sqrtf(u_sq);
			tot_cells += is_solid ? 0 : 1;
		}
	}

	return tot_u / (float)tot_cells;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles) {
	int tot_cells = 0; /* no. of cells used in calculation */
	float tot_u;	   /* accumulated magnitudes of velocity for each cell */

	/* initialise */
	tot_u = 0.f;

	/* loop over all non-blocked cells */
	for (int jj = 0; jj < params.ny; jj++) {
		int jj_nx = jj * params.nx;
		for (int ii = 0; ii < params.nx; ii++) {
			int idx = ii + jj_nx;
			/* ignore occupied cells */
			if (!obstacles[idx]) {
				/* local density total */
				float local_density = cells->s0[idx] + cells->s1[idx] + cells->s2[idx] + cells->s3[idx] + cells->s4[idx] + cells->s5[idx] + cells->s6[idx] + cells->s7[idx] + cells->s8[idx];

				/* x-component of velocity */
				float u_x = (cells->s1[idx] +
							 cells->s5[idx] +
							 cells->s8[idx] -
							 (cells->s3[idx] +
							  cells->s6[idx] +
							  cells->s7[idx])) /
							local_density;
				/* compute y velocity component */
				float u_y = (cells->s2[idx] +
							 cells->s5[idx] +
							 cells->s6[idx] -
							 (cells->s4[idx] +
							  cells->s7[idx] +
							  cells->s8[idx])) /
							local_density;
				/* accumulate the norm of x- and y- velocity components */
				tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
				/* increase counter of inspected cells */
				++tot_cells;
			}
		}
	}

	return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
			   t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
			   int** obstacles_ptr, float** av_vels_ptr) {
	char message[1024]; /* message buffer */
	FILE* fp;			/* file pointer */
	int xx, yy;			/* generic array indices */
	int blocked;		/* indicates whether a cell is blocked by an obstacle */
	int retval;			/* to hold return value for checking */

	/* open the parameter file */
	fp = fopen(paramfile, "r");

	if (fp == NULL) {
		sprintf(message, "could not open input parameter file: %s", paramfile);
		die(message, __LINE__, __FILE__);
	}

	/* read in the parameter values */
	retval = fscanf(fp, "%d\n", &(params->nx));

	if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

	retval = fscanf(fp, "%d\n", &(params->ny));

	if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

	retval = fscanf(fp, "%d\n", &(params->maxIters));

	if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

	retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

	if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

	retval = fscanf(fp, "%f\n", &(params->density));

	if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

	retval = fscanf(fp, "%f\n", &(params->accel));

	if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

	retval = fscanf(fp, "%f\n", &(params->omega));

	if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

	/* and close up the file */
	fclose(fp);

	/*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

	// staggered allocation to prevent cache thrashing
	const int padding_floats = 16;
	int single_grid_floats = params->ny * params->nx;
	int stride_floats = single_grid_floats + padding_floats;
	// Allocate one contiguous block for all 9 speeds
	int total_bytes = stride_floats * 9 * sizeof(float);

	// Allocate the main cells block and assign the base to s0
	cells_ptr->s0 = (float*)aligned_alloc(64, total_bytes);
	// Manually offset the pointers for s1 through s8 by our staggered stride
	cells_ptr->s1 = cells_ptr->s0 + (1 * stride_floats);
	cells_ptr->s2 = cells_ptr->s0 + (2 * stride_floats);
	cells_ptr->s3 = cells_ptr->s0 + (3 * stride_floats);
	cells_ptr->s4 = cells_ptr->s0 + (4 * stride_floats);
	cells_ptr->s5 = cells_ptr->s0 + (5 * stride_floats);
	cells_ptr->s6 = cells_ptr->s0 + (6 * stride_floats);
	cells_ptr->s7 = cells_ptr->s0 + (7 * stride_floats);
	cells_ptr->s8 = cells_ptr->s0 + (8 * stride_floats);

	tmp_cells_ptr->s0 = (float*)aligned_alloc(64, total_bytes);
	tmp_cells_ptr->s1 = tmp_cells_ptr->s0 + (1 * stride_floats);
	tmp_cells_ptr->s2 = tmp_cells_ptr->s0 + (2 * stride_floats);
	tmp_cells_ptr->s3 = tmp_cells_ptr->s0 + (3 * stride_floats);
	tmp_cells_ptr->s4 = tmp_cells_ptr->s0 + (4 * stride_floats);
	tmp_cells_ptr->s5 = tmp_cells_ptr->s0 + (5 * stride_floats);
	tmp_cells_ptr->s6 = tmp_cells_ptr->s0 + (6 * stride_floats);
	tmp_cells_ptr->s7 = tmp_cells_ptr->s0 + (7 * stride_floats);
	tmp_cells_ptr->s8 = tmp_cells_ptr->s0 + (8 * stride_floats);

	/* the map of obstacles */
	*obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));
	if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

	/* initialise densities */
	float w0 = params->density * 4.f / 9.f;
	float w1 = params->density / 9.f;
	float w2 = params->density / 36.f;

	for (int jj = 0; jj < params->ny; jj++) {
		int jj_nx = jj * params->nx;
		for (int ii = 0; ii < params->nx; ii++) {
			int idx = ii + jj_nx;
			/* centre */
			cells_ptr->s0[idx] = w0;
			/* axis directions */
			cells_ptr->s1[idx] = w1;
			cells_ptr->s2[idx] = w1;
			cells_ptr->s3[idx] = w1;
			cells_ptr->s4[idx] = w1;
			/* diagonals */
			cells_ptr->s5[idx] = w2;
			cells_ptr->s6[idx] = w2;
			cells_ptr->s7[idx] = w2;
			cells_ptr->s8[idx] = w2;
		}
	}

	/* first set all cells in obstacle array to zero */
	for (int jj = 0; jj < params->ny; jj++) {
		for (int ii = 0; ii < params->nx; ii++) {
			(*obstacles_ptr)[ii + jj * params->nx] = 0;
		}
	}

	/* open the obstacle data file */
	fp = fopen(obstaclefile, "r");

	if (fp == NULL) {
		sprintf(message, "could not open input obstacles file: %s", obstaclefile);
		die(message, __LINE__, __FILE__);
	}

	/* read-in the blocked cells list */
	while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
		/* some checks */
		if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

		if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

		if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

		if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

		/* assign to array */
		(*obstacles_ptr)[xx + yy * params->nx] = blocked;
	}

	/* and close the file */
	fclose(fp);

	/*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
	*av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

	return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
			 int** obstacles_ptr, float** av_vels_ptr) {
	/*
  ** free up allocated memory
  */
	// allocated as one big block so just need to free s0
	free(cells_ptr->s0);
	free(tmp_cells_ptr->s0);

	free(*obstacles_ptr);
	*obstacles_ptr = NULL;

	free(*av_vels_ptr);
	*av_vels_ptr = NULL;

	return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, t_speed* cells, int* obstacles) {
	const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

	return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells) {
	float total = 0.f; /* accumulator */

	for (int jj = 0; jj < params.ny; jj++) {
		for (int ii = 0; ii < params.nx; ii++) {
			int idx = ii + jj * params.nx;
			total = cells->s0[idx] + cells->s1[idx] + cells->s2[idx] + cells->s3[idx] + cells->s4[idx] + cells->s5[idx] + cells->s6[idx] + cells->s7[idx] + cells->s8[idx];
		}
	}

	return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels) {
	FILE* fp;					  /* file pointer */
	const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
	float local_density;		  /* per grid cell sum of densities */
	float pressure;				  /* fluid pressure in grid cell */
	float u_x;					  /* x-component of velocity in grid cell */
	float u_y;					  /* y-component of velocity in grid cell */
	float u;					  /* norm--root of summed squares--of u_x and u_y */

	fp = fopen(FINALSTATEFILE, "w");

	if (fp == NULL) {
		die("could not open file output file", __LINE__, __FILE__);
	}

	for (int jj = 0; jj < params.ny; jj++) {
		for (int ii = 0; ii < params.nx; ii++) {
			/* an occupied cell */
			if (obstacles[ii + jj * params.nx]) {
				u_x = u_y = u = 0.f;
				pressure = params.density * c_sq;
			}
			/* no obstacle */
			else {
				int idx = ii + jj * params.nx;
				local_density = cells->s0[idx] + cells->s1[idx] + cells->s2[idx] + cells->s3[idx] + cells->s4[idx] + cells->s5[idx] + cells->s6[idx] + cells->s7[idx] + cells->s8[idx];

				/* compute x velocity component */
				u_x = (cells->s1[idx] +
					   cells->s5[idx] +
					   cells->s8[idx] -
					   (cells->s3[idx] +
						cells->s6[idx] +
						cells->s7[idx])) /
					  local_density;
				/* compute y velocity component */
				u_y = (cells->s2[idx] +
					   cells->s5[idx] +
					   cells->s6[idx] -
					   (cells->s4[idx] +
						cells->s7[idx] +
						cells->s8[idx])) /
					  local_density;
				/* compute norm of velocity */
				u = sqrtf((u_x * u_x) + (u_y * u_y));
				/* compute pressure */
				pressure = local_density * c_sq;
			}

			/* write to file */
			fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
		}
	}

	fclose(fp);

	fp = fopen(AVVELSFILE, "w");

	if (fp == NULL) {
		die("could not open file output file", __LINE__, __FILE__);
	}

	for (int ii = 0; ii < params.maxIters; ii++) {
		fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
	}

	fclose(fp);

	return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file) {
	fprintf(stderr, "Error at line %d of file %s:\n", line, file);
	fprintf(stderr, "%s\n", message);
	fflush(stderr);
	exit(EXIT_FAILURE);
}

void usage(const char* exe) {
	fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
	exit(EXIT_FAILURE);
}
