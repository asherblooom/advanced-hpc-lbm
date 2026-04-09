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
#include <mpi.h>
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
	int nx; /* no. of cells in x-direction */
	int ny; /* no. of cells in y-direction */
	int local_nx;
	int local_ny;
	int startX;
	int startY;
	int maxIters;	  /* no. of iterations */
	int reynolds_dim; /* dimension for Reynolds number */
	float density;	  /* density per link */
	float accel;	  /* density redistribution */
	float omega;	  /* relaxation parameter */
} t_param;

typedef struct
{
	int size;
	int rank;
	int w_rank;
	int e_rank;
	int n_rank;
	int s_rank;
} t_ranks;

typedef struct
{
	int x_buf_size;
	int y_buf_size;
	float* send_west;
	float* recv_west;
	float* send_east;
	float* recv_east;
	float* recv_south;
	float* send_south;
	float* send_north;
	float* recv_north;
} t_buffers;

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
			   t_param* params, t_ranks* ranks, t_buffers* buffers, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
			   int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow() & timestep_merged()
*/
float timestep(const t_param params, const t_ranks ranks, t_buffers* buffers, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
float timestep_merged(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
void exchange_halos(const t_param params, const t_ranks ranks, t_buffers* buffers, t_speed* cells);
int write_values(const t_param params, const t_ranks ranks, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_buffers* buffers, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
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
	MPI_Init(&argc, &argv);

	char* paramfile = NULL;	   /* name of the input parameter file */
	char* obstaclefile = NULL; /* name of a the input obstacle file */
	t_param params;			   /* struct to hold parameter values */
	t_ranks ranks;
	t_buffers buffers;
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
	initialise(paramfile, obstaclefile, &params, &ranks, &buffers, &cells, &tmp_cells, &obstacles, &av_vels);

	/* Init time stops here, compute time starts*/
	gettimeofday(&timstr, NULL);
	init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
	comp_tic = init_toc;

	for (int tt = 0; tt < params.maxIters; tt++) {
		av_vels[tt] = timestep(params, ranks, &buffers, &cells, &tmp_cells, obstacles);
		// Swap cells and tmp_cells, so cells has the new values for the next timestep!
		t_speed swap = cells;
		cells = tmp_cells;
		tmp_cells = swap;
#ifdef DEBUG
		float density = total_density(params, cells);
		if (ranks.rank == 0) {
			printf("==timestep: %d==\n", tt);
			printf("av velocity: %.12E\n", av_vels[tt]);
			printf("tot density: %.12E\n", density);
		}
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
	float reynolds = calc_reynolds(params, &cells, obstacles);
	if (ranks.rank == 0) {
		printf("==done==\n");
		printf("Reynolds number:\t\t%.12E\n", reynolds);
		printf("Elapsed Init time:\t\t\t%.6lf (s)\n", init_toc - init_tic);
		printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
		printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc - col_tic);
		printf("Elapsed Total time:\t\t\t%.6lf (s)\n", tot_toc - tot_tic);
	}
	write_values(params, ranks, &cells, obstacles, av_vels);
	finalise(&params, &buffers, &cells, &tmp_cells, &obstacles, &av_vels);

	MPI_Finalize();
	return EXIT_SUCCESS;
}

float timestep(const t_param params, const t_ranks ranks, t_buffers* buffers, t_speed* cells, t_speed* tmp_cells, int* obstacles) {
	accelerate_flow(params, cells, obstacles);
	exchange_halos(params, ranks, buffers, cells);
	return timestep_merged(params, cells, tmp_cells, obstacles);
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles) {
	/* only modify the 2nd row of the grid */
	if (params.startY > params.ny - 2 || params.startY + params.local_ny <= params.ny - 2) return EXIT_SUCCESS;

	float* restrict c1 = cells->s1;
	float* restrict c3 = cells->s3;
	float* restrict c5 = cells->s5;
	float* restrict c6 = cells->s6;
	float* restrict c7 = cells->s7;
	float* restrict c8 = cells->s8;

	/* compute weighting factors */
	float w1 = params.density * params.accel / 9.f;
	float w2 = params.density * params.accel / 36.f;

	int jj = (params.ny - 2) - params.startY + 1;
	int jj_nx = jj * (params.local_nx + 2);

#pragma omp assume holds(params.local_nx % 16 == 0)
// #pragma omp simd aligned(c1, c3, c5, c6, c7, c8 : 64)
#pragma omp simd
	for (int ii = 1; ii < params.local_nx + 1; ii++) {
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
	for (int jj = 1; jj < params.local_ny + 1; jj++) {
		// these dont rely on ii, so calculate them here
		// int y_n = (jj + 1) % params.ny;
		// int y_s = (jj == 1) ? (jj + params.ny - 1) : (jj - 1);
		int y_n = jj + 1;
		int y_s = jj - 1;
		int jj_nx = jj * (params.local_nx + 2);
		int yn_nx = y_n * (params.local_nx + 2);
		int ys_nx = y_s * (params.local_nx + 2);

#pragma omp assume holds(params.local_nx % 16 == 0)
// #pragma omp simd aligned(c0, c1, c2, c3, c4, c5, c6, c7, c8, t0, t1, t2, t3, t4, t5, t6, t7, t8 : 64) reduction(+ : tot_u, tot_cells)
#pragma omp simd reduction(+ : tot_u, tot_cells)
		for (int ii = 1; ii < params.local_nx + 1; ii++) {
			// int x_e = (ii + 1) % params.nx;
			// int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
			int x_e = ii + 1;
			int x_w = ii - 1;
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
	float global_tot_u = 0.0f;
	int global_tot_cells = 0;
	MPI_Allreduce(&tot_u, &global_tot_u, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&tot_cells, &global_tot_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	return global_tot_u / (float)global_tot_cells;
}

void exchange_halos(const t_param params, const t_ranks ranks, t_buffers* buffers, t_speed* cells) {
	// X-Axis Exchange (Left <--> Right)

	// Pack X
	for (int jj = 1; jj <= params.local_ny; jj++) {
		int i_left = 1 + jj * (params.local_nx + 2);				 // Leftmost internal cell
		int i_right = params.local_nx + jj * (params.local_nx + 2);	 // Rightmost internal cell
		// subtract 1 to start from 0. Multiply by 3 because we are packing 3 speeds into each array section
		int b_idx = (jj - 1) * 3;

		// Going West: speeds 3 (W), 6 (NW), 7 (SW)
		buffers->send_west[b_idx + 0] = cells->s3[i_left];
		buffers->send_west[b_idx + 1] = cells->s6[i_left];
		buffers->send_west[b_idx + 2] = cells->s7[i_left];

		// Going East: speeds 1 (E), 5 (NE), 8 (SE)
		buffers->send_east[b_idx + 0] = cells->s1[i_right];
		buffers->send_east[b_idx + 1] = cells->s5[i_right];
		buffers->send_east[b_idx + 2] = cells->s8[i_right];
	}

	// Exchange X
	MPI_Sendrecv(buffers->send_west, buffers->x_buf_size, MPI_FLOAT, ranks.w_rank, 0,
				 buffers->recv_east, buffers->x_buf_size, MPI_FLOAT, ranks.e_rank, 0,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	MPI_Sendrecv(buffers->send_east, buffers->x_buf_size, MPI_FLOAT, ranks.e_rank, 1,
				 buffers->recv_west, buffers->x_buf_size, MPI_FLOAT, ranks.w_rank, 1,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	// Unpack X
	for (int jj = 1; jj <= params.local_ny; jj++) {
		int h_left = 0 + jj * (params.local_nx + 2);					   // West halo
		int h_right = (params.local_nx + 1) + jj * (params.local_nx + 2);  // East halo
		int b_idx = (jj - 1) * 3;

		// Received from East, placing in Right halo
		cells->s3[h_right] = buffers->recv_east[b_idx + 0];
		cells->s6[h_right] = buffers->recv_east[b_idx + 1];
		cells->s7[h_right] = buffers->recv_east[b_idx + 2];

		// Received from West, placing in Left halo
		cells->s1[h_left] = buffers->recv_west[b_idx + 0];
		cells->s5[h_left] = buffers->recv_west[b_idx + 1];
		cells->s8[h_left] = buffers->recv_west[b_idx + 2];
	}

	// Y-Axis Exchange (Down <--> Up)

	// Pack Y (Cols 0 to nx+1 - this includes x-halos for corners)
	for (int ii = 0; ii <= params.local_nx + 1; ii++) {
		int i_down = ii + 1 * (params.local_nx + 2);			  // Bottommost internal cell
		int i_up = ii + params.local_ny * (params.local_nx + 2);  // Topmost internal cell
		int b_idx = ii * 3;

		// Going South: speeds 4 (S), 7 (SW), 8 (SE)
		buffers->send_south[b_idx + 0] = cells->s4[i_down];
		buffers->send_south[b_idx + 1] = cells->s7[i_down];
		buffers->send_south[b_idx + 2] = cells->s8[i_down];

		// Going North: speeds 2 (N), 5 (NE), 6 (NW)
		buffers->send_north[b_idx + 0] = cells->s2[i_up];
		buffers->send_north[b_idx + 1] = cells->s5[i_up];
		buffers->send_north[b_idx + 2] = cells->s6[i_up];
	}

	// Exchange Y
	MPI_Sendrecv(buffers->send_south, buffers->y_buf_size, MPI_FLOAT, ranks.s_rank, 2,
				 buffers->recv_north, buffers->y_buf_size, MPI_FLOAT, ranks.n_rank, 2,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	MPI_Sendrecv(buffers->send_north, buffers->y_buf_size, MPI_FLOAT, ranks.n_rank, 3,
				 buffers->recv_south, buffers->y_buf_size, MPI_FLOAT, ranks.s_rank, 3,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	// Unpack Y
	for (int ii = 0; ii <= params.local_nx + 1; ii++) {
		int h_down = ii + 0 * (params.local_nx + 2);					// South halo
		int h_up = ii + (params.local_ny + 1) * (params.local_nx + 2);	// North halo
		int b_idx = ii * 3;

		// Received from North, placing in Top halo
		cells->s4[h_up] = buffers->recv_north[b_idx + 0];
		cells->s7[h_up] = buffers->recv_north[b_idx + 1];
		cells->s8[h_up] = buffers->recv_north[b_idx + 2];

		// Received from South, placing in Bottom halo
		cells->s2[h_down] = buffers->recv_south[b_idx + 0];
		cells->s5[h_down] = buffers->recv_south[b_idx + 1];
		cells->s6[h_down] = buffers->recv_south[b_idx + 2];
	}
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles) {
	int tot_cells = 0; /* no. of cells used in calculation */
	float tot_u;	   /* accumulated magnitudes of velocity for each cell */

	/* initialise */
	tot_u = 0.f;

	/* loop over all non-blocked cells */
	for (int jj = 1; jj < params.local_ny + 1; jj++) {
		int jj_nx = jj * (params.local_nx + 2);
		for (int ii = 1; ii < params.local_nx + 1; ii++) {
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

	float global_tot_u = 0.0f;
	int global_tot_cells = 0;
	MPI_Allreduce(&tot_u, &global_tot_u, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&tot_cells, &global_tot_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	return global_tot_u / (float)global_tot_cells;
}

// FIXME: ONLY ONE GUY READS FROM FILES????????????
int initialise(const char* paramfile, const char* obstaclefile,
			   t_param* params, t_ranks* ranks, t_buffers* buffers, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
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

	MPI_Comm_size(MPI_COMM_WORLD, &ranks->size);
	MPI_Comm_rank(MPI_COMM_WORLD, &ranks->rank);

	int p_dim = round(sqrt(ranks->size));
	// check if the number of ranks is a perfect square
	if (p_dim * p_dim != ranks->size) {
		printf("Number of MPI ranks (%d) is not a perfect square. Cannot make square blocks!\n", ranks->size);
		return EXIT_FAILURE;
	}
	// check if the grid perfectly divides into this processor grid
	else if (params->nx % p_dim != 0) {
		printf("Grid dimension (%d) is not divisible by processor grid dimension (%d)!\n", params->nx, p_dim);
		return EXIT_FAILURE;
	}

	int rank_x = ranks->rank % p_dim;
	int rank_y = ranks->rank / p_dim;
	ranks->w_rank = (rank_x == 0) ? (ranks->rank + p_dim - 1) : (ranks->rank - 1);
	ranks->e_rank = (rank_x == p_dim - 1) ? (ranks->rank - p_dim + 1) : (ranks->rank + 1);
	ranks->s_rank = (rank_y == 0) ? (ranks->rank + ranks->size - p_dim) : (ranks->rank - p_dim);
	ranks->n_rank = (rank_y == p_dim - 1) ? (ranks->rank - ranks->size + p_dim) : (ranks->rank + p_dim);

	params->local_nx = params->nx / p_dim;
	params->local_ny = params->ny / p_dim;
	params->startX = (ranks->rank * params->local_nx) % params->nx;
	params->startY = ((ranks->rank * params->local_nx) / params->nx) * params->local_ny;

	//sending/receiving 3 speeds
	buffers->x_buf_size = params->local_ny * 3;
	buffers->send_west = (float*)malloc(buffers->x_buf_size * sizeof(float));
	buffers->send_east = (float*)malloc(buffers->x_buf_size * sizeof(float));
	buffers->recv_west = (float*)malloc(buffers->x_buf_size * sizeof(float));
	buffers->recv_east = (float*)malloc(buffers->x_buf_size * sizeof(float));

	// include x diagonals
	buffers->y_buf_size = (params->local_nx + 2) * 3;
	buffers->send_south = (float*)malloc(buffers->y_buf_size * sizeof(float));
	buffers->send_north = (float*)malloc(buffers->y_buf_size * sizeof(float));
	buffers->recv_south = (float*)malloc(buffers->y_buf_size * sizeof(float));
	buffers->recv_north = (float*)malloc(buffers->y_buf_size * sizeof(float));

	// staggered allocation to prevent cache thrashing
	const int padding_floats = 16;
	int grid_size_sq = (params->local_nx + 2) * (params->local_ny + 2);	 // add halo spaces
	int stride_floats = grid_size_sq + padding_floats;
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
	*obstacles_ptr = malloc(sizeof(int) * grid_size_sq);
	if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

	/* initialise densities */
	float w0 = params->density * 4.f / 9.f;
	float w1 = params->density / 9.f;
	float w2 = params->density / 36.f;

	for (int jj = 1; jj < params->local_ny + 1; jj++) {
		int jj_nx = jj * (params->local_nx + 2);
		for (int ii = 1; ii < params->local_nx + 1; ii++) {
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
	for (int jj = 0; jj < params->local_ny + 2; jj++) {
		for (int ii = 0; ii < params->local_nx + 2; ii++) {
			(*obstacles_ptr)[ii + jj * (params->local_nx + 2)] = 0;
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

		if (xx < params->startX || xx >= params->startX + params->local_nx) continue;
		if (yy < params->startY || yy >= params->startY + params->local_ny) continue;
		/* assign to array */
		(*obstacles_ptr)[(xx - params->startX + 1) + (yy - params->startY + 1) * (params->local_nx + 2)] = blocked;
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

int finalise(const t_param* params, t_buffers* buffers, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
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

	free(buffers->send_west);
	free(buffers->send_east);
	free(buffers->recv_west);
	free(buffers->recv_east);
	free(buffers->send_south);
	free(buffers->send_north);
	free(buffers->recv_south);
	free(buffers->recv_north);

	return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, t_speed* cells, int* obstacles) {
	const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

	return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells) {
	float total = 0.f; /* accumulator */

	for (int jj = 1; jj < params.local_ny + 1; jj++) {
		for (int ii = 1; ii < params.local_nx + 1; ii++) {
			int idx = ii + jj * (params.local_nx + 2);
			total = cells->s0[idx] + cells->s1[idx] + cells->s2[idx] + cells->s3[idx] + cells->s4[idx] + cells->s5[idx] + cells->s6[idx] + cells->s7[idx] + cells->s8[idx];
		}
	}
	float global_total = 0.f;
	MPI_Allreduce(&total, &global_total, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	return global_total;
}

int write_values(const t_param params, const t_ranks ranks, t_speed* cells, int* obstacles, float* av_vels) {
	FILE* fp;					  /* file pointer */
	const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
	// float local_density;		  /* per grid cell sum of densities */
	// float pressure;				  /* fluid pressure in grid cell */
	// float u_x;					  /* x-component of velocity in grid cell */
	// float u_y;					  /* y-component of velocity in grid cell */
	// float u;					  /* norm--root of summed squares--of u_x and u_y */

	if (ranks.rank == 0) {
		fp = fopen(FINALSTATEFILE, "w");
		if (fp == NULL) die("could not open file output file", __LINE__, __FILE__);
		fclose(fp);
	}
	for (int turn = 0; turn < ranks.size; turn++) {
		MPI_Barrier(MPI_COMM_WORLD);

		if (ranks.rank == turn) {
			fp = fopen(FINALSTATEFILE, "a");
			if (fp == NULL) die("could not open file output file", __LINE__, __FILE__);

			for (int jj = 1; jj < params.local_ny + 1; jj++) {
				for (int ii = 1; ii < params.local_nx + 1; ii++) {
					int idx = ii + jj * (params.local_nx + 2);

					int global_x = params.startX + (ii - 1);
					int global_y = params.startY + (jj - 1);

					float u_x, u_y, u, pressure;

					/* an occupied cell */
					if (obstacles[idx]) {
						u_x = u_y = u = 0.f;
						pressure = params.density * c_sq;
					}
					/* no obstacle */
					else {
						float local_density = cells->s0[idx] + cells->s1[idx] + cells->s2[idx] + cells->s3[idx] + cells->s4[idx] + cells->s5[idx] + cells->s6[idx] + cells->s7[idx] + cells->s8[idx];

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
					fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", global_x, global_y, u_x, u_y, u, pressure, obstacles[idx]);
				}
			}
			fclose(fp);
		}
	}

	// FIXME: USE MPI_Allreduce ONLY ONCE HERE!!!!!!!!!!!!!!!!!!
	if (ranks.rank == 0) {
		fp = fopen(AVVELSFILE, "w");
		if (fp == NULL) {
			die("could not open file output file", __LINE__, __FILE__);
		}
		for (int ii = 0; ii < params.maxIters; ii++) {
			fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
		}
		fclose(fp);
	}

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
