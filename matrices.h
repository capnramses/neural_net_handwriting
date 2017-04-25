/* 13 Apr 2017 Anton Gerdelan <antonofnote@gmail.com>
my own little version of what people use "numpy" for neural nets

the transpose and N*M matrix/vector funcs here compile to 83 lines of x86_64 asm
with -O3 and ~100 with no opt.
*/

#pragma once

// TODO randmat using a normal distribution around num nodes

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef M_E
#define M_E 2.71828
#endif

static inline void print_mat( const float *mat, int rows, int cols );
static inline void transpose_mat( const float *in, float *out, int rows, int cols );
static inline void mult_mat_vec( const float *mat, int rows, int cols,
																 const float *vec_in, float *vec_out );
static inline void randomise_mat( float *mat, int rows, int cols );
static inline void sigmoid( const float *vec_in, float *vec_out, int elements );
static inline void colrow_vec_mult( const float *col_vec, const float *row_vec,
																		int rows, int cols, float *matrix_out );

// my matrices, of any dimensions, are treated as 1D arrays and are arranged in
// column-major memory order. functions use column-major mathematical conventions
static inline void print_mat( const float *mat, int rows, int cols ) {
	assert( mat );

	// printf( "\n" );
	for ( int r = 0; r < rows; r++ ) {
		printf( "| " );
		for ( int c = 0; c < cols; c++ ) {
			printf( "%f ", mat[c * rows + r] );
		}
		printf( "|\n" );
	}
}

// notes:
// * MxN in results in NxM out
// * can avoid this func by just accessing matrix elements in different order
static inline inline void transpose_mat( const float *in, float *out, int rows,
																				 int cols ) {
	assert( in );
	assert( out );

	for ( int m = 0; m < rows; m++ ) {
		for ( int n = 0; n < cols; n++ ) {
			int in_idx = m + n * rows;
			int out_idx = n + m * cols;
			out[out_idx] = in[in_idx];
		}
	}
}

// notes:
// * input vector must have elements == matrix COLS
// * output vector must have elements == matrix ROWS
// note: can ~speed up by switching loops around. i went with formula repro 1st
static inline inline void mult_mat_vec( const float *mat, int rows, int cols,
																				const float *vec_in, float *vec_out ) {
	assert( mat );
	assert( vec_in );
	assert( vec_out );

	// general formula is for ea matrix row mult ea col el in row w eac vec el
	// m[row 0, col 0] * v[0] + m[row 0][col 1] * v[1] ... m[row 0][col m] * v[n]
	// m[row 1, col 0] * v[0] + m[row 1][col 1] * v[1] ... m[row 1][col m] * v[n]
	// ...
	// m[row n, col 0] * v[0] + m[row n][col 1] * v[1] ... m[row n][col m] * v[n]
	//memset( vec_out, 0, sizeof( float ) * rows );
	for ( int r = 0; r < rows; r++ ) {
		vec_out[r] = 0.0f;
		for ( int c = 0; c < cols; c++ ) {
			int mat_idx = r + c * rows;
			float m = mat[mat_idx];
			float vval = vec_in[c];
			vec_out[r] += ( m * vval );
		}
	}
}

// randomises matrix weight values between 0.1 and 0.9
// note: uses rand(). remember to srand()
static inline void randomise_mat( float *mat, int rows, int cols ) {
	assert( mat );

	for ( int i = 0; i < rows * cols; i++ ) {
		float val = (float)rand() / RAND_MAX;
		// val = val * 0.8 + 0.1;
		val -= 0.5f; // allow negative range -0.5 to 0.5
		mat[i] = val;
	}
}

static inline void sigmoid( const float *vec_in, float *vec_out, int elements ) {
	assert( vec_in );
	assert( vec_out );

	for ( int i = 0; i < elements; i++ ) {
		vec_out[i] = 1.0f / ( 1.0f + powf( M_E, -vec_in[i] ) );
	}
}

// note: final rows = rows in first, final cols = cols in second
//	|1|           | 1x2 1x3 1x4 |   | 2 3 4  |
//	|2| [2 3 4] = | 2x2 2x3 2x4 | = | 4 6 8  |
//	|3|           | 3x2 3x3 3x4 |   | 6 9 12 |
static inline void colrow_vec_mult( const float *col_vec, const float *row_vec,
																		int rows, int cols, float *matrix_out ) {
	for ( int c = 0; c < cols; c++ ) {
		for ( int r = 0; r < rows; r++ ) {
			int mat_idx = r + c * rows;
			matrix_out[mat_idx] = col_vec[r] * row_vec[c];
		}
	}
}