/* 14 Apr 2017 Dr Anton Gerdelan <antonofnote@gmail.com>
C99    compile: gcc -std=c99 main.c -lm

Notes:
* timer only works on linux
Knobs and Dials:
 * try normal distribution for initial random weights
 * learning rate
 * number of nodes in hidden layer
 * repeat entire training set on same network by i `epochs`
 * size of training set and test set
Todo:
 * output results to .xy for gnuplot comparisons
 * save trained network weights to file for quick requery
*/

#define _POSIX_C_SOURCE 199309L // for the timer on linux
#include "csv.h"
#include "matrices.h"
#include <time.h>

//#define DEBUG_PRINT
//#define DRAW_OUT_IMAGES
//#define PRINT_TRAINING

typedef struct network_t {
	int ninputs;
	int nhiddens;
	int noutputs;
	int max_steps;
	float learning_rate;
	float *input_to_hidden_weights;
	float *input_to_hidden_delta_weights;
	float *hidden_to_output_weights;
	float *hidden_to_output_back_weights;
	float *hidden_to_output_delta_weights;
	float *inputs;
	float *hiddens;
	float *outputs;
	float *output_errors;
	float *hidden_errors;
	float *hiddens_deltas;
	float *outputs_deltas;
} network_t;

static network_t network;

void init_network( int ninputs, int nhiddens, int noutputs, float learning_rate,
									 int max_steps );
// note: In C99, you can provide the dimensions of the array before passing it
void train_network( int nsamples, int ninputs, int noutputs,
										float inputs_list[nsamples][ninputs],
										float targets_list[nsamples][noutputs] );
void query( const float *inputs );
double apg_time_linux();

int main() {
	int num_pixels = 28 * 28;
	int ninputs = num_pixels;
	int nhiddens = 100;
	int noutputs = 10;
	float learning_rate = 0.3f;
	int max_steps = 1; // i dont think they do any steps in book -- suspect this will cause overtraining
	int epochs = 10; // 1 to 10 went from ~80% to ~95% accuracy

	int num_lines = 2000; // more samples = more accurate.
	float labels[num_lines];
	float pixels[num_lines][num_pixels];
	read_csv( "mnist_train.csv", num_lines, labels, pixels );
#ifdef DRAW_OUT_IMAGES
	for ( int j = 0; j < num_lines; j++ ) { // write first image out to test it worked
		char tmp[128];
		sprintf( tmp, "img%i_label_%i.ppm", j, (int)labels[j] );
		FILE *f = fopen( tmp, "w" );
		assert( f );
		fprintf( f, "P3\n28 28\n255\n" );
		for ( int i = 0; i < num_pixels; i++ ) {
			int v =
				(int)( 255.0f - 255.0f * ( ( pixels[j][i] - 0.01f ) * ( 1.0f / 0.99f ) ) );
			fprintf( f, "%i %i %i\n", v, v, v );
		}
		fclose( f );
	}
#endif

	init_network( ninputs, nhiddens, noutputs, learning_rate, max_steps );

	for ( int epoch = 0; epoch < epochs; epoch++ ) { // training run
		// prepare targets in form       0   1   2    3   4   5 ... 9
		//                              [0.1 1.0 0.1 0.1 0.1 0.1...0.1]
		float targets_list[num_lines][noutputs];
		for ( int l = 0; l < num_lines; l++ ) {
			for ( int o = 0; o < noutputs; o++ ) {
				if ( (int)labels[l] == o ) {
					targets_list[l][o] = 1.0f;
				} else {
					targets_list[l][o] = 0.1f; // lowest useful signal
				}
			}
		}

#ifdef DEBUG_PRINT
		printf( "targets list:\n" );
		print_mat( targets_list[5], noutputs, 1 ); // '9'
#endif

		train_network( num_lines, ninputs, noutputs, pixels, targets_list );
	}

	// TODO load test set and do this

	{
		int num_lines = 100;
		float labels[num_lines];
		float pixels[num_lines][num_pixels];
		read_csv( "mnist_test.csv", num_lines, labels, pixels );
		int sum_correct = 0;
		for ( int i = 0; i < num_lines; i++ ) {
			query( pixels[i] );
			int maxi = 0;
			float maxf = network.outputs[0];
			for ( int j = 1; j < network.noutputs; j++ ) {
				if ( network.outputs[j] > maxf ) {
					maxf = network.outputs[j];
					maxi = j;
				}
			}
			printf( "queried. our answer %i (%f conf) - correct answer %i\n", maxi,
							network.outputs[maxi], (int)labels[i] );
			if ( maxi == (int)labels[i] ) {
				sum_correct++;
			}
#ifdef DEBUG_PRINT
			print_mat( network.outputs, network.noutputs, 1 );
#endif
			//		printf( "getchar:\n" );
			//	getchar();
		}
		float accuracy = (float)sum_correct / (float)num_lines;
		printf( "total accuracy = %f\n", accuracy );
	}
	return 0;
}

void init_network( int ninputs, int nhiddens, int noutputs, float learning_rate,
									 int max_steps ) {
	printf( "initialising...\n" );
	printf( "math e constant = %.12f\n", M_E );
	unsigned int seed = time( NULL );
	srand( seed );
	printf( "seed = %u\n", seed );

	network.ninputs = ninputs;
	network.nhiddens = nhiddens;
	network.noutputs = noutputs;
	network.max_steps = max_steps;
	network.learning_rate = learning_rate;

	{ // program memory allocation
		size_t sz_a = sizeof( float ) * ninputs * nhiddens;
		size_t sz_b = sizeof( float ) * nhiddens * noutputs;
		size_t sz_inputs = sizeof( float ) * ninputs;
		size_t sz_hiddens = sizeof( float ) * nhiddens;
		size_t sz_outputs = sizeof( float ) * noutputs;
		// matrices between layers
		network.input_to_hidden_weights = (float *)malloc( sz_a );
		assert( network.input_to_hidden_weights );
		network.input_to_hidden_delta_weights = (float *)malloc( sz_a );
		assert( network.input_to_hidden_delta_weights );
		network.hidden_to_output_weights = (float *)malloc( sz_b );
		assert( network.hidden_to_output_weights );
		network.hidden_to_output_back_weights = (float *)malloc( sz_b );
		assert( network.hidden_to_output_back_weights );
		network.hidden_to_output_delta_weights = (float *)malloc( sz_b );
		assert( network.hidden_to_output_delta_weights );
		// vectors for each layer
		network.inputs = (float *)calloc( ninputs, sizeof( float ) );
		assert( network.inputs );
		network.hiddens = (float *)calloc( nhiddens, sizeof( float ) );
		assert( network.hiddens );
		network.hidden_errors = (float *)calloc( nhiddens, sizeof( float ) );
		assert( network.hidden_errors );
		network.outputs = (float *)calloc( noutputs, sizeof( float ) );
		assert( network.outputs );
		network.output_errors = (float *)calloc( noutputs, sizeof( float ) );
		assert( network.output_errors );
		network.hiddens_deltas = (float *)calloc( nhiddens, sizeof( float ) );
		network.outputs_deltas = (float *)calloc( noutputs, sizeof( float ) );
		size_t sz_total =
			sz_a * 2 + sz_b * 3 + sz_inputs + sz_hiddens * 3 + sz_outputs * 3;
		printf( "allocated %lu bytes (%lu kB) (%lu MB)\n", sz_total, sz_total / 1024,
						sz_total / ( 1024 * 1024 ) );
#ifdef DEBUG_PRINT
		printf( "  inputs %lu\n", sz_inputs );
		printf( "  hiddens %lu\n", sz_hiddens );
		printf( "  hidden errors %lu\n", sz_hiddens );
		printf( "  hidden deltas %lu\n", sz_hiddens );
		printf( "  outputs %lu\n", sz_outputs );
		printf( "  output errors %lu\n", sz_outputs );
		printf( "  output deltas %lu\n", sz_outputs );
		printf( "  weights input->hidden %lu\n", sz_a );
		printf( "  delta weights input->hidden %lu\n", sz_a );
		printf( "  weights hidden->outputs %lu\n", sz_b );
		printf( "  weights outputs->hidden %lu\n", sz_b );
		printf( "  delta weights hidden->outputs %lu\n", sz_b );
#endif
	}

	randomise_mat( network.input_to_hidden_weights, ninputs, nhiddens );
	randomise_mat( network.hidden_to_output_weights, nhiddens, noutputs );
}

void train_network( int nsamples, int ninputs, int noutputs,
										float inputs_list[nsamples][ninputs],
										float targets_list[nsamples][noutputs] ) {
	assert( inputs_list );
	assert( targets_list );

	printf( "training...\n" );
	double start_time = apg_time_linux();

	// samples loop here
	for ( int curr_sample = 0; curr_sample < nsamples; curr_sample++ ) {

		// learning rate loop here somewhere -- probably vice versa?
		for ( int step = 0; step < network.max_steps; step++ ) {

			{ // first part is same as query()
				memcpy( network.inputs, inputs_list[curr_sample],
								network.ninputs * sizeof( float ) );
				// ANTON: i switched around row and cols vars here because valgrind told me
				// i was wrong
				mult_mat_vec( network.input_to_hidden_weights, network.nhiddens,
											network.ninputs, network.inputs, network.hiddens );
				sigmoid( network.hiddens, network.hiddens, network.nhiddens );
				// ANTON: i switched around row and cols vars here because valgrind told me
				// i was wrong
				mult_mat_vec( network.hidden_to_output_weights, network.noutputs,
											network.nhiddens, network.hiddens, network.outputs );
				sigmoid( network.outputs, network.outputs, network.noutputs );
			}
			{ // second part compares result to desired result and back-propagates error
				for ( int i = 0; i < network.noutputs; i++ ) {
					network.output_errors[i] =
						targets_list[curr_sample][i] - network.outputs[i];

					// printf( "errors[%i] = %f\n", i, network.output_errors[i] );
				}

				// transpose the hidden->output matrix to go backwards
				transpose_mat( network.hidden_to_output_weights,
											 network.hidden_to_output_back_weights, network.nhiddens,
											 network.noutputs );
				// work out proportional errors for hidden layer too
				// ANTON: i switched around row and cols vars here because valgrind told me
				// i was wrong
				mult_mat_vec( network.hidden_to_output_back_weights, network.nhiddens,
											network.noutputs, network.output_errors,
											network.hidden_errors );

				{ // adjust hidden->output weights
					for ( int i = 0; i < network.noutputs; i++ ) {
						network.outputs_deltas[i] = network.output_errors[i] *
																				network.outputs[i] *
																				( 1.0f - network.outputs[i] );
						//	printf( "output delta[%i] = %f\n", i, network.outputs_deltas[i] );
					}
					colrow_vec_mult( network.outputs_deltas, network.hiddens,
													 network.noutputs, network.nhiddens,
													 network.hidden_to_output_delta_weights );

					//	printf( "delta weights matrix:\n" );
					//	print_mat( network.hidden_to_output_delta_weights, network.noutputs,
					//						 network.nhiddens );

					for ( int i = 0; i < ( network.nhiddens * network.noutputs ); i++ ) {
						network.hidden_to_output_delta_weights[i] *= network.learning_rate;
						//	printf( "output weight %i before = %f\n", i,
						//					network.hidden_to_output_weights[i] );
						network.hidden_to_output_weights[i] +=
							network.hidden_to_output_delta_weights[i];
						//	printf( "output weight %i after = %f\n", i,
						//					network.hidden_to_output_weights[i] );
					}
				}

				{ // adjust input->hidden weights
					for ( int i = 0; i < network.nhiddens; i++ ) {
						network.hiddens_deltas[i] = network.hidden_errors[i] *
																				network.hiddens[i] *
																				( 1.0f - network.hiddens[i] );
						//	printf( "hidden delta[%i] = %f\n", i, network.hiddens_deltas[i] );
					}
					colrow_vec_mult( network.hiddens_deltas, network.inputs, network.nhiddens,
													 network.ninputs, network.input_to_hidden_delta_weights );

					// printf( "delta weights matrix:\n" );
					//	print_mat( network.input_to_hidden_delta_weights, network.nhiddens,
					//					 network.ninputs );

					for ( int i = 0; i < ( network.ninputs * network.nhiddens ); i++ ) {
						network.input_to_hidden_delta_weights[i] *= network.learning_rate;

						//	printf( "output weight %i before = %f\n", i,
						//				network.input_to_hidden_weights[i] );
						network.input_to_hidden_weights[i] +=
							network.input_to_hidden_delta_weights[i];

						//	printf( "output weight %i after = %f\n", i,
						//				network.input_to_hidden_weights[i] );
					}
				}

			} // end back-propagation

#ifdef PRINT_TRAINING
			if ( step % 100 == 0 ) {
				printf( "end of step %i\n", step );
				float error_sum = 0.0f;
				for ( int i = 0; i < network.noutputs; i++ ) {
					printf( "output[%i] = %f target = %f\n", i, network.outputs[i],
									targets_list[curr_sample][i] );
					error_sum += ( targets_list[curr_sample][i] - network.outputs[i] );
				}
				printf( "error sum was %f\n", error_sum );
			}
#endif
		}
	} // end steps loop

	double end_time = apg_time_linux();
	double elapsed = end_time - start_time;
	printf( "training took %f seconds\n", elapsed );
}

// note: make sure inputs vector is populated first
void query( const float *inputs ) {
	assert( inputs );

	printf( "querying...\n" );
	memcpy( network.inputs, inputs, network.ninputs * sizeof( float ) );
	memset( network.outputs, 0, network.noutputs );

	// feed input -> hidden layer
	// ANTON reversed rows/cols here too
	mult_mat_vec( network.input_to_hidden_weights, network.nhiddens, network.ninputs,
								network.inputs, network.hiddens );
	// apply sigmoid to dampen activation signal before output
	sigmoid( network.hiddens, network.hiddens, network.nhiddens );
	// feed hidden -> output layer
	// ANTON reversed rows/cols here too
	mult_mat_vec( network.hidden_to_output_weights, network.noutputs,
								network.nhiddens, network.hiddens, network.outputs );
	// apply sigmoid to dampen activation signal before output
	sigmoid( network.outputs, network.outputs, network.noutputs );
}

// get a monotonic time value in seconds w/ nanosecond precision (linux only)
// value is some arbitrary system time but is invulnerable to clock changes
// CLOCK_MONOTONIC -- vulnerable to adjtime() and NTP changes
// CLOCK_MONOTONIC_RAW -- vulnerable to voltage and heat changes
double apg_time_linux() {
#ifdef _WIN32
	return 0.0;
#elif __APPLE__
	return 0.0;
#else
	struct timespec t;
	static double prev_value = 0.0;
	int r = clock_gettime( CLOCK_MONOTONIC, &t );
	if ( r < 0 ) {
		fprintf( stderr, "WARNING: could not get time value\n" );
		return prev_value;
	}
	double ns = t.tv_nsec;
	double s = ns * 0.000000001;
	time_t tts = t.tv_sec;
	s += difftime( tts, 0 );
	prev_value = s;
	return s;
#endif
}
