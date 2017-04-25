#include "csv.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// because i so hate strtok i gave my only func
int get_next_csv_int( const char* str, int* curr_idx ) {
	int len = strlen( str );
	int v = -1;
	for (int i = *curr_idx; i < len - 1; i++ ) {
		if ( str[i] == ',' ) {
			sscanf( &str[i], ",%i", &v);
			//printf("found %i at index %i where curr_idx = %i\n", v, i, *curr_idx);
			*curr_idx = i + 1;
			return v;
		}
	}
	*curr_idx = -1;
	return v;
}

void read_csv( const char* file_name, int nentries, float labels[nentries], float pixels[nentries][784] ) {
	assert( file_name );
	assert( labels );
	assert( pixels );

	int line_count = 0;
	
	// 28x28 images = 784
	// label number,784 values
	
	FILE* f = fopen( file_name, "r" );
	assert( f );
	
	char line[2048];
	line[0] = '\0';
	while ( fgets ( line, 2048, f ) ) {
		if (nentries == line_count) {
			break;
		}
		int label = -1;
		sscanf( line, "%i,", &label);
		labels[line_count] = (float)label;
		//printf("label = %i\n", label);
		int curr_idx = 0;
		int pixel_count = 0;
		while (curr_idx >= 0 ) {
			int val = get_next_csv_int( line, &curr_idx );
			if (curr_idx < 0 || pixel_count >= (28 * 28)) {
				break;
			}
			//printf("pixels[%i][%i]=%i\n", line_count, pixel_count,val);
			float fval = (float)val / 255.0f;
			fval = fval * 0.99f + 0.01f;
			pixels[line_count][pixel_count] = fval;
			pixel_count++;
		}
		line_count++;
	}
	
	fclose( f );
	printf("line count %i\n", line_count);
}

