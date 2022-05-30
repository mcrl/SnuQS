#include "socl.h"
#include <stdio.h>
#include <stdlib.h>

int main()
{
	printf("socl-io-test\n");
	int err;
	err = socl_io_init();
	if (err) {
		fprintf(stderr, "socl_io_init failed\n");
		exit(EXIT_FAILURE);
	}
	err = socl_io_finalize();
	if (err) {
		fprintf(stderr, "socl_io_finalize failed\n");
		exit(EXIT_FAILURE);
	}
	return 0;
}
