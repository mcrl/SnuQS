#include "socl.h"
#include <stdio.h>
#include <stdlib.h>

int main()
{
	int err;
	err = socl_init();
	if (err) {
		fprintf(stderr, "socl_init failed\n");
		exit(EXIT_FAILURE);
	}
	err = socl_finalize();
	if (err) {
		fprintf(stderr, "socl_finalize failed\n");
		exit(EXIT_FAILURE);
	}
	return 0;
}
