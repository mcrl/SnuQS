#include "spsdb.h"
#include "spsdu.h"
#include <stdio.h>
#include <assert.h>

#define COUNT (1ul << 20)
int entries[COUNT];

void test_spsdb(int *entries) {
	struct spsdb spsdb;
	spsdb_init(&spsdb, COUNT);

	for (size_t i = 0; i < COUNT; i++) {
		spsdb_enqueue(&spsdb, &entries[i]);
	}

	for (size_t i = 0; i < COUNT; i++) {
		int* ptr = spsdb_dequeue(&spsdb);
		assert(*ptr == i);
	}

	spsdb_deinit(&spsdb);
	printf("spsdb test done\n");
}

void test_spsdu(int *entries) {
	struct spsdu spsdu;
	spsdu_init(&spsdu);

	for (size_t i = 0; i < COUNT; i++) {
		spsdu_enqueue(&spsdu, &entries[i]);
	}

	for (size_t i = 0; i < COUNT; i++) {
		int* ptr = spsdu_dequeue(&spsdu);
		assert(*ptr == i);
	}

	spsdu_deinit(&spsdu);
	printf("spsdu test done\n");
}

int main(int argc, char *argv[]) {
	for (size_t i = 0; i < COUNT; i++) {
		entries[i] = i;
	}

	test_spsdb(entries);
	test_spsdu(entries);

	return 0;
}
