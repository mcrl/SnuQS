#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "socl-io.h"
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static const char* config_file = "socl-io.config";

static size_t socl_io_num_devices = 0;
static struct socl_io_device *socl_io_devices = NULL;
static size_t socl_io_block_size = 0;
static size_t socl_io_row_size = 0;

static int socl_io_init_device(struct socl_io_device *dev, const char *name)
{
	int err = 0;
	strncpy(dev->name, name, SOCL_IO_NAME_LENGTH);

#ifdef _SOCL_IO_URING_SUPPORTED_
	err = io_uring_queue_init(SOCL_IO_QUEUE_DEPTH, &dev->ring, 0);
	if (err) {
		fprintf(stderr, "Cannot initialize io_uring queue %d\n", err);
		return err;
	}
#else
	// Must initialized
	memset(&dev->ctx, 0, sizeof(aio_context_t));
	err = io_setup(SOCL_IO_QUEUE_DEPTH, &dev->ctx);
	if (err) {
		return err;
	}
#endif

	dev->fd = open(name, O_RDWR|O_DIRECT);
	if (dev->fd == -1) {
		fprintf(stderr, "open fail %s\n", name);
		return -EINVAL;
	}

	return err;
}

int socl_io_init()
{
	FILE *fp = fopen(config_file, "rb");
	if (!fp) {
		fprintf(stderr, "Cannot open file %s\n", config_file);
		return -EINVAL;
	}

	char *line = NULL;
	ssize_t nread;
	size_t len;
	char *name;

	//
	// Get num of devices
	//
	while ((nread = getline((char**)&line, &len, fp)) != -1) {
		name = strtok(line, " \n");
		// Ignore empty a line or a line that starts with #
		if (!name || name[0] == '#') {
			continue;
		}
		socl_io_num_devices++;
	}

	socl_io_devices = malloc(sizeof(struct socl_io_device) * socl_io_num_devices);
	if (!socl_io_devices) {
		fprintf(stderr, "Cannot allocate socl_io_devices\n");;
		fclose(fp);
		return -ENOMEM;
	}


	//
	// Parse config file
	//
	int i = 0;
	int err;
	rewind(fp);
	while ((nread = getline((char**)&line, &len, fp)) != -1) {
		name = strtok(line, " \n");
		// Ignore empty a line or a line that starts with #
		if (!name || name[0] == '#') {
			continue;
		}

		err = socl_io_init_device(&socl_io_devices[i++], name);
		if (err) {
			fclose(fp);
			return err;
		}
	}

	socl_io_block_size = SOCL_IO_DEFAULT_BLOCK_SIZE;
	socl_io_row_size = socl_io_block_size * socl_io_num_devices;

	fclose(fp);
	return 0;
}

int socl_io_finalize()
{
	// TODO
	return 0;
}

int socl_io_set_block_size(size_t block_size)
{
	socl_io_block_size = block_size;
	socl_io_row_size = socl_io_block_size * socl_io_num_devices;
	return 0;
}

int socl_io_hdlr_init(socl_io_hdlr_t *hdlr)
{
	socl_io_hdlr_t new_hdlr = malloc(sizeof(**hdlr));
	if (!new_hdlr) {
		fprintf(stderr, "Cannot allocate hdlr\n");
		return -ENOMEM;
	}

	*hdlr = new_hdlr;
	return 0;
}

int socl_io_hdlr_deinit(socl_io_hdlr_t hdlr)
{
	free(hdlr);
	return 0;
}

#ifdef _SOCL_IO_URING_SUPPORTED_
static int socl_io_enqueueIO(socl_io_hdlr_t hdlr, void *buf, size_t addr, size_t count, int isH2S)
{
	int err;
	struct io_uring_sqe *sqe;
	while (count > 0) {
		int devno = (addr / socl_io_block_size) % socl_io_num_devices;
		struct socl_io_device *dev = &socl_io_devices[devno];
		size_t off = (addr / socl_io_row_size) * socl_io_block_size
			        + addr % socl_io_block_size;
		size_t transfer_count = socl_io_block_size - (off % socl_io_block_size);

		if (count < transfer_count)
			transfer_count = count;

		size_t left_count = transfer_count;

		while (left_count > 0) {
			sqe = io_uring_get_sqe(&dev->ring);
			if (!sqe) {
				printf("queue_write failed\n");
				return -ENOENT;
			}

			size_t actual_count = left_count;
			if (actual_count > SOCL_IO_MAX_TRANSFER_SIZE) {
				actual_count = SOCL_IO_MAX_TRANSFER_SIZE;
			}
			if (isH2S) {
				io_uring_prep_write(sqe, dev->fd, (void*)addr, actual_count, off);
			} else {
				io_uring_prep_read(sqe, dev->fd, (void*)addr, actual_count, off);
			}
			addr += actual_count;
			off += actual_count;
			left_count -= actual_count;
		}

		count -= transfer_count;
	}

	for (int d = 0; d < socl_io_num_devices; ++d) {
		struct socl_io_device *dev = &socl_io_devices[d];
		err = io_uring_submit(&dev->ring);
		if (err < 0) {
			fprintf(stderr, "Cannot submit I/O commands to device %d\n", d);
			return err;
		}
	}

	return 0;
}
#else
static int socl_io_enqueueIO(socl_io_hdlr_t hdlr, void *buf, size_t addr, size_t count, int isH2S)
{
	return 0;
}
#endif

int socl_io_enqueueH2S(socl_io_hdlr_t hdlr, void *buf, size_t addr, size_t count)
{
	return socl_io_enqueueIO(hdlr, buf, addr, count, 1);
}

int socl_io_enqueueS2H(socl_io_hdlr_t hdlr, void *buf, size_t addr, size_t count)
{
	return socl_io_enqueueIO(hdlr, buf, addr, count, 1);
}

int socl_io_enqueueSync(socl_io_hdlr_t hdlr)
{
	return 0;
}
