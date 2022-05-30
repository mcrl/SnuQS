#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <unistd.h>
#include "baio.h"
#include "baio_utils.h"

//#define ALIGN_SIZE (1ul << 21)
#define ALIGN_SIZE 4096

#define MAX_TRANSFER_SIZE 2147479552ul


static int baio_queue_submit_device(struct baio_device *dev);

static int baio_init_device(struct baio_device *dev, int queue_depth, const char *devname, size_t size_in_GB)
{
	int err = 0;
	strncpy(dev->devname, devname, 31);
	dev->size_in_GB = size_in_GB;
	dev->fd = open(devname, O_RDWR|O_DIRECT);
	if (dev->fd == -1) {
		printf("open fail %s\n", devname);
		return -EINVAL;
	}

	dev->num_reads = 0;
	dev->num_writes = 0;

	err = io_uring_queue_init(queue_depth, &dev->rring, 0);
	if (err) {
		printf("ASDF\n");
		return err;
	}
	err = io_uring_queue_init(queue_depth, &dev->wring, 0);
	if (err) {
		printf("ESDF\n");
		return err;
	}
	return err;
}

int baio_init(struct baio_handle *hdlr, unsigned queue_depth, const char *config_file)
{
	int err;

	if (!hdlr) {
		return -EINVAL;
	}

	if (!config_file)
		config_file = BAIO_CONFIG_FILE;

	hdlr->devices = (struct baio_device*)malloc(sizeof(struct baio_device) * BAIO_MAX_DEVS);
	if (!hdlr->devices) {
		printf("Cannot allocate memory");
		return -ENOMEM;
	}

	for (int i = 0; i < BAIO_MAX_DEVS; i++) {
		memset(&hdlr->devices[i], 0, sizeof(struct baio_device));
	}

	FILE *fp = fopen(config_file, "rb");
	if (!fp) {
		printf("Cannot open file\n");
		return -EINVAL;
	}

	char *line = NULL;
	ssize_t nread;
	size_t len;
	char *devname;

	size_t size_in_GB = (1ul << 11);
	hdlr->num_devices = 0;
	while ((nread = getline((char**)&line, &len, fp)) != -1) {
		if (hdlr->num_devices + 1 >= BAIO_MAX_DEVS) {
			free(line);
			fclose(fp);
			printf("A\n");
			return -EINVAL;
		}

		devname = strtok(line, " \n");
		if (!devname || devname[0] == '#') {
			continue;
		}

		err = baio_init_device(&hdlr->devices[hdlr->num_devices], queue_depth, devname, size_in_GB);
		if (err) {
			free(line);
			fclose(fp);
			printf("B\n");
			return -EINVAL;
		}

		hdlr->num_devices++;
	}
	fclose(fp);
	hdlr->queue_depth = queue_depth;
	return 0;
}

int baio_setup_blocks(struct baio_handle *hdlr, 
		size_t block_size_per_device,
		size_t row_size,
		size_t width,
		size_t base
		)
{
	if (!hdlr)
		return -EINVAL;

	hdlr->block_size_per_device = ROUND_UP(block_size_per_device, ALIGN_SIZE);
	hdlr->row_size = row_size;
	hdlr->width = width;
	hdlr->base = base;
	return 0;
}

void baio_finalize(struct baio_handle *hdlr)
{
	if (!hdlr) return;
}

void* baio_malloc(size_t size) 
{
	void *ptr;
	if (posix_memalign(&ptr, ALIGN_SIZE, size)) {
		return NULL;
	}
	return ptr;
}

inline static int get_dev(struct baio_handle *hdlr, size_t addr)
{
	size_t i = (addr / hdlr->block_size_per_device) % hdlr->num_devices;
	//size_t r = (addr / hdlr->row_size);
	size_t r = (addr / (1ul << 34));
	size_t k = (hdlr->width / hdlr->block_size_per_device) % hdlr->num_devices;

	printf("%lu + %lu * %lu = %lu\n", i, r, k, i + r * k);

	return (i + r * k) % hdlr->num_devices;

	//size_t r = (addr / row_size);
//	size_t r = (addr / (1ul << 34));
//
//	//(addr / hdlr->row_size)
//
//
//	size_t off = (hdlr->base / hdlr->block_size_per_device);
//
//	int d = (addr / (hdlr->block_size_per_device)) % hdlr->num_devices;
//	d = (d + off * r) % hdlr->num_devices;
//	return d;
}

inline static size_t get_offset(struct baio_handle *hdlr, size_t addr)
{
	size_t block_size = (hdlr->num_devices * hdlr->block_size_per_device);
	return (addr / block_size) * hdlr->block_size_per_device + (addr % hdlr->block_size_per_device);
}

int baio_queue_read(struct baio_handle *hdlr, void *buf, off_t size, off_t offset)
{
	size_t block_size_per_device = hdlr->block_size_per_device;

	unsigned long addr = (unsigned long)buf;

	//int err;
	struct baio_device *dev; 
	struct io_uring_sqe *sqe;
	size_t size_to_read;
	size_t dev_off;
	while (size > 0) {
		int di = get_dev(hdlr, offset);
		dev_off = get_offset(hdlr, offset);
		dev = &hdlr->devices[di];
		size_to_read = block_size_per_device - (offset % block_size_per_device);
		size_to_read = (size_to_read < size) ? size_to_read : size;

		//size_t off = 0;
		size_t left_size = size_to_read;
		while (left_size > 0) {
			sqe = io_uring_get_sqe(&dev->rring);
			if (!sqe) {
				printf("queue_read failed\n");
				return -ENOENT;
			}
			size_t transfer_size = left_size;
			if (transfer_size > MAX_TRANSFER_SIZE) {
				transfer_size = MAX_TRANSFER_SIZE;
			}
			//printf("read from %d from 0x%lx by %lu at %lu\n", di, addr, transfer_size, dev_off);
			io_uring_prep_read(sqe, dev->fd, (void*)addr, transfer_size, dev_off);
			dev->num_reads++;

			addr += transfer_size;
			dev_off += transfer_size;
			left_size -= transfer_size;
		}

		size -= size_to_read;
		offset += size_to_read;
	}

	return 0;
}

int baio_queue_write(struct baio_handle *hdlr, void *buf, off_t size, off_t offset)
{
	//size_t block_size = hdlr->block_size;
	size_t block_size_per_device = hdlr->block_size_per_device;

	unsigned long addr = (unsigned long)buf;


	//int err;
	struct baio_device *dev; 
	struct io_uring_sqe *sqe;
	size_t size_to_write;
	size_t dev_off;

	while (size > 0) {
		int di = get_dev(hdlr, offset);
		dev_off = get_offset(hdlr, offset);
		dev = &hdlr->devices[di];
		size_to_write = block_size_per_device - (offset % block_size_per_device);
		size_to_write = (size_to_write < size) ? size_to_write : size;

		//printf("%lu %lu %lu %lu %lu\n", offset, block_size, block_size_per_device, size_to_write, size);

		size_t left_size = size_to_write;
		while (left_size > 0) {
			sqe = io_uring_get_sqe(&dev->wring);
			if (!sqe) {
				printf("queue_write failed\n");
				return -ENOENT;
			}

			size_t transfer_size = left_size;
			if (transfer_size > MAX_TRANSFER_SIZE) {
				transfer_size = MAX_TRANSFER_SIZE;
			}
			printf("write to %d from 0x%lx by 0x%lx at 0x%lx\n", di, addr, transfer_size, dev_off);
			io_uring_prep_write(sqe, dev->fd, (void*)addr, transfer_size, dev_off);
			//io_uring_sqe_set_data(sqe, addr);
			dev->num_writes++;

			addr += transfer_size;
			dev_off += transfer_size;
			left_size -= transfer_size;
		}
		size -= size_to_write;
		offset += size_to_write;
	}

	return 0;
}

static int baio_queue_submit_device_read(struct baio_device *dev) 
{
	int err = 0;
	int nr = 0;

	unsigned num_reads = dev->num_reads;
	for (unsigned j = 0; j < num_reads; j++) {
		err = io_uring_submit(&dev->rring);
		if (err < 0) {
			printf("submit err: %d\n", err);
			return -EINVAL;
		} 
		nr += err;
	}
	//assert(nr == dev->num_reads);
	return 0;
}

static int baio_queue_submit_device_write(struct baio_device *dev) 
{
	int err = 0;
	int nr = 0;

	unsigned num_writes = dev->num_writes;
	for (unsigned j = 0; j < num_writes; j++) {
		err = io_uring_submit(&dev->wring);
		if (err < 0) {
			printf("submit error: %d\n", err);
			return -EINVAL;
		}
		nr += err;
	}
	//assert(nr == dev->num_writes);
	return 0;
}

static int baio_queue_submit_device(struct baio_device *dev) 
{
	int err;
	err = baio_queue_submit_device_read(dev);
	if (err) {
		return err;
	}
	err = baio_queue_submit_device_write(dev);
	if (err) {
		return err;
	}
	return 0;
}

int baio_queue_submit(struct baio_handle *hdlr) 
{
	int err = 0;
#pragma omp parallel for reduction(+:err)
	for (unsigned i = 0; i < hdlr->num_devices; i++) {
		struct baio_device *dev = &hdlr->devices[i];
//		struct timespec s, e;
//		clock_gettime(CLOCK_MONOTONIC, &s);
		err += baio_queue_submit_device(dev);
//		clock_gettime(CLOCK_MONOTONIC, &e);
	}
	return err;
}

static int baio_wait_all_device_read(struct baio_device *dev)
{
	int err;
	struct io_uring_cqe *cqe;
	
	size_t num_reads;
	num_reads = dev->num_reads;

	for (unsigned j = 0; j < num_reads; j++) {
		err = io_uring_wait_cqe(&dev->rring, &cqe);
		if (err < 0) {
			printf("err %d %d\n", j, err);
			return -EINVAL;
		}
		if (cqe->res < 0) {
			printf("rres %d %d\n", j, cqe->res);
			if (cqe->res == -EAGAIN) {
				num_reads++;
				struct io_uring_sqe *sqe = io_uring_get_sqe(&dev->wring);
				io_uring_prep_read(sqe, dev->fd, io_uring_cqe_get_data(cqe), (1ul << 28), 0);
				err = io_uring_submit(&dev->rring);
				if (err < 0) {
					printf("Submit failed %d\n", err);
					return err;
				}
			} else {
				return -EINVAL;
			}
		} 
		io_uring_cqe_seen(&dev->rring, cqe);
	}
	//dev->num_reads -= num_reads;
	dev->num_reads = 0;
	return 0;
}

static int baio_wait_all_device_write(struct baio_device *dev)
{
	int err;
	struct io_uring_cqe *cqe;
	
	size_t num_writes;
	num_writes = dev->num_writes;

	for (unsigned j = 0; j < num_writes; j++) {
		err = io_uring_wait_cqe(&dev->wring, &cqe);
		if (err < 0) {
			printf("err %d %d\n", j, err);
			return -EINVAL;
		}
		if (cqe->res < 0) {
			printf("wres %d %d\n", j, cqe->res);
			if (cqe->res == -EAGAIN) {
				num_writes++;
				struct io_uring_sqe *sqe = io_uring_get_sqe(&dev->wring);
				io_uring_prep_write(sqe, dev->fd,
				io_uring_cqe_get_data(cqe), (1ul << 28), 0);
				err = io_uring_submit(&dev->wring);
				if (err < 0) {
					printf("Submit failed %d\n", err);
					return err;
				}
			} else {
				return -EINVAL;
			}
		} else {
			//printf("res %u\n", cqe->res);
		}
		io_uring_cqe_seen(&dev->wring, cqe);
	}

	dev->num_writes = 0;
	//dev->num_writes -= num_writes;

	return 0;
}

static int baio_wait_all_device(struct baio_device *dev)
{
	int err;
	err = baio_wait_all_device_read(dev);
	if (err) {
		return err;
	}
	err = baio_wait_all_device_write(dev);
	if (err)
		return err;

	return 0;
}

int baio_wait_all(struct baio_handle *hdlr) 
{
	int err = 0;
#pragma omp parallel for reduction(+:err)
	for (unsigned i = 0; i < hdlr->num_devices; i++) {
//		struct timespec s, e;
//		clock_gettime(CLOCK_MONOTONIC, &s);

		struct baio_device *dev = &hdlr->devices[i];
		err += baio_wait_all_device(dev);
		if (err) {
			printf("Wait failed %d\n", i);
		} 
//
//		clock_gettime(CLOCK_MONOTONIC, &e);
//		printf("dev %d, wait time: %lfms\n", i, 
//				(e.tv_sec-s.tv_sec) * 1000. + (e.tv_nsec-s.tv_nsec) / 1000000.);
	}
	return err;
}

static int baio_fsync_device(struct baio_device *dev) {
	struct io_uring_sqe *sqe;
	int err;

	sqe = io_uring_get_sqe(&dev->rring);
	io_uring_prep_fsync(sqe, dev->fd, 0);
	err = io_uring_submit(&dev->rring);
	if (err < 0) {
		return err;
	}

	sqe = io_uring_get_sqe(&dev->wring);
	io_uring_prep_fsync(sqe, dev->fd, 0);
	err = io_uring_submit(&dev->wring);
	if (err < 0) {
		return err;
	}
	return 0;
}

int baio_fsync(struct baio_handle *hdlr) {
	int err = 0;
#pragma omp parallel for
	for (unsigned i = 0; i < hdlr->num_devices; i++) {
		struct baio_device *dev = &hdlr->devices[i];
		err = baio_fsync_device(dev);
		if (err < 0) {
			printf("baio_fsync_device failed: %d\n", i);
		}
	}
	return err;
}
