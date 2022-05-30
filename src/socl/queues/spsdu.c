# include "spsdu.h"
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>

int spsdu_init(struct spsdu *spsdu)
{
	atomic_init(&spsdu->head, 0);
	atomic_init(&spsdu->tail, 0);

	return 0;
}

void spsdu_deinit(struct spsdu *spsdu) 
{
}

int spsdu_enqueue(struct spsdu *spsdu, void *ent)
{
	struct spsdu_node *node_ = malloc(sizeof(struct spsdu_node));
	node_->ent = ent;
	atomic_init(&node_->next, 0);
	uintptr_t node = (uintptr_t)node_;
	

	uintptr_t nullptr;
	atomic_init(&nullptr, 0);


	while (1) {
		struct spsdu_node *last = (struct spsdu_node*) spsdu->tail;
		if (last != (struct spsdu_node*)spsdu->tail) continue;
		if (last == NULL) {
			if (atomic_compare_exchange_weak(&spsdu->tail, &last, (uintptr_t)node)) {
				atomic_init(&spsdu->head, node);
				return 0;
			} else {
				continue;
			}
		} 
		struct spsdu_node *next = (struct spsdu_node*) last->next;
		if (next == NULL) {
			if (atomic_compare_exchange_weak(&last->next, &nullptr, (uintptr_t)node)) {
				atomic_compare_exchange_weak(&spsdu->tail, &last, (uintptr_t)node);
				return 0;
			}
		} else {
			atomic_compare_exchange_weak(&spsdu->tail, &last, (uintptr_t)next);
		}
	}

	return 0;
}

void* spsdu_dequeue(struct spsdu *spsdu)
{
	while (1) {
		struct spsdu_node *first = (struct spsdu_node*)spsdu->head;
		if ((struct spdu_node*)first == NULL) {
			return NULL;
		}
		if (!atomic_compare_exchange_weak(&spsdu->head, &first, first->next)) 
			continue;
		void *ent = (struct spsdu_node*)(first)->ent;
		free(first);
		return ent;
	} 
	return NULL;
}
