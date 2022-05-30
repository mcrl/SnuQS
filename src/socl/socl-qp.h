#pragma once


struct socl_request_queue {
};

struct socl_completion_queue {
};

struct socl_queue_pair {
	struct socl_request_queue req_q;
	struct socl_completion_queue comp_q;
};

