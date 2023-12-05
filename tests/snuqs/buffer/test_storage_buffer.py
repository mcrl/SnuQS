import unittest
from snuqs import StorageBuffer

import random


class StorageBufferTest(unittest.TestCase):
    def test_buffer_get_set(self):
        count = 2**30
        buffer = StorageBuffer(count)

        random.seed(-1)
        rands = {random.randint(0, count-1): random.random() for _ in range(50)}
        for k, v in rands.items():
            buffer[k] = v

        for k, v in rands.items():
            self.assertEqual(buffer[k], v)

        self.assertRaises(NotImplementedError, lambda: buffer[0:5])


if __name__ == '__main__':
    unittest.main()
