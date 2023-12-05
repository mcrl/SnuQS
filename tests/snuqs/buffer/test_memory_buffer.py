import unittest
from snuqs import MemoryBuffer

import random


class MemoryBufferTest(unittest.TestCase):
    def test_buffer_get_set(self):
        count = 2**20
        buffer = MemoryBuffer(8*count)

        rands = {random.randint(0, count-1): random.random() for _ in range(min(120, count))}
        for k, v in rands.items():
            buffer[k] = v

        for k, v in rands.items():
            self.assertEqual(buffer[k], v)

        self.assertRaises(NotImplementedError, lambda: buffer[0:5])


if __name__ == '__main__':
    unittest.main()
