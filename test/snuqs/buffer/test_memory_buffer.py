import unittest
from snuqs import MemoryBuffer

import random


class MemoryBufferTest(unittest.TestCase):
    def test_buffer_get_set(self):
        count = 2**20
        buffer = MemoryBuffer(count)

        random.seed(-1)
        rands = {random.randint(0, count-1): random.random()+random.random()*1j for _ in range(50)}
        for k, v in rands.items():
            buffer[k] = v

        for k, v in rands.items():
            self.assertEqual(buffer[k], v)

        self.assertRaises(NotImplementedError, lambda: buffer[0:5])


if __name__ == '__main__':
    unittest.main()
