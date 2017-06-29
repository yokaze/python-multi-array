import multi_array as ma
import numpy as np
import unittest

class TestMultiArray(unittest.TestCase):
    def test_attributes(self):
        self.assertEqual(ma.bool8, np.bool8)
        self.assertEqual(ma.uint8, np.uint8)
        self.assertEqual(ma.uint16, np.uint16)
        self.assertEqual(ma.uint32, np.uint32)
        self.assertEqual(ma.uint64, np.uint64)
        self.assertEqual(ma.int8, np.int8)
        self.assertEqual(ma.int16, np.int16)
        self.assertEqual(ma.int32, np.int32)
        self.assertEqual(ma.int64, np.int64)
        self.assertEqual(ma.float32, np.float32)
        self.assertEqual(ma.float64, np.float64)

    def test_make_with_types(self):
        self.assertEqual(ma.make(10, np.float32).shape(), (10,))
        self.assertEqual(ma.make([1, 2, 3, 4], np.float32).shape(), (1, 2, 3, 4))
        self.assertEqual(ma.make((1, 2, 3, 4), np.float32).shape(), (1, 2, 3, 4))
        self.assertEqual(ma.make(np.array([1, 2, 3, 4]), np.float32).shape(), (1, 2, 3, 4))

    def test_make_all(self):
        for iiter in range(100):
            dtypes = [
                np.bool8,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.float32,
                np.float64
            ]
            ndim = np.int32(np.random.rand() * 8 + 1)       # 1 to 8
            shape = np.int32(np.random.rand(ndim) * 4 + 1)  # 1 to 4
            dtype = dtypes[int(np.random.rand() * len(dtypes))]
            nelem = np.array(shape).prod()

            x = ma.make(shape, dtype)
            self.assertEqual(x.element(), dtype)
            self.assertEqual(x.shape(), tuple(shape))
            self.assertEqual(x.num_dimensions(), ndim)
            self.assertEqual(x.num_elements(), nelem)
            self.assertEqual(abs(x.get()).max(), 0)

            y = (np.random.rand(nelem) * 10 - 5).reshape(shape)
            x.set(y)
            self.assertTrue((x.get() == dtype(y)).all())

            ix = np.int32(shape * np.random.rand(ndim))
            self.assertEqual(x[ix], dtype(y[tuple(ix)]))

if (__name__ == '__main__'):
    unittest.main()
