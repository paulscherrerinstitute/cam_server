import logging
import unittest
import numpy
from scipy import signal

from cam_server.pipeline.data_processing.functions import gauss_fit, calculate_slices, linear_fit, find_index


class FunctionsTest(unittest.TestCase):
    def test_gauss_fit(self):
        size = 101
        standart_deviation_set = 6.3
        center_set = 5

        data = signal.gaussian(size, standart_deviation_set)
        axis = numpy.array(range(size)).astype('f')

        axis += center_set

        gauss_function, offset, amplitude, center, standard_deviation, _, _ = gauss_fit(data, axis)

        logging.info("Retrieved standard deviation: %f" % standard_deviation)
        logging.info("Retrieved center: %f" % center)

        self.assertAlmostEqual(standart_deviation_set, standard_deviation, delta=0.0001)
        self.assertAlmostEqual(int(size / 2) + center_set, center, delta=0.0001)

    def test_calculate_slices(self):
        size = 1000
        center = 300
        standard_deviation = 12

        axis = numpy.array(range(size)).astype('f')

        indexes, n_indices_half_slice = calculate_slices(axis, center, standard_deviation,
                                                         scaling=2, number_of_slices=11)
        # we expect 11 slices - as the middle slice is half half in the center, therefore 12 indexes are needed
        self.assertEqual(len(indexes), 12)

        # size_slice = scaling * standard_deviation / number_of_slices
        # 2*12/11 /2  = 1.0909...  - We expect a half slice size to be 1
        self.assertEqual(n_indices_half_slice, 1)

        # Indexes should look something like this [189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211]
        self.assertEqual(indexes, [289, 291, 293, 295, 297, 299, 301, 303, 305, 307, 309, 311])

        # Use of different axis, less slices

        # move the axis values to the left
        axis = numpy.array(range(100, 100 + size)).astype('f')

        indexes, n_indices_half_slice = calculate_slices(axis, center, standard_deviation,
                                                         scaling=2, number_of_slices=5)
        # we expect 5 slices - therefore 6 indexes are needed
        self.assertEqual(len(indexes), 6)

        # size_slice = scaling * standard_deviation / number_of_slices
        # 2*12/5 /2  = 2.4  -  We expect a half slice size to be 2
        self.assertEqual(n_indices_half_slice, 2)

        # Indexes should look something like this [189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211]
        self.assertEqual(indexes, [190, 194, 198, 202, 206, 210])

        standard_deviation = 3
        indexes, n_indices_half_slice = calculate_slices(axis, center, standard_deviation,
                                                         scaling=2, number_of_slices=5)
        # 2*6/5 /2 = 0.6 - We expect a half slice size to be at least 1
        self.assertEqual(n_indices_half_slice, 1)

        size = 10
        center = 1
        standard_deviation = 12

        axis = numpy.array(range(size)).astype('f')

        indexes, n_indices_half_slice = calculate_slices(axis, center, standard_deviation,
                                                         scaling=2, number_of_slices=11)

        # 2*12/11 /2  = 1.0909...  - We expect a half slice size to be 1
        self.assertEqual(n_indices_half_slice, 1)
        # Only expecting 1 slice as the other indexes would be out of range
        self.assertEqual(indexes, [0, 2])
        print(indexes)

    def test_find_index(self):
        size = 300
        axis = numpy.array(range(size)).astype('f')

        # cam.functions.find_index(axis, 4.5)
        self.assertEqual(find_index(axis, 200), 200)
        self.assertEqual(find_index(axis, 200.5), 200)
        self.assertEqual(find_index(axis, 199.9), 199)
        self.assertEqual(find_index(axis, 200.9), 200)

        self.assertEqual(find_index(axis, size), size - 1)  # needs to be last index
        self.assertEqual(find_index(axis, size + 10), size - 1)  # needs to be last index

        self.assertEqual(find_index(axis, 0.1), 0)
        self.assertEqual(find_index(axis, -1), 0)

        # Reverse axis
        axis = axis[::-1]
        self.assertEqual(find_index(axis, 200.9), 99)
        self.assertEqual(find_index(axis, 0), 299)
        self.assertEqual(find_index(axis, -1), 299)
        self.assertEqual(find_index(axis, 300), 0)

    def test_get_slice_data(self):
        pass

    def test_linear_fit(self):
        x = numpy.array([0., 1., 2., 3., 4.])
        y = numpy.array([0., 1., 2., 3., 4.])
        slope, offset = linear_fit(x, y)
        logging.info("slope: %f offset: %f" % (slope, offset))
        self.assertAlmostEqual(slope, 1)
        self.assertAlmostEqual(offset, 0)

        x = numpy.array([0., 1., 2., 3., 4.])
        y = numpy.array([1., 2., 3., 4., 5.])
        slope, offset = linear_fit(x, y)
        logging.info("slope: %f offset: %f" % (slope, offset))
        self.assertAlmostEqual(slope, 1)
        self.assertAlmostEqual(offset, 1)

        x = numpy.array([0., 1., 2., 3., 4.])
        y = numpy.array([-0.1, -1.1, -2.1, -3.1, -4.1])
        slope, offset = linear_fit(x, y)
        logging.info("slope: %f offset: %f" % (slope, offset))
        self.assertAlmostEqual(slope, -1)
        self.assertAlmostEqual(offset, -0.1)

        x = numpy.array([0., 1., 2., 3., 4.])
        y = numpy.array([0., 0.1, 0.2, 0.3, 0.4])
        slope, offset = linear_fit(x, y)
        logging.info("xx slope: %f offset: %f" % (slope, offset))
        self.assertAlmostEqual(slope, 0.1)
        self.assertAlmostEqual(offset, 0.0)

        x = [605.0, 609.0, 613.0, 617.0, 621.0, 625.0, 629.0, 633.0, 637.0, 641.0, 645.0]
        y = [476.65531378025469, 476.55426997267142, 476.60364191555584, 476.54863572007542, 476.56270299019877,
             476.54710034527858, 476.54197609352912, 476.57245390887124, 476.50808807086082, 476.57282667093847,
             476.56745922293777]
        slope, offset = linear_fit(x, y)
        logging.info("xx slope: %f offset: %f" % (slope, offset))


if __name__ == '__main__':
    unittest.main()
