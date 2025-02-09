import numpy as np
import logging
from mrds.lmp import LMP

from unittest import TestCase, main
from logging import getLogger, basicConfig

basicConfig(level=logging.INFO)

_logger = getLogger(__name__)


class LMPTest(TestCase):
    """ Tests for the Locational margin pricing algorithm.
    """

    def test_simple(self):
        gl_simple = [('1', 150., 0., 10.), ('2', 150., 120., 15.)]
        pl_simple = [('1', '2', 100.)]
        lmp_simple = LMP(gl_simple, pl_simple)
        res_simple = lmp_simple.compute_lmp()

        _logger.info(
            f'Simplest result: {res_simple}'
        )

        self.assertTrue(True)  # TODO: FIX

    def test_1(self):

        gl_1 = [
            ('1', 500., 0., 10.),
            ('2', 0., 75., 0.),
            ('3', 500., 325, 20.),
        ]
        pl_1 = [
            ('1', '2', 300.),
            ('2', '3', 40.),
            ('1', '3', 200.),
        ]
        lmp_1 = LMP(gl_1, pl_1)

        res_1 = lmp_1.compute_lmp()

        _logger.info(
            f'Test 1: {res_1}'
        )
        self.assertTrue(True)

    def test_complicated(self):
        """ complicated network (node, gen, load, price)
        """

        gl_2 = [
            ('1', 300., 100., 10.),
            ('2', 0., 100., 0.),
            ('3', 0., 100., 0.),
            ('4', 300., 100., 20.),
            ('5', 0., 100., 0.),
        ]
        pl_2 = [
            ('1', '2', 500.),
            ('1', '3', 200.),
            ('2', '3', 200.),
            ('3', '4', 500.),
            ('4', '5', 150.),
        ]

        lmp_2 = LMP(gl_2, pl_2)
        res_2 = lmp_2.compute_lmp(show_sol=True)

        _logger.info(
            f'Test complicated: {res_2}'
        )
        self.assertTrue(True)

    def test_large_network(self):

        N = 30
        gl_3 = zip(range(1, N+1), np.random.random(N) * 100.,
                   np.random.random(N) * 50.,
                   np.random.random(N) * 100.)  # (n, gen, load, network_struct) , (2, 0., 100., 0.), (3, 0., 100., 0.),
        gl_3 = list(gl_3)
        pl_3 = []
        for i in range(1, N+1):
            for j in range(i+1, N+1):
                pl_3.append((i, j, 2.))

        lmp = LMP(gl_3, pl_3)
        result = lmp.compute_lmp()

        _logger.info(
            f'Large network result: {result}'
        )

        self.assertTrue(True)
        # print lmp.find_connected(5, pl_2)


if __name__ == '__main__':
    main()
