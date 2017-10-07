# -*- coding: utf-8 -*-
"""
test suite for monte carlo call overlap script

Created on Wed Oct 04 16:08:56 2017

@author: tbeleyur
"""

import sys
import unittest
import numpy as np
from testfixtures import LogCapture

from MC_echo_call_overlap import *

module_path = 'C:\\Users\\tbeleyur\\Google Drive\\Holger Goerlitz- IMPRS\\PHD_2015\\projects and analyses\\2016_jamming response modelling\\analytical_modelling\\'
sys.path.append(module_path)

class TestMCcalloverlap(unittest.TestCase):

    def setUp(self):
        self.echo_range = [10,30]

        self.overlap_call = [ [10,30] ]

        self.nonoverlap_call = [ [35,45] ]

        self.overlap_subcall = [ [10,20] ]

        self.short_call = [ [10]  ]

        # of the call
        self.leftedge_touching = [[31,35]]

        self.rightedge_touching = [[5,9]]

        # fwd and backward maskers :
        self.forward_masker = [8,9]
        self.backward_masker = [31,35]
        self.fwd_bkwd_masking = [self.forward_masker,self.backward_masker]


    def test_simultaneousmasking(self):
        '''
        checks for a set of cases where simultaneous masking occurs, and
        doesn't
        '''

        self.assertEqual( check_simultaneous_masking(self.echo_range, self.overlap_call )  , 1 )

        self.assertEqual( check_simultaneous_masking(self.echo_range, self.nonoverlap_call )  , 0 )

        self.assertEqual( check_simultaneous_masking(self.echo_range, self.overlap_subcall )  , 0 )

        self.assertEqual( check_simultaneous_masking(self.echo_range, self.short_call )  , 0 )

        self.assertEqual( check_simultaneous_masking(self.echo_range, self.rightedge_touching )  , 0 )

        self.assertEqual( check_simultaneous_masking(self.echo_range, self.leftedge_touching )  , 0 )


    def test_check_masking(self):


        # repeat some of the same tests without any masking windows including in argument

        self.assertEqual( check_masking(self.echo_range, self.overlap_call )  , 1 )

        self.assertEqual( check_masking(self.echo_range, self.nonoverlap_call )  , 0 )

        self.assertEqual( check_masking(self.echo_range, self.overlap_subcall )  , 1 )


        with self.assertRaises(IndexError):
            check_masking(self.echo_range, self.short_call )

        # now test the function with non-default masking window arguments included

        self.assertEqual( check_masking(self.echo_range, self.rightedge_touching, masking_region=[1,1] )  , 1 )

        self.assertEqual( check_masking(self.echo_range, self.leftedge_touching, masking_region=[1,0] )  , 0 )

        self.assertEqual( check_masking(self.echo_range, self.rightedge_touching, masking_region=[0,1] )  , 0 )


        # makse sure that check_masking always only give 1/0 to indicate masking or not !
        self.assertEqual( check_masking(self.echo_range,self.fwd_bkwd_masking , masking_region=[2,2] ),1)

        self.assertEqual( check_masking(self.echo_range,[self.rightedge_touching[0],self.backward_masker] , masking_region=[0,2] ),1)




















if __name__ == '__main__' :
    unittest.main()