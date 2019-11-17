###############################################################################
# CUED PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015, 2016
# Cambridge University Engineering Department Dialogue Systems Group
#
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################

'''
LSPILib.py - LSPI algorithm
=============================================

Copyright TSI-TUC 2017

This module encapsulates classes and functionality of various basis functions used in LSPI.

************************

'''

__author__ = "tsi_tuc_group"

import numpy as np


class BlockBasis(object):
    """Basis functions over (s,a) which activates the appropriate block of state feature for each action.
    All other blocks have 0 values.

    Parameters
    ----------
    """
    def __init__(self, domainString, num_actions, state_dim):
        """Initialize BlockBasis.
        """
        
        self.num_actions = num_actions
        self.state_dim = state_dim


    def size(self):
        r"""Return the vector size of the basis function.
        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        # return self.state_dim * self.num_actions   
        return self.state_dim * self.num_actions + 1   # Add 1 for a single bias

    def evaluate(self, belief, action):
        r"""Return a :math:`\phi` vector that has a single active block of state features.
        """
        phi = np.zeros(self.size())
        if action.actid > -1:
            base = action.actid * self.state_dim
            state_vector = belief.getStateVector() #encoded_state
            phi[base:base + self.state_dim] = state_vector 
        else:
            phi[-1] = 1.0 # single bias

        return phi

  

#END OF FILE
