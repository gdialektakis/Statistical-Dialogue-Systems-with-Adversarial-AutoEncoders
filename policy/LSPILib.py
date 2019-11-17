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
import os.path
import pickle as pkl
import PolicyUtils
from utils import ContextLogger
logger = ContextLogger.getLogger('')
from sklearn.externals import joblib

class ActionEmbeddingBasis(object):
    """Basis functions over (s,a) which concatenates a hotword action vector to the state feature.
    All other blocks have 0 values.

    Parameters
    ----------
    """
    def __init__(self, domainString, actions, numStateFeatures, embedLen, EmbedFile):
        """Initialize ActionHotwordBasis.
        """
        self.actions = actions
        self.embedLen = embedLen
        self.EmbedFile = EmbedFile
        self.numActions = len(actions.action_names)
        self.numStateFeatures = numStateFeatures
        randseed = 5 #If you want to replicate the pseudorandom embedding
        if os.path.isfile(self.EmbedFile):
            self.LSPIActEmbedLoad()
            #print("Loaded Action Embeddings from:", self.EmbedFile)
        else:
            np.random.seed(randseed)
            self.actembeddings = np.random.uniform(0.0, 1.0, size=[self.numActions, self.embedLen])
            self.LSPIActEmbedSave()
            #print("Saved Action Embeddings to:", self.EmbedFile)


        # self.size = self.numStateFeatures * self.numActions    # for no bias
        #self.size = self.numStateFeatures * self.numActions + 1   # for a single bias
        # self.size = (self.numStateFeatures+1) * self.numActions    # for one bias per block
        print('ActionEmbeddingBasis ', 'self.numActions ', self.numActions, 'self.numStateFeatures ', self.numStateFeatures, ' embedLen ', embedLen)            

    def LSPIActEmbedSave(self):
        '''
        Saves the LSPI policy.
        '''
        print("Saving Action Embeddings to:", self.EmbedFile)
        PolicyUtils.checkDirExistsAndMake(self.EmbedFile)
        pkl_file = open(self.EmbedFile, 'wb')
        pkl.dump(self.actembeddings, pkl_file)
        pkl_file.close()
        np.savetxt(self.EmbedFile + '.acttext', self.actembeddings, delimiter=',', fmt='%1.4e')
        return

    def LSPIActEmbedLoad(self):
        '''
        Loads the LSPI policy.
        '''
        print("Loading Action Embeddings from:", self.EmbedFile)
        pkl_file = open(self.EmbedFile, 'rb')
        self.actembeddings = pkl.load(pkl_file)
        #print('\n\t LOADED-WEIGHTS:\n')
        #print(np.matrix(self.weights.reshape(17,16)))
        pkl_file.close()
        return

    def set_size(self, num_features):
        r"""Return the vector size of the basis function.
        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        if self.numStateFeatures == 0:
            self.numStateFeatures = num_features
        if self.numStateFeatures != num_features:
            self.numStateFeatures = num_features

    def size(self):
        r"""Return the vector size of the basis function.
        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        return (self.numStateFeatures + self.embedLen)   # Add 1 for a single bias

    def evaluate(self, belief, action):
        r"""Return a :math:`\phi` vector that has a single active block of state features.
        """
        phi_len = self.size()
        phi = np.zeros(phi_len)
        phi[0:self.embedLen] = self.actembeddings[action.actid]
        stateFeatures = belief.beliefStateVec2
        phi[self.embedLen:phi_len] = stateFeatures
        return phi


class ActionHotwordBasis(object):
    """Basis functions over (s,a) which concatenates a hotword action vector to the state feature.
    All other blocks have 0 values.

    Parameters
    ----------
    """
    def __init__(self, domainString, actions, numStateFeatures):
        """Initialize ActionHotwordBasis.
        """
        self.actions = actions
        self.numActions = len(actions.action_names)
        self.numStateFeatures = numStateFeatures
        # self.size = self.numStateFeatures * self.numActions    # for no bias
        #self.size = self.numStateFeatures * self.numActions + 1   # for a single bias
        # self.size = (self.numStateFeatures+1) * self.numActions    # for one bias per block
        print('ActionHotwordBasis ', 'self.numActions ', self.numActions, 'self.numStateFeatures ', self.numStateFeatures) 

    def set_size(self, num_features):
        r"""Return the vector size of the basis function.
        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        if self.numStateFeatures == 0:
            self.numStateFeatures = num_features
        if self.numStateFeatures != num_features:
            self.numStateFeatures = num_features

    def size(self):
        r"""Return the vector size of the basis function.
        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        return (self.numStateFeatures + self.numActions)   # Add 1 for a single bias

    def evaluate(self, belief, action):
        r"""Return a :math:`\phi` vector that has a single active block of state features.
        """
        actionvec = np.zeros(self.numActions)
        actionvec[action.actid] = 1
        phi_len = self.size()
        phi = np.zeros(phi_len)
        phi[0:self.numActions] = actionvec
        stateFeatures = belief.beliefStateVec2
        phi[self.numActions:phi_len] = stateFeatures
        return phi

class BlockBasis(object):
    """Basis functions over (s,a) which activates the appropriate block of state feature for each action.
    All other blocks have 0 values.

    Parameters
    ----------
    """
    def __init__(self, domainString, actions, numStateFeatures):
        """Initialize BlockBasis.
        """
        #self.domainString = domainString
        #if (self.domainString == "CamRestaurants"):
        #    self.numStateFeatures = 34
        #elif (self.domainString == "SFRestaurants"):
        #    self.numStateFeatures = 50
        #elif (self.domainString == "CamHotels"):
        #    self.numStateFeatures = 50
        #elif (self.domainString == "SFHotels"):
        #    self.numStateFeatures = 71
        #elif (self.domainString == "Laptops6"):
        #    self.numStateFeatures = 53
        #elif (self.domainString == "Laptops11"):
        #    self.numStateFeatures = 64
        #elif (self.domainString == "TV"):
        #    self.numStateFeatures = 66
        #elif (self.domainString == "BCM"):
        #    self.numStateFeatures = 34
        #else:
        #    self.numStateFeatures = 0
        #self.numStateFeatures = 16  # for DIP or DIPlinear
        #self.numStateFeatures = 32  # for Autoencoder Features

        self.actions = actions
        self.numActions = len(actions.action_names)
        self.numStateFeatures = numStateFeatures
        print('BlockBasis', 'self.numActions ', self.numActions, 'self.numStateFeatures ', self.numStateFeatures) 
        # self.size = self.numStateFeatures * self.numActions    # for no bias
        #self.size = self.numStateFeatures * self.numActions + 1   # for a single bias
        # self.size = (self.numStateFeatures+1) * self.numActions    # for one bias per block

    def set_size(self, num_features):
        r"""Return the vector size of the basis function.
        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        if self.numStateFeatures == 0:
            self.numStateFeatures = num_features
        if self.numStateFeatures != num_features:
            self.numStateFeatures = num_features

    def size(self):
        r"""Return the vector size of the basis function.
        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        return self.numStateFeatures * self.numActions   # Add 1 for a single bias

    def evaluate(self, belief, action):
        r"""Return a :math:`\phi` vector that has a single active block of state features.
        """
        phi = np.zeros(self.size())
        #base = self.actions.action_names.index(action) * self.numStateFeatures
        base = action.actid * self.numStateFeatures
        #stateFeatures = self.getStateDIP(belief)
        stateFeatures = belief.beliefStateVec2 #encoded_state
        phi[base:base + self.numStateFeatures] = stateFeatures
        #VVV phi[self.size()-1]=1   # Add a single constant term (bias)
        # phi[base + self.numStateFeatures] = 1  # Add a constant term (bias) to the active block
        return phi

    def getStateONE(self, belief):
        stateONE = np.zeros(1)
        stateONE[0] = 1
        return stateONE

    def getStateDIP(self, belief):
        stateDIP = np.zeros(self.numStateFeatures)
        pos = 0
        ind = 0
        for b in sorted(belief["beliefs"].keys()):
            if b not in {"discourseAct", "method", "requested"}:
                if (("none" in belief["beliefs"][b] and belief["beliefs"][b]["none"]>0.5)
                    or ("**NONE**" in belief["beliefs"][b] and belief["beliefs"][b]["**NONE**"]>0.5)):
                    pos += 0
                else:
                    pos += 2**ind
                ind += 1
        stateDIP[pos] = 1.0
        return stateDIP

    def getStateDIPlinear(self, belief):
        stateDIPlinear = np.zeros(self.numStateFeatures)
        pos = 0
        for b in sorted(belief["beliefs"].keys()):
            if b not in {"discourseAct", "method", "requested"}:
                if (("none" in belief["beliefs"][b] and belief["beliefs"][b]["none"]<0.5)): 
                    stateDIPlinear[pos] = 1.0 - belief["beliefs"][b]["none"]
                elif (("**NONE**" in belief["beliefs"][b] and belief["beliefs"][b]["**NONE**"]<0.5)):
                    stateDIPlinear[pos] = 1.0 - belief["beliefs"][b]["**NONE**"]
                else:
                    stateDIPlinear[pos] = 0.0
                pos += 1
        return stateDIPlinear

    def getStateFeatures(self, belief):
        stateFeatures = np.zeros(0)
        #stateFeatures = np.append(stateFeatures, np.array(1))  #1 - constant term

        if (self.domainString == "CamRestaurants"): # total:241  (current: 34)
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["discourseAct"].values()))  #7
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["method"].values()))  #6
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["requested"].values()))  #9
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["area"].values()))  #7
            # stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["food"].values()))  #93
            # if ("dontcare" in belief["beliefs"]["name"]):
            #     longer = np.array(belief["beliefs"]["name"].values())
            #     truncated = np.append(longer[0:1],longer[2:])  # extraneous dontcare entry comes 2nd in order - remove it
            #     stateFeatures = np.append(stateFeatures, truncated)
            # else:
            #     stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["name"].values()))  #114
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["pricerange"].values()))  #5

        if (self.domainString == "SFRestaurants"): # total: 608  (current: 50)
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["discourseAct"].values()))  #7
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["method"].values()))  #6
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["requested"].values()))  #11
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["allowedforkids"].values()))  #4
            #stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["area"].values()))  #157
            #stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["food"].values()))  #61
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["goodformeal"].values()))  #6
            #stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["name"].values()))  #243
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["near"].values()))  #11
            #stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["price"].values()))  #97
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["pricerange"].values()))  #5

        if (self.domainString == "CamHotels"):  # total: 84  (current: 50)
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["discourseAct"].values()))  #7
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["method"].values()))  #6
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["requested"].values()))  #11
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["area"].values()))  #7
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["hasparking"].values()))  #4
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["kind"].values()))  #4
            #stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["name"].values()))  #34
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["pricerange"].values()))  #5
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["stars"].values()))  #6

        if (self.domainString == "SFHotels"):  # total: 411  (current: 71)
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["discourseAct"].values()))  #7
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["method"].values()))  #6
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["requested"].values()))  #10
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["acceptscreditcards"].values()))  #4
            #stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["area"].values()))  #157
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["dogsallowed"].values()))  #4
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["hasinternet"].values()))  #4
            #stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["name"].values()))  #183
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["near"].values()))  #30
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["pricerange"].values()))  #6

        if (self.domainString == "Laptops6"):  # total: 177  (current: 53)
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["discourseAct"].values()))  #7
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["method"].values()))  #6
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["requested"].values()))  #10
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["batteryrating"].values()))  #5
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["driverange"].values()))  #5
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["family"].values()))  #6
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["isforbusinesscomputing"].values()))  #4
            #stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["name"].values()))  #124
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["pricerange"].values()))  #5
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["weightrange"].values()))  #5

        if (self.domainString == "Laptops11"):  # total: 230  (current: 64)
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["discourseAct"].values()))  #7
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["method"].values()))  #6
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["requested"].values()))  #21
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["batteryrating"].values()))  #5
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["driverange"].values()))  #5
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["family"].values()))  #6
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["isforbusinesscomputing"].values()))  #4
            #stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["name"].values()))  #124
            #stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["platform"].values()))  #7
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["pricerange"].values()))  #5
            #stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["processorclass"].values()))  #12
            #stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["sysmemory"].values()))  #9
            #stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["utility"].values()))  #9
            #stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["warranty"].values()))  #5
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["weightrange"].values()))  #5

        if (self.domainString == "TV"):  # total: 101  (current: 66)
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["discourseAct"].values()))  #7
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["method"].values()))  #6
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["requested"].values()))  #14
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["eco"].values()))  #7
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["hdmi"].values()))  #6
            #stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["name"].values()))  #95
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["pricerange"].values()))  #3
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["screensizerange"].values()))  #5
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["series"].values()))  #14
            stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"]["usb"].values()))  #4

        if (self.domainString == "BCM"):
            for b in belief["beliefs"]:   # B1:255 (!name:154) B2:655 (!name:154) B3:1155 (!name:154) B4:355 (!name:254) B5:395 (!name:294)
            #for b in ["discourseAct","method","requested"]:  # B1: 34
                if b != "name":
                    stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"][b].values()))
        #if len(stateFeatures)>96:
        #    print("Bingo!")

        return stateFeatures


class PCABasis(object):
    """PCA-Based basis functions over (s,a) 

    Parameters
    ----------
    """
    def __init__(self, pca_file, actions):
        """Initialize PCABasis.
        """
        self.actions = actions
        self.numActions = len(actions.action_names)
        self.pca_file = pca_file
        if pca_file!=None:
            self.pca = joblib.load(self.pca_file)
            self.size = self.pca.n_components + 1 # +1 for a single constant term (bias)
            self.size = self.pca.n_components + self.numActions  # for multiple constant terms (bias)

    def evaluate(self, belief, action):
        r"""Return a :math:`\phi` vector in the reduced pca space including state and action features.
        """
        phi = self.getHighDimStateAction(belief, action)
        pcaphi = (self.pca.transform(phi.reshape(1, -1)))[0]
        #bias = np.array(1)  # Add a constant term (bias)
        bias = np.zeros(self.numActions)
        bias[self.actions.action_names.index(action)] = 1
        pcaphi = np.append(pcaphi, bias)
        return pcaphi

    def getHighDimStateAction(self, belief, action):
        stateFeatures = self.getStateFeatures(belief)
        numStateFeatures = len(stateFeatures)
        phi = np.zeros(numStateFeatures * self.numActions)
        base = self.actions.action_names.index(action) * numStateFeatures
        phi[base:base + numStateFeatures] = stateFeatures
        return phi

    def getStateFeatures(self, belief):
        stateFeatures = np.zeros(0)
        for b in sorted(belief["beliefs"].keys()):
            if (b == "name") and ("dontcare" in belief["beliefs"]["name"]):
                longer = np.array(belief["beliefs"][b].values())
                truncated = np.append(longer[0:1],longer[2:])  # extraneous dontcare entry comes 2nd in order - remove it
                stateFeatures = np.append(stateFeatures, truncated)
            else:
                stateFeatures = np.append(stateFeatures, np.array(belief["beliefs"][b].values()))
        return stateFeatures

#END OF FILE
