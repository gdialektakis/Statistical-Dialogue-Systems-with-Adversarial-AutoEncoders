###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015 - 2017
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
LSPIPolicy.py - Least-Squares Policy Iteration (LSPI) policy
============================================

Copyright TSI-TUC 2017

   
**Relevant Config variables** [Default values]::

    [lspipolicy]
    phitype = block

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`policy.LSPILib` |.|
    import :mod:`policy.Policy` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************

'''
__author__ = "tsi_tuc_group"

import math
import copy
import numpy as np
import json
import os
import sys
import time
import pickle
import PolicyUtils
from itertools import starmap, izip, combinations, product
from operator import mul    #,sub
from scipy.stats import entropy
from collections import OrderedDict


from Policy import Policy, Action, State, TerminalAction, TerminalState
from policy import PolicyCommittee, SummaryUtils
from LSPILib2 import BlockBasis
from ontology import Ontology
from utils import Settings, ContextLogger

# Fotis
# Modifications for autoencoder
from policy.flatten_state import flatten_belief
import ontology.FlatOntologyManager as FlatOnt
# End of modifications

logger = ContextLogger.getLogger('')

class LSPIPolicy(Policy,PolicyCommittee.CommitteeMember):
    '''
    An implementation of the dialogue policy based on the LSPI algorithm to optimise actions.
    
    The class implements the public interfaces from :class:`~Policy.Policy` and :class:`~PolicyCommittee.CommitteeMember`.
    '''
    def __init__(self, domainString, learning, sharedParams=None):
        super(LSPIPolicy, self).__init__(domainString, learning) 

        # DEFAULTS:
        self.discount = 0.99
        self.inpolicyfile = ''
        self.outpolicyfile = ''
        self.phitype = 'block'
        self.pcafile = ''
        self.episodesperbatch = 50
        self.trainingepisodes = {}
        self.trainingepisodes_count = 0
        self.doForceSave = False 
        self.delta = 0.001 # Precondition value

        self._byeAction = None
        self.replace = {}
        self.slot_abstraction_file = os.path.join(Settings.root, 'policy/slot_abstractions/'+domainString + '.json')       # default mappings
        self.abstract_slots = False
        self.unabstract_slots = False

        # CONFIG:
        if Settings.config.has_option('policy', 'inpolicyfile'):
            self.inpolicyfile = Settings.config.get('policy', 'inpolicyfile')
        if Settings.config.has_option('policy', 'outpolicyfile'):
            self.outpolicyfile = Settings.config.get('policy', 'outpolicyfile')
        if Settings.config.has_option('policy_' + domainString, 'inpolicyfile'):
            self.inpolicyfile = Settings.config.get('policy_' + domainString, 'inpolicyfile')
        if Settings.config.has_option('policy_' + domainString, 'outpolicyfile'):
            self.outpolicyfile = Settings.config.get('policy_' + domainString, 'outpolicyfile')

        if Settings.config.has_option("lspi_"+domainString, "discount"):
            self.discount = Settings.config.getfloat("lspi_"+domainString, "discount")

        if Settings.config.has_option('policy_'+domainString, 'inpolicyfile'):
            self.inpolicyfile = Settings.config.get('policy_'+domainString, 'inpolicyfile')
            self.basefilename = '.'.join(self.inpolicyfile.split('.')[:-1])
            self.inpolicyfile = self.inpolicyfile + '.' + str(os.getpid())
            self.basefilename = self.basefilename + '.' + str(os.getpid())

        #if Settings.config.has_option('policy_'+domainString, 'outpolicyfile'):
        #    self.outpolicyfile = Settings.config.get('policy_'+domainString, 'outpolicyfile')
        #    self.outpolicyfile = self.outpolicyfile + '.' + str(os.getpid())

        if Settings.config.has_option('lspipolicy_'+domainString, 'phitype'):
            self.phitype = Settings.config.get('lspipolicy_' + domainString, 'phitype')

        if Settings.config.has_option('lspipolicy_'+domainString, 'pcafile'):
            self.pcafile = Settings.config.get('lspipolicy_'+domainString, 'pcafile')

        if Settings.config.has_option('exec_config', 'traindialogsperbatch'):
            self.episodesperbatch = int(Settings.config.get('exec_config', 'traindialogsperbatch'))

        policyType = 'all'
        if Settings.config.has_option('policy_'+domainString, 'policytype'):
            policyType = Settings.config.get('policy_'+domainString, 'policytype')

        # Fotis
        # Modifications for autoencoder
        self.save_step = 100
        if Settings.config.has_option('policy_'+domainString, 'save_step'):
            self.save_step = Settings.config.getint('policy_'+domainString, 'save_step')

        self.episodecount = 0
        self.learning = learning

        # LSPI stuff
        if os.path.isfile(self.inpolicyfile):
            self.loadLSPIParameters()
            self.setBasisFunctions()
            self.isInitialized = True
        else:
            self.isInitialized = False

        """Needed for AE | Fotis"""
        # Start
        self.domainUtil = FlatOnt.FlatDomainOntology(domainString)
        self.state_buffer = []

        self.is_transfer = False
        if Settings.config.has_option('exec_config', 'transfer'):
            self.is_transfer = Settings.config.getboolean('exec_config', 'transfer')

        self.is_Single = False
        if Settings.config.has_option('exec_config', 'single_autoencoder_type'):
            self.is_Single = True
        # self.batch_size = 256
        # if Settings.config.has_option('exec_config', 'autoencoder_minibatch_size'):
        #    self.batch_size = Settings.config.getint('exec_config', 'autoencoder_minibatch_size')

        # fotis
        self.save_step = 100
        if Settings.config.has_option('exec_config', 'save_step'):
            self.save_step = Settings.config.getint('exec_config', 'save_step')

        self.save_episodes = False
        if Settings.config.has_option('exec_config', 'save_episodes'):
            self.save_episodes = Settings.config.getboolean('exec_config', 'save_episodes')

        self.episodecount = 0

        # Modifications for autoencoders | fotis
        self.dae = None
        self.transfer_autoencoder = None

        if Settings.config.has_option('exec_config', 'autoencoder') and Settings.config.getboolean(
                'exec_config', 'autoencoder'):
            autoencoder_type = Settings.config.get('exec_config', 'single_autoencoder_type')
            self.dae = self.initSingleAutoEncoder(domainString, autoencoder_type)
        self.isAE = False
        if Settings.config.has_option('exec_config', 'autoencoder'):
            if Settings.config.getboolean('exec_config', 'autoencoder'):
                self.isAE = True
        # End

        #########################################################
        # Fotis | Initialisation method for Autoencoders
        #########################################################

    def initSingleAutoEncoder(self, domainString, autoencoder_type=None):
        if autoencoder_type == 'dense':
            from autoencoder.src.model import Autoencoder
        elif autoencoder_type == 'dae_transfer':
            from autoencoder.dae_transfer.model import Autoencoder
        elif autoencoder_type == 'variational_dense_denoising':
            from autoencoder.variational_dense_denoising.model import Autoencoder
        elif autoencoder_type == 'dense_denoising':
            from autoencoder.dense_denoising.model import Autoencoder
        elif autoencoder_type == 'variational':
            from autoencoder.variational.model import Autoencoder
        elif autoencoder_type == 'dense_multi':
            from autoencoder.dense_multi.model import Autoencoder
        elif autoencoder_type == 'sparse':
            from autoencoder.dense_sparse.model import Autoencoder
        else:
            from autoencoder.dense_multi.model import Autoencoder

        single_autoencoder = Autoencoder(domainString=domainString, policyType="lspi",
                                         variable_scope_name=domainString)
        return single_autoencoder

        
    def setBasisFunctions(self):
        # Basis functions:
        if self.phitype == 'block':
            self.basis = BlockBasis(self.domainString, self.numActions, self.stateDim)
        else:
            self.basis = None 

    def initializeLSPIparameters(self, stateDim):
        self.stateDim = stateDim
  
        self.setBasisFunctions() 

        self.A_inv = np.eye(self.basis.size())
        np.fill_diagonal(self.A_inv, 1.0/self.delta)
        self.b = np.zeros((self.basis.size(), 1))
        self.w = np.random.uniform(-0.1, 0.1, self.basis.size())



#########################################################
# overridden methods from Policy
######################################################### 
    
    def nextAction(self, belief):
        '''
        Selects next action to take based on the current belief and a list of non executable actions
        NOT Called by BCM
        
        :param belief:
        :type belief:
        :param hyps:
        :type hyps:
        :returns:
        '''
        nonExecutableActions = self.actions.getNonExecutable(belief, self.lastSystemAction)
        
        goalMethod = belief["beliefs"]["method"]
        if "finished" in goalMethod:
            if goalMethod["finished"] > 0.85 and self._byeAction is not None:
                return self._byeAction
            
        if self._byeAction is not None:
            nonExecutableActions.append(self._byeAction)
        currentstate = self.get_State(belief)
        executable = self._createExecutable(nonExecutableActions)

        if len(executable) < 1:
            logger.error("No executable actions")

        if not self.isInitialized:
            self.initializeLSPIparameters(len(currentstate.getStateVector()))
            self.isInitialized = True 

        best_action, best_Qvalue = self.policy(belief=currentstate, executable=executable)
        summaryAct = self._actionString(best_action.act) #best_action[0].act
        
        if self.learning:                    
            best_action.Qvalue = best_Qvalue  

        self.actToBeRecorded = best_action #summaryAct
        # Finally convert action to MASTER ACTION
        masterAct = self.actions.Convert(belief, summaryAct, self.lastSystemAction)
        return masterAct
    
    def savePolicy(self, FORCE_SAVE=False):
        '''
        Saves the LSPI policy.
        
        :param belief:
        :type belief:
        '''
        pass
        #if self.learning or (FORCE_SAVE and self.doForceSave):
        #    self.saveLSPIParameters() #learner.savePolicy()

    def savePolicyInc(self, FORCE_SAVE=False):
        """
        save model and replay buffer
        """
        if self.episodecount % self.save_step == 0:
            if self.learning or (FORCE_SAVE and self.doForceSave):
                self.saveLSPIParameters()
            # Fotis
            if self.dae is not None:
                self.dae.save_variables()
 
            print('savePolicyInc')   
            #print "episode", self.episodecount
            # save_path = self.saver.save(self.sess, self.out_policy_file+'.ckpt')
            '''self.dqn.save_network(self.out_policy_file + '.dqn.ckpt')

            f = open(self.out_policy_file + '.episode', 'wb')
            for obj in [self.samplecount, self.episodes[self.domainString]]:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            '''
            # logger.info("Saving model to %s and replay buffer..." % save_path) 

    def saveLSPIParameters(self):
        '''
        Saves the LSPI policy.
        '''
        print("Saving LSPI parameters to: " + self.outpolicyfile)
        PolicyUtils.checkDirExistsAndMake(self.outpolicyfile)
        with open(self.outpolicyfile, 'wb') as f:
            pickle.dump(self.stateDim, f)
            pickle.dump(self.A_inv, f)
            pickle.dump(self.b, f)
            pickle.dump(self.w, f)
        return

    def loadLSPIParameters(self):
        '''DAE_transfer
        Loads the LSPI policy.
        '''
        print("Loading LSPI parameters from:", self.inpolicyfile)
        with open(self.inpolicyfile, 'rb') as f:
            self.stateDim = pickle.load(f)
            self.A_inv = pickle.load(f)
            self.b = pickle.load(f)
            self.w = pickle.load(f)
        
        return
    
        
    def train(self):
        '''
        At the end of learning episode calls LearningStep for accumulated states and actions and rewards
        '''
        
        # SOMEWHAT TEMPORARY THING FOR HYPERPARAM PLAY
#         if self.collect_data:
#             if self.episode.rtrace[-1] == 20:  # check success
#                 self.data_for_hp.add_data(blist=self.episode.strace, alist=self.episode.atrace)
#             if self.data_for_hp.met_length():
#                 self.data_for_hp.write_data()
#                 raw_input('ENOUGH DATA COLLECTED')
#             return
#                 
        if self.USE_STACK: 
            self.episode_stack.add_episode(copy.deepcopy(self.episodes))    
            if self.episode_stack.get_stack_size() == self.PROCESS_EPISODE_STACK:
                self._process_episode_stack(self.episode_stack)
          
            self.savePolicyInc()

            return
        # process single episode
        else:
            for dstring in self.episodes:
                if self.episodes[dstring] is not None:
                    if len(self.episodes[dstring].atrace):   # domain may have just been part of committee but
                        # never in control - and whenever policy is booted an Episode() is created for its own domain ... 
                        episode = self.episodes[dstring]   
                        self._process_single_episode(episode)   

            self.savePolicyInc()

        return
    
    def convertStateAction(self, state, action):
        '''
        
        :param belief:
        :type belief:
        :param belief:
        :type belief:
        '''
        cState = state
        cAction = action
        
        if not isinstance(state, LSPIState):
            if isinstance(state, TerminalState):
                cState = TerminalLSPIState()
            else:
                cState = self.get_State(state)
                
        if not isinstance(action, LSPIAction):
            if isinstance(action, TerminalAction):
                cAction = TerminalLSPIAction()
            else:
                cAction = self.get_Action(action)

        return cState, cAction

#########################################################
# overridden methods from CommitteeMember
######################################################### 
    
    def get_State(self, beliefstate, keep_none=False):     
        '''
        Called by BCM
        
        :param beliefstate:
        :type beliefstate:
        :param keep_none:
        :type keep_none:
        '''
        # Fotis
        return LSPIState(beliefstate, autoencoder=self.dae, keep_none=keep_none, replace=self.replace,
                         domainString=self.domainString, domainUtil=self.domainUtil)

    def get_Action(self, action):     
        '''
        Called by BCM
        
        :param action:
        :type action:
        '''   
        actionIndex = self.actions.action_names.index(action.act) 
        return LSPIAction(action.act, actionIndex, self.numActions, replace=self.replace)

    
    def abstract_actions(self, actions):
        '''
        convert a list of domain acts to their abstract form based on self.replace
        
        :param actions:
        :type actions:
        '''
        if len(self.replace)>0:
            abstract = []
            for act in actions:
                if '_' in act:
                    [prefix,slot] = act.split('_')
                    if slot in self.replace:
                        abstract.append(prefix+'_'+self.replace[slot])
                    else:
                        abstract.append(act)
                else:
                    abstract.append(act)
            return abstract
        else:
            logger.error('No slot abstraction mapping has been given - check config')
              
    def unabstract_action(self, actions):
        '''
        action is a string
        
        :param actions:
        :type actions:
        '''
        if len(actions.split("_")) != 2:        # handle not abstracted acts like 'inform' or 'repeat' 
            return actions        
        [prefix, slot] = actions.split("_")
        if prefix == 'inform':              # handle not abstracted acts like 'inform_byname' or 'inform_requested'
            return actions
        else:                               # handle abstracted acts like 'request_slot00' or 'confirm_slot03'
            matching_actions = []
            for abs_slot in self.abstraction_mapping['abstract2real'].keys():
                if abs_slot == slot:                
                    match = prefix +'_'+ self.abstraction_mapping['abstract2real'][abs_slot]      
                    matching_actions.append(match)
            Settings.random.shuffle(matching_actions)                            
            return Settings.random.choice(matching_actions)
        
        logger.error('{} - no real slot found for this abstract slot'.format(actions)) 
        
#########################################################
# public methods
#########################################################  
            
    def getPolicyFileName(self):
        '''
        Returns the policy file name
        '''
        return self.policy_file
        
#########################################################
# private methods
######################################################### 
   

    def _createExecutable(self,nonExecutableActions):
        '''
        Produce a list of executable actions from non executable actions
        
        :param nonExecutableActions:
        :type nonExecutableActions:
        '''                
        executable_actions = []
        for act_i in self.actions.action_names:
            act_i_index = self.actions.action_names.index(act_i)

            if act_i in nonExecutableActions:
                continue
            elif len(self.replace) > 0:                            # with abstraction  (ie BCM)           
                # check if possibly abstract act act_i is in nonExecutableActions
                if '_' in act_i:
                    [prefix,slot] = act_i.split('_')
                    if slot in self.replace.keys():
                        if prefix+'_'+self.replace[slot] not in nonExecutableActions:       # assumes nonExecutable is abstract 
                            executable_actions.append(LSPIAction(act_i, act_i_index, self.numActions, replace=self.replace))
                        else:
                            pass # dont add in this case
                    else:       # some actions like 'inform_byname' have '_' in name but are not abstracted
                        executable_actions.append(LSPIAction(act_i, act_i_index, self.numActions, replace=self.replace))
                else:           # only abstract actions with '_' in them like request_area --> request_slot1 etc
                    executable_actions.append(LSPIAction(act_i, act_i_index, self.numActions, replace=self.replace))                
            else:                   # no abstraction
                executable_actions.append(LSPIAction(act_i, act_i_index, self.numActions))    #replace not needed here - no abstraction
        return executable_actions



    def _actionString(self, act):
        '''
        Produce a string representation from an action - checking as well that the act coming in is valid
        Should only be called with non abstract action. Use _unabstract_action() otherwise
        
        :param act:
        :type act:
        '''        
        if act in self.actions.action_names:
            return act           
        logger.error('Failed to find action %s' % act)
        
    def _process_episode_stack(self, episode_stack):
        '''With BCM - items on the stack are now dictionaries (keys=domain names, values=Episode() instances)
        '''
        
        # copy original policy to observe how far we deviate from it as we sequentially move through our batch of episodes, updating
        #self.orig_learner = copy.deepcopy(self.learner)  # nb: deepcopy is slow
        
        # process episodes - since adding BCM - now have domain_episodes -- 
        for episode_key in episode_stack.episode_keys():                    
            domain_episodes = episode_stack.retrieve_episode(episode_key)
            for dstring in domain_episodes:
                if domain_episodes[dstring] is not None:
                    if len(domain_episodes[dstring].atrace):   # domain may have just been part of committee but
                        # never in control - and whenever policy is booted an Episode() is created for its own domain ... 
                        self._process_single_episode(domain_episodes[dstring], USE_STACK=True)
        return 
    
    def _process_single_episode(self, episode, USE_STACK = False):
        if len(episode.strace) == 0:
            logger.warning("Empty episode")
            return
        if not self.learning:
            logger.warning("Policy not learning")
            return

        episode.check()  # just checks that traces match up.  
        # Fotis
        # Modifications for autoencoder
        # Transfered the state buffer in the autoencoder
        # transfer AE exists in PolicyManager.py
	if self.isAE:
            self.check_n_train_ae(episode)

        self.episodecount += 1
        # End of modifications

        i = 1
        r = 0
        is_ratios = []
        while i < len(episode.strace) and self.learning:
            
            # FIXME how are state/action-pairs recorded? generic or specific objects, ie, State or LSPIState?
            
            # pLSPIState = self.get_State(episode.strace[i-1])
            # pLSPIAction = self.get_Action(episode.atrace[i-1])
            # cLSPIState = self.get_State(episode.strace[i])
            # cLSPIAction = self.get_Action(episode.atrace[i])
            
            pLSPIState = episode.strace[i-1]
            pLSPIAction = episode.atrace[i-1]
            cLSPIState = episode.strace[i]
            cLSPIAction = episode.atrace[i]

            self.isInitial = False
            self.isTerminal = False
              
            if i == 1:
                self.isInitial = True
            
            if i+1 == len(episode.strace) or isinstance(episode.strace[i], TerminalLSPIState):
                self.isTerminal = True
                r = episode.getWeightedReward()
                
            self.learningStep(pLSPIState, pLSPIAction, r, cLSPIState, cLSPIAction)
            i+=1
            
            if (self.isTerminal and i < len(episode.strace)):
                logger.warning("There are {} entries in episode after terminal state for domain {} with episode of domain {}".format(len(episode.strace)-i,self.domainString,episode.learning_from_domain))
                break

        #self.saveLSPIParameters()  
        return
    # Fotis
    def check_n_train_ae(self, episode):
        if self.learning:
            #if not (type(episode).__module__ == np.__name__):
            for i in range(len(episode.strace)):
                if episode.atrace[i].toString() != 'TerminalLSPIAction':

                    if self.is_Single:
                        self.dae.saveToStateBuffer(episode.strace[i].flatBeliefVec, episode.strace_clean[i].flatBeliefVec)

                        if self.dae.checkReadyToTrain():
                            state_batch, state_clean_batch = self.dae.getTrainBatch()
                            self.dae.train(state_batch, state_clean_batch)
                            #self.autoencoder.saveEpisodesToFile(self.save_episodes)
                            self.dae.resetStateBuffer()
                            try:
                                path = self.dae.save_variables()
                                #print("Variables Saved at: ", path)
                            except:
                                print("Variables Save Failed!")
                                pass
                    if self.is_transfer:
                        # check if we use the AE in PolicyManager
                        self.transfer_autoencoder.saveToStateBuffer(episode.strace[i].flatBeliefVec, episode.strace_clean[i].flatBeliefVec)

                        if self.transfer_autoencoder.checkReadyToTrain():
                            state_batch, state_clean_batch = self.transfer_autoencoder.getTrainBatch()
                            self.transfer_autoencoder.train(state_batch, state_clean_batch)
                            # self.autoencoder.saveEpisodesToFile(self.save_episodes)
                            self.transfer_autoencoder.resetStateBuffer()
                            try:
                                path = self.transfer_autoencoder.save_variables()
                                #print("Variables Saved at: ", path)
                            except:
                                print("Variables Save Failed!")
                                pass

    def Qvalue(self, belief, action):
        """
        :returns: Q value for a given state, action and the basis function
        """
        phi=self.basis.evaluate(belief, action)
        qvalue = self.w.dot(phi)
        return qvalue

    def policy(self, belief, executable):
        """
        :returns: best action according to Q values
        """
        if len(executable) == 0:
            logger.error("No executable actions")

        if not self.isInitialized:
            # Settings.random.shuffle(executable)        -- can be random.choose()
            # print "Choosing randomly ", executable[0].act
            action = Settings.random.choice(executable)
            cAction = self.get_Action(action)
            return [cAction, 0.0]

        Q = []
        for action in executable:
            cAction = self.get_Action(action)
            value = self.Qvalue(belief, cAction)
            Q.append((cAction, value))
        Q = sorted(Q, key=lambda val: val[1], reverse=True)

        best_action, best_Qvalue = Q[0][0], Q[0][1]

        return best_action, best_Qvalue
   
    def learningStep(self, pLSPIState, pLSPIAction, r, cLSPIState, cLSPIAction):
        k = self.basis.size()
        
        phi_sa = self.basis.evaluate(pLSPIState, pLSPIAction).reshape((-1, 1))

        if pLSPIState is not TerminalLSPIState:
            #best_action = self.best_action(cLSPIState)
            phi_sprime = self.basis.evaluate(cLSPIState, cLSPIAction).reshape((-1, 1))
        else:
            phi_sprime = np.zeros((k, 1))

        A1 = np.dot(self.A_inv, phi_sa)
        g = (phi_sa - self.discount*phi_sprime).T

        self.A_inv += - np.dot(A1, np.dot(g, self.A_inv))/(1 + np.dot(g, A1))

        self.b += phi_sa*r

        self.w = np.dot(self.A_inv, self.b).reshape((-1, ))
    
    
class LSPIAction(Action):
    '''
    Definition of summary action used for LSPI.
    '''
    def __init__(self, action, actionIndex, numActions, replace={}):    
        self.numActions = numActions
        self.act=action  
        self.actid = actionIndex   
        self.is_abstract = True if len(replace) else False           # record whether this state has been abstracted -   
        
        # append to the action the Q value from when we chose it --> for access in batch calculations later
        self.Qvalue = 0
        
        if len(replace) > 0:
            self.act = self.replaceAction(action, replace)
        
                
    def replaceAction(self, action, replace):
        '''
        Used for making abstraction of an action
        '''
        if "_" in action:
            slot = action.split("_")[1]
            if slot in replace:
                replacement = replace[slot]
                return action.replace(slot, replacement)        # .replace() is a str operation
        return action


    def __eq__(self, a):
        """
        Action are the same if their strings match
        :rtype : bool
        """
        if self.numActions != a.numActions:
            return False
        if self.act != a.act:
                return False
        return True

    def __ne__(self, a):
        return not self.__eq__(a)

    def show(self):
        '''
        Prints out action and total number of actions
        '''
        print str(self.act), " ", str(self.numActions)


    def toString(self):
        '''
        Prints action
        '''
        return self.act
    
    def __repr__(self):
        return self.toString()
    
class LSPIState(State):
    '''
    Definition of state representation needed for LSPI algorithm
    Main requirement for the ability to compute kernel function over two states
    '''    
    def __init__(self, belief, autoencoder=None, keep_none=False, replace={}, domainString=None, domainUtil=None):
        self.domainString = domainString
 
        self.autoencoder = autoencoder
        self._bstate = {}
        self.keep_none = keep_none
        
        self.is_abstract = True if len(replace) else False           # record whether this state has been abstracted -
        #self.is_abstract = False 
        
        self.beliefStateVec = None
        self.flatBeliefVec = None

        self.isSummary = None
        if Settings.config.has_option('policy', 'summary'):
            self.isSummary = Settings.config.get('policy', 'summary')

        # self.extractBelief(b, replace)
        # Fotis
        if belief is not None:
            if autoencoder is None:  # Modifications for autoencoder
                if isinstance(belief, LSPIState):
                    self._convertState(belief, replace)
                else:
                    if self.isSummary:
                        self.extractBelief(belief, replace)
                    else:
                        self.extractSimpleBelief(belief, replace)
            else:
                # Modifications for AE
                # Fotis
                self.flatBeliefVec = np.array(flatten_belief(belief, domainUtil), dtype=np.float32)
                self.beliefStateVec = autoencoder.encode(self.flatBeliefVec.reshape((1, -1))).reshape((-1,))
                self.hello = True

                # End of modifications

    def extractBeliefWithOther(self, belief, sort=True):
        '''
        Copies a belief vector, computes the remaining belief, appends it and returnes its sorted value

        :return: the sorted belief state value vector
        '''

        bel = copy.deepcopy(belief)
        res = []

        if '**NONE**' not in belief:
            res.append(1.0 - sum(belief.values()))  # append the none probability
        else:
            res.append(bel['**NONE**'])
            del bel['**NONE**']

        # ensure that all goal slots have dontcare entry for GP belief representation
        if 'dontcare' not in belief:
            bel['dontcare'] = 0.0

        if sort:
            # sorting all possible slot values including dontcare
            res.extend(sorted(bel.values(), reverse=True))
        else:
            res.extend(bel.values())
        return res

    def extractSingleValue(self, val):
        '''
        for a probability p returns a list  [p,1-p]
        '''
        return [val, 1 - val]

    def extractSimpleBelief(self, b, replace={}):
        '''
        From the belief state b extracts discourseAct, method, requested slots, name, goal for each slot,
        history whether the offer happened, whether last action was inform none, and history features.
        Sets self._bstate
        '''
        with_other = 0
        without_other = 0
        self.isFullBelief = True

        for elem in b['beliefs'].keys():
            if elem == 'discourseAct':
                self._bstate["goal_discourseAct"] = b['beliefs'][elem].values()
                without_other += 1
            elif elem == 'method':
                self._bstate["goal_method"] = b['beliefs'][elem].values()
                without_other += 1
            elif elem == 'requested':
                for slot in b['beliefs'][elem]:
                    cur_slot = slot
                    if len(replace) > 0:
                        cur_slot = replace[cur_slot]
                    self._bstate['hist_' + cur_slot] = self.extractSingleValue(b['beliefs']['requested'][slot])
                    without_other += 1
            else:
                if elem == 'name':
                    self._bstate[elem] = self.extractBeliefWithOther(b['beliefs']['name'])
                    with_other += 1
                else:
                    cur_slot = elem
                    if len(replace) > 0:
                        cur_slot = replace[elem]

                    self._bstate['goal_' + cur_slot] = self.extractBeliefWithOther(b['beliefs'][elem])
                    with_other += 1

                    additionalSlots = 2
                    # if elem not in Ontology.global_ontology.get_system_requestable_slots(self.domainString):
                    #     additionalSlots = 1
                    if len(self._bstate['goal_' + cur_slot]) != \
                            Ontology.global_ontology.get_len_informable_slot(self.domainString,
                                                                             slot=elem) + additionalSlots:
                        print self._bstate['goal_' + cur_slot]
                        logger.error("Different number of values for slot " + cur_slot + " " + str(
                            len(self._bstate['goal_' + cur_slot])) + \
                                     " in ontology " + str(
                            Ontology.global_ontology.get_len_informable_slot(self.domainString, slot=elem) + 2))

        self._bstate["hist_offerHappened"] = self.extractSingleValue(1.0 if b['features']['offerHappened'] else 0.0)
        without_other += 1
        self._bstate["hist_lastActionInformNone"] = self.extractSingleValue(
            1.0 if len(b['features']['informedVenueSinceNone']) > 0 else 0.0)
        without_other += 1
        for i, inform_elem in enumerate(b['features']['inform_info']):
            self._bstate["hist_info_" + str(i)] = self.extractSingleValue(1.0 if inform_elem else 0.0)
            without_other += 1

        # Tom's speedup: convert belief dict to numpy vector
        self.beliefStateVec = self.slowToFastBelief(self._bstate)

        return

    def extractBelief(self, b, replace={}):
        '''NB - untested function since __init__ makes choice to use extractSimpleBelief() instead
        '''
        self.isFullBelief = True

        self._bstate["hist_lastActionInformNone"] = self.extractSingleValue(b["features"]["lastActionInformNone"])
        self._bstate["hist_offerHappened"] = self.extractSingleValue(b["features"]["offerHappened"])
        self._bstate["goal_name"] = self.extractBeliefWithOther(b["beliefs"]["name"])
        self._bstate["goal_discourseAct"] = b["beliefs"]["discourseAct"].values()
        self._bstate["goal_method"] = b["beliefs"]["method"].values()
        '''
        for i in xrange(len(b["goal"])):
            curSlotName = b["slotAndName"][i]
            if len(replace) > 0:
                curSlotName = replace[curSlotName]
            self._bstate["goal_" + curSlotName] = self.extractBeliefWithOther(b["goal"][i])

        for i in range(min(len(b["slotAndName"]), len(b["goal_grounding"]))):
            histName = b["slotAndName"][i]
            if len(replace) > 0:
                histName = replace[histName]
            self._bstate["hist_" + histName] = b["goal_grounding"][i].values()

        for i in range(min(len(b["infoSlots"]), len(b["info_grounding"]))):
            infoName = b["infoSlots"][i]
            if len(replace) > 0:
                infoName = replace[infoName]
            self._bstate["hist_" + infoName] = b["info_grounding"][i].values()
        '''
        self.state_size = len(self._bstate)
        # Tom's speedup: convert belief dict to numpy vector
        self.beliefStateVec = self.slowToFastBelief(self._bstate)
        self.myname = True

    def slowToFastBelief(self, bdic):
        '''Converts dictionary format to numpy vector format
        '''
        values = np.array([])
        for slot in sorted(bdic.keys()):
            if slot == "hist_location":
                continue
            #             if "goal" in slot and slot != "goal_discourseAct" and slot != "goal_method":
            #                 toadd = np.array(bdic[slot])
            #                 values = np.concatenate((values, np.sort(toadd)[::-1]))
            #             else :
            #                 values = np.concatenate((values, np.array(bdic[slot])))

            # su259 sorting already done before
            values = np.concatenate((values, np.array(bdic[slot])))
            return values

    def _convertState(self, b, replace={}):
        '''
        converts GPState to GPState of shape of current domain by padding/truncating slots/values

        assumes that non-slot information is the same for both
        '''

        # 1. take care of non-slot information
        self._bstate["goal_discourseAct"] = copy.deepcopy(b._bstate['goal_discourseAct'])
        self._bstate["goal_method"] = copy.deepcopy(b._bstate['goal_method'])

        self._bstate['hist_offerHappened'] = copy.deepcopy(b._bstate['hist_offerHappened'])
        self._bstate['hist_lastActionInformNone'] = copy.deepcopy(b._bstate['hist_lastActionInformNone'])

        # copy remaining hist information:
        for elem in b._bstate:
            if 'hist_info_' in elem:
                self._bstate[elem] = copy.deepcopy(b._bstate[elem])

        # requestable slots
        origRequestSlots = Ontology.global_ontology.get_requestable_slots(self.domainString)
        if len(replace) > 0:
            requestSlots = map(lambda x: replace[x], origRequestSlots)
        else:
            requestSlots = origRequestSlots

        for slot in requestSlots:
            if 'hist_' + slot in b._bstate:
                self._bstate['hist_' + slot] = copy.deepcopy(b._bstate['hist_' + slot])
            else:
                self._bstate['hist_' + slot] = self.extractSingleValue(0.0)

        # informable slots

        origInformSlots = Ontology.global_ontology.get_informable_slots(self.domainString)
        informSlots = {}
        for slot in origInformSlots:
            curr_slot = slot
            if len(replace) > 0:
                curr_slot = replace[curr_slot]
            informSlots[curr_slot] = Ontology.global_ontology.get_len_informable_slot(self.domainString,
                                                                                      slot) + 2  # dontcare + none

        slot = 'name'
        self._bstate[slot] = b._bstate[slot]
        if len(self._bstate[slot]) > informSlots[slot]:
            # truncate
            self._bstate[slot] = self._bstate[slot][0:informSlots[slot]]
        elif len(self._bstate[slot]) < informSlots[slot]:  # 3 < 5 => 5 - 3
            # pad with 0
            self._bstate[slot].extend([0] * (informSlots[slot] - len(self._bstate[slot])))
        del informSlots[slot]

        for curr_slot in informSlots:
            slot = 'goal_' + curr_slot
            if slot in b._bstate:
                self._bstate[slot] = b._bstate[slot]
                if len(self._bstate[slot]) > informSlots[curr_slot]:
                    # truncate
                    self._bstate[slot] = self._bstate[slot][0:informSlots[curr_slot]]
                elif len(self._bstate[slot]) < informSlots[curr_slot]:  # 3 < 5 => 5 - 3
                    # pad with 0
                    self._bstate[slot].extend([0] * (informSlots[curr_slot] - len(self._bstate[slot])))
            else:
                # create empty entry
                self._bstate[slot] = [0] * informSlots[curr_slot]
                self._bstate[slot][0] = 1.0  # the none value set to 1.0

        # Tom's speedup: convert belief dict to numpy vector
        self.beliefStateVec = self.slowToFastBelief(self._bstate)

        return

    def getStateVector(self):
        return self.beliefStateVec 
    
    
    def toString(self):
        '''
        String representation of the belief
        '''
        res = ""

        if len(self._bstate) > 0:
            res += str(len(self._bstate)) + " "
            for slot in self._bstate:
                res += slot + " "
                for elem in self._bstate[slot]:
                    for e in elem:
                        res += str(e) + " "
        return res
    
    def __repr__(self):
        return self.toString()
    
class TerminalLSPIAction(TerminalAction, LSPIAction):
    '''
    Class representing the action object recorded in the (b,a) pair along with the final reward. 
    '''
    def __init__(self):
        self.act = 'TerminalLSPIAction'
        self.actid = -1
        self.is_abstract = None
        self.numActions = None

class TerminalLSPIState(LSPIState,TerminalState):
    '''
    Basic object to explicitly denote the terminal state. Always transition into this state at dialogues completion. 
    '''
    def __init__(self):
        super(TerminalLSPIState, self).__init__(None)

# END OF FILE
