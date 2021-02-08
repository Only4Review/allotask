# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:03:33 2020

@author: xxx
"""

class Alg3PolicyAllocator:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env
    
    def select_action(self, use_moments):
        action = self.policy.evaluate(self.env, use_moments=use_moments)
        return action
        
    def update_state(self, action):
        new_state, done = self.env.step(action)
        return action, new_state, done
