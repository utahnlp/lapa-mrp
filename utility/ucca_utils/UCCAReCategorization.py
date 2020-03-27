#!/usr/bin/env python3.6
# coding=utf-8
'''

UCCAReCategorizor use a set of templates built from training corpous and deterministic rules
to recombine/recategorize a fragment of UCCA graph into a single node for concept identification.
It also stores frequency of sense for frame concept. (based on training set)
Different from AMR, named entity will not be recategorized,
for now, I only thinks the mwe may can be categorized here.

@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-30
'''
from utility.ucca_utils.UCCAStringCopyRules import *

from utility.data_helper import *
import logging

logger = logging.getLogger("ucca.ReCategorization")

import threading
class UCCAReCategorizor(object):
    def __init__(self):
        pass
