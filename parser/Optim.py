import math,torch
import torch.optim as optim
import numpy as np
from torch.nn.utils.clip_grad import *
from pytorch_transformers.optimization import *
import logging
import re
logger = logging.getLogger("mrp.Optim")

SCHEDULES = {
    "none" : None,
    "constant":WarmupConstantSchedule,
    "warmup_constant": WarmupConstantSchedule,
    "warmup_linear" : WarmupLinearSchedule,
    "warmup_cosine" : WarmupCosineSchedule,
    "warmup_cosine_hardrestart" : WarmupCosineWithHardRestartsSchedule
}

class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.grouped_params, lr=self.lr,weight_decay = self.weight_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.grouped_params, lr=self.lr,weight_decay = self.weight_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.grouped_params, lr=self.lr,weight_decay = self.weight_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.grouped_params, betas=[0.9,0.9],lr=self.lr,weight_decay = self.weight_decay)
        elif self.method == "RMSprop":
            self.optimizer = optim.RMSprop(self.grouped_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.method == 'AdamW' or self.method == "BertAdam":
            # if the params is already a parameter group, then just add it
            # do grad norm outside the BertAdam, not use the grad_norm in BertAdam
            # https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/optimization.py
            self.optimizer = AdamW(self.grouped_params,
                                      lr=self.lr,
                                      correct_bias=False) # To reproduce BertAdam specific behavior set correct_bias=False
        else:
            raise RuntimeError("Invalid optim method: " + self.method)


    def _makeOptimScheduler(self):
        # initialize schedule object
        if self.optim_scheduler_name in SCHEDULES:
            scheduler_type = SCHEDULES[self.optim_scheduler_name]
            if scheduler_type != None:
                self.optim_scheduler = scheduler_type(self.optimizer, warmup_steps=self.warmup_steps, t_total=self.num_train_optimization_steps)
            else:
                self.optim_scheduler = None
        else:
            self.optim_scheduler = None
            raise RuntimeError("Invalid optim scheduler: " + self.optim_scheduler_name)

    def __init__(self, named_params, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None, weight_decay=0, warmup_proportion = -1.0, num_train_optimization_steps = -1, optim_scheduler_name= "none",
                 grouped_dict_configs = None):
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.num_train_optimization_steps = num_train_optimization_steps
        self.warmup_steps = warmup_proportion * self.num_train_optimization_steps
        self.optim_scheduler_name = optim_scheduler_name
        if self.method == 'AdamW':
            logger.info("When method is AdamW, lr_schudler = {}, num_train_optimization_step={}, warmup_propotion = {}".format(self.optim_scheduler_name, self.num_train_optimization_steps, self.warmup_steps))

        self.global_step = 0
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self.weight_decay =  weight_decay
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_model = ['bert_model', '_scalar_mix']  #  make sure all bert_model is initialized as a module called "bert_model", '_scalar_mix' for mixed weight
        self.named_params = list(named_params)
        self.params = [p for n, p in self.named_params]
        # logger.info("named_parameters : {}".format([n for n, p in self.named_params]))
        # The parameters here will overwrite the global
        if grouped_dict_configs is None:
            self.grouped_params= [
                {'params': [p for n, p in self.named_params if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self.named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        else:
            # if given grouped_dict configs:
            self.grouped_params = []
            covered_param_names = []
            for k, v in grouped_dict_configs.items():
                d = {}
                assert isinstance(v,dict), "valud of grouped dict should be a dict"
                n_p = [(n, p) for n, p in self.named_params if re.match(k, n)]
                d['params'] = [p for n, p in n_p]
                covered_param_names.extend([n for n, p in n_p])
                d.update(v)
                self.grouped_params.append(d)

            uncovered_params = [p for n, p in self.named_params if n not in covered_param_names]
            uncovered_dict = {'params' : uncovered_params}
            self.grouped_params.append(uncovered_dict)
            logger.info("covered_param_names:{}".format(covered_param_names))

        self.weight_shirnk = 1.0 - weight_decay

        self._makeOptimizer()
        if self.method == 'AdamW':
            self._makeOptimScheduler()
        else:
            self.optim_scheduler = None


    def step(self):
        if self.optim_scheduler:
            self.optim_scheduler.step()
        # when gradient is exploded, try to cut them, then grad_norm
        # clip_grad_value_(self.params, self.max_grad_norm)
        grad_norm = clip_grad_norm_(self.params, self.max_grad_norm, norm_type=2)
        # Now warmup is only been used in BertAdam
        self.optimizer.step()
        self.global_step += 1
        #for param in self.params:
        #    assert not np.isnan(np.sum(param.data.cpu().numpy())),("befotr shrink\n",param)
        #    param.data.mul_(self.weight_shirnk) #+ torch.normal(0,1e-3*torch.ones(param.data.size()).cuda())
        #    assert not np.isnan(np.sum(param.data.cpu().numpy())),("after shrink\n",param)
        return grad_norm

#    def updateLearningRateWithScheduler(self):
#        """
#        BertAdam has implemented the schuelder in it self, hence, skip it
#        For other method, we update the learning rate.
#        """
#        if self.method != 'AdamW' and self.optim_scheduler is not None:
#            lr_this_step = self.lr * self.optim_scheduler.get_lr(
#                self.global_step/self.num_train_optimization_steps,
#                self.warmup_steps)
#            for param_group in self.optimizer.param_groups:
#                param_group['lr'] = lr_this_step

    def get_est_global_step(self):
        """
        in the optimizer, differe parameter may have different state, which contains the step.
        it is the time for step() function get called.
        it can be used for warmup or lr schedule
        """
        return self.global_step

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRateWithLRDecay(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_ppl = ppl

        self._makeOptimizer()

