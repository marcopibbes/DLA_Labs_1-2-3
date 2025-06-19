import os
import time
import copy
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

class BaseTrainingPipeline:
    def __init__(
            self,
            
            model,
            optimizer,
            lr_scheduler = None,
            metrics = {},
            device = "cpu",

            run_id = None,
            checkpoint_dir = None,
            resume = False,
            
            log_dir = None,
            
            callbacks = {}
        ):

        self._model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics

        self.device = torch.device(device)
        self.model.to(device)

        #history
        self.history = {}
        self._current_epoch = -1
        self._current_batch = -1
        self._steps_without_improv = -1
        self.best_score = None
        self.best_state = None

        #checkpoints
        self.run_id = run_id or f'{self.model.__class__.__name__}_{time.strftime("%Y%m%d-%H%M%S")}'
        self.checkpoint_path = None
        self.interrupted_checkpoint_path = None
        self.resume = resume

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok = True)
            self.checkpoint_path = os.path.join(checkpoint_dir, self.run_id) + ".pth"
            self.interrupted_checkpoint_path = os.path.join(checkpoint_dir, self.run_id) + "_interrupted.pth"
            
            #bit crappy resume, could probably be improved/re-thought #TODO
            if self.resume:
                #we first check if a interrupted version exist of the given run_id (we suppose its more up to date)
                if os.path.exists(self.interrupted_checkpoint_path):
                    self.load(interrupted = True)
                #else we try to get the run_id checkpoint as it is
                elif os.path.exists(self.checkpoint_path):
                    self.load(interrupted = False)

        #tensorboard
        self.log_dir = None
        self.writer = None

        if log_dir:
            os.makedirs(log_dir, exist_ok = True)
            self.log_dir = os.path.join(log_dir, self.run_id)
            self.writer = SummaryWriter(log_dir = self.log_dir)

        #callbacks
        base_callbacks = {
            'on_init': [],
            'on_fit_begin': [],
            'on_fit_end': [],
            'on_train_begin': [],
            'on_train_end': [],
            'on_val_begin': [],
            'on_val_end': [],
            'on_eval_begin': [],
            'on_eval_end': [],
            'on_best': []
        }

        if self.callbacks is None:
            self.callbacks = base_callbacks
        else: #this way if derived class want to define specific callbacks they can before calling the base constructor
            self.callbacks.update(base_callbacks)

        self.callbacks_data = {} #to pass data between callbacks

        for phase, callback_list in callbacks.items():
            assert phase in self.callbacks, f"phase '{phase}' not supported"
            self.callbacks[phase].extend(callback_list)

        self._execute_callbacks('on_init')

    #making some aliases for convenience 
    #model will become policy in RL pipeline
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    #current epoch will become current episode
    @property
    def current_epoch(self):
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, value):
        self._current_epoch = value

    #current batch will become current step
    @property 
    def current_batch(self):
        return self._current_batch

    @current_batch.setter
    def current_batch(self, value):
        self._current_batch = value
    ###########################################

    def _execute_callbacks(self, phase, **context):
        assert phase in self.callbacks, f"phase '{phase}' not supported" #probably not needed to do this check at runtime
        
        for callback in self.callbacks[phase]:
            callback(self, **context)

    def _init_history(self):
        if self.lr_scheduler:
            self.history['lr'] = []

    def plot(self):
        assert self.history and len(self.history) >= 1, "must call fit() first, no history to plot"

        #checking history keys, if the have common sufixes than add to same plot (like for train_loss and val_loss)
        #if instead they are different (like for loss and accuracy) than add to different plots
        #we will aspect that metrics to have the following structure {stage}_{metric}
 
        #make the groups to plot together the same metrics
        groups = {}
        for key in self.history:
            if '_' in key: stage, metric = key.rsplit('_', 1)
            else: stage, metric = None, key
            groups.setdefault(metric, []).append((stage, key))

        n_metrics = len(groups)
        fig, axs = plt.subplots(n_metrics, 1, figsize = (10, 5 * n_metrics), sharex = True)
        
        if n_metrics == 1:
            axs = [axs]

        for ax, (metric, entries) in zip(axs, groups.items()):
            for stage, key in entries:
                values = np.array(self.history[key])
                x = np.arange(len(values))
                ax.plot(x[:], values[:], label = (f"{stage}_{metric}" if stage else metric))
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()

        return fig, axs

    def save(self, interrupted = False):
        assert self.checkpoint_path, "checkpoint path not set"
        file_path = self.interrupted_checkpoint_path if interrupted else self.checkpoint_path
        os.makedirs(os.path.dirname(file_path), exist_ok = True)

        states = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_score': self.best_score,
            'best_state': self.best_state,
            'run_id': self.run_id
            #we will not save current batch since we always interrupt the training at the end of an epoch
        }

        if self.lr_scheduler: 
            states['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        self._add_states(states)
        
        torch.save(states, file_path)

    def _add_states(self, states):
        pass

    def load(self, interrupted = False):
        assert self.checkpoint_path, "checkpoint path not set"
        file_path = self.interrupted_checkpoint_path if interrupted else self.checkpoint_path

        if not os.path.exists(file_path):
            print(f"checkpoint file not found: {file_path}")
            self.resume = False  #cannot resume if file not found
            return
        
        states = torch.load(file_path, map_location = self.device)
        
        self.model.load_state_dict(states['model_state_dict'])
        self.optimizer.load_state_dict(states['optimizer_state_dict'])
        
        if self.lr_scheduler and 'scheduler_state_dict' in states:
            self.lr_scheduler.load_state_dict(states['scheduler_state_dict'])
        
        self.history = states.get('history', {})
        self.best_score = states.get('best_score')
        self.best_state = states.get('best_state')
        self.run_id = states.get('run_id', self.run_id)
        self.current_epoch = states.get('epoch', -1)
        self.current_batch = -1  
        self._step_without_improv = -1

        self._load_states(states)
        self.resume = True

    def _load_states(self, states):
        pass

    def get_callbacks_data(self):
        return self.callbacks_data

    def _handle_best(self, results, metric_to_monitor, monitor_mode):
        current_value = results.get(metric_to_monitor)

        if  (monitor_mode == 'min' and current_value < self.best_score) or \
            (monitor_mode == 'max' and current_value > self.best_score):
             
            self.best_score = current_value
            self.best_state = copy.deepcopy(self.model.state_dict())
            self._steps_without_improv = 0
            
            self._execute_callbacks('on_best', best_score = self.best_score, best_state = self.best_state)
            
            if self.checkpoint_path:
                self.save()
            
        else:
            self._steps_without_improv += 1
        
    def _handle_lr_scheduler(self, results, metric_to_monitor):
        if not self.lr_scheduler:
            return

        current_lr = self.optimizer.param_groups[0]['lr']
        scheduler_step_params = inspect.signature(self.lr_scheduler.step).parameters
        metric_for_scheduler = results.get(metric_to_monitor)
        
        if 'metrics' in scheduler_step_params and metric_for_scheduler is not None:
            if not np.isnan(metric_for_scheduler):
                self.lr_scheduler.step(metric_for_scheduler)
        else:
            self.lr_scheduler.step()

        new_lr = self.optimizer.param_groups[0]['lr']
        if self.history["lr"] is None or len(self.history["lr"]) == 0 or current_lr != self.history["lr"][-1]:
            self.history["lr"].append(current_lr)
        self.history["lr"].append(new_lr)

    def _handle_history_and_log(self, results, step):
        #history
        for key, value in results.items():
            if key in self.history:
                self.history[key].append(value)
            else:
                self.history[key] = [value]
        
        #tensorboard
        if self.writer:
            for key, value in results.items():
                if not isinstance(value, (np.ndarray, list)) and not np.isnan(value):
                    self.writer.add_scalar(key, value, step)
            self.writer.flush()

    def _cleanup_and_reload_best(self):
        #final save to have the whole history even if training stops early
        if self.checkpoint_path:
            self.save()

        if self.writer:
            self.writer.close()
            self.writer = None
        
        if self.best_state:
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)