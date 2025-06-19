from BaseTrainingPipeline import BaseTrainingPipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import inspect
import copy
import os
import time
import copy
import inspect
import numpy as np
import matplotlib.pyplot as plt

class SLTrainingPipeline(BaseTrainingPipeline):
    def __init__(
            self,
            *,
            criterion = nn.CrossEntropyLoss(),
            **base_kwargs        
        ):
        
        #specific callbacks
        self.callbacks = {
            'on_epoch_begin': [],
            'on_epoch_end': [],
            'on_batch_begin': [],
            'on_after_back': [],
            'on_batch_end': []
        }

        super().__init__(**base_kwargs)

        self.criterion = criterion
        if hasattr(self.criterion, "to"):
            self.criterion.to(self.device)

    def _init_history(self):
        self.history = {'train_loss': [], 'val_loss': []} #default always tracked metrics that we want to monitor
        for m in self.metrics.keys():
            self.history[f'train_{m}'] = []
            self.history[f'val_{m}'] = []

        super()._init_history()

    def _run_epoch(self, data_loader, verbose = True):
        tot_loss = 0.0
        tot_metrics = {m: 0.0 for m in self.metrics.keys()}
        n_batches = 0
        tot_samples = 0
        
        update_grad = torch.enable_grad()
        if not self.model.training: 
            update_grad = torch.no_grad()

        with update_grad:
            for batch_idx, batch in enumerate(tqdm(data_loader, leave = False, disable = not verbose)):
                self.current_batch = batch_idx

                X, y = batch
                X = X.to(self.device, non_blocking = True)

                if isinstance(y, (list, tuple)): #ugly handling of dinstillation
                    y_tuple = tuple(item.to(self.device, non_blocking = True) for item in y)
                    y = y_tuple
                else:
                    y = y.to(self.device, non_blocking = True)

                batch_size = X.size(0)

                self._execute_callbacks('on_batch_begin', X = X, y = y, batch_size = batch_size, is_training = self.model.training)

                logits = self.model(X)
                loss = self.criterion(logits, y)

                if self.model.training:
                    self.optimizer.zero_grad()
                    loss.backward()        
                    self._execute_callbacks('on_after_back', X = X, y = y, logits = logits, loss = loss.item(), batch_size = batch_size, metrics = tot_metrics, is_training = self.model.training)
                    self.optimizer.step()

                tot_loss += loss.item() * batch_size
                
                for m, func in self.metrics.items():
                    metric_score = func(logits.detach(), y)
                    tot_metrics[m] += metric_score * batch_size

                self._execute_callbacks('on_batch_end', X = X, y = y, logits = logits, loss = loss.item(), batch_size = batch_size, metrics = tot_metrics, is_training = self.model.training)

                n_batches += 1
                tot_samples += batch_size

        results = {}
        results['loss'] = tot_loss / tot_samples
        
        for m, tot_metric in tot_metrics.items():
            results[m] = tot_metric / tot_samples

        return results

    def fit(
        self,
        epochs,
        train_loader,
        val_loader,
        patience = None,
        metric_to_monitor = 'val_loss',
        monitor_mode = 'min',
        verbose = True
    ):
        assert monitor_mode == 'min' or monitor_mode == 'max', "invalid mode"
        
        if self.lr_scheduler and hasattr(self.lr_scheduler, 'mode') and self.lr_scheduler.mode != monitor_mode:
            self.lr_scheduler.mode = monitor_mode
            print("warning: lr scheduler mode was changed to align with monitor mode")

        if not self.resume:
            self._init_history()
        
        self.best_score = float('inf') if monitor_mode == 'min' else float('-inf')
        self.best_state = None
        
        if self.log_dir and not self.writer: #writer is closed at the end of the function, im ensuring is open at the start if someone calls fit two times in a row
            self.writer = SummaryWriter(log_dir = self.log_dir)

        start_epoch = self.current_epoch + 1
        start_time = time.time()

        self._execute_callbacks('on_fit_begin', train_loader = train_loader, val_loader = val_loader, epochs = epochs) #add other arguments if needed

        try:
            for epoch in tqdm(range(start_epoch, epochs), initial = start_epoch, total = epochs, leave = True, disable = False): #= not verbose):
                self.current_epoch = epoch
                results = {}
                self._execute_callbacks('on_epoch_begin')

                #train 
                self._execute_callbacks('on_train_begin')

                self.model.train(True)
                train_results = self._run_epoch(train_loader, verbose = verbose)
                for key, value in train_results.items():
                    results[f'train_{key}'] = value
                    if verbose and not key.startswith('grad_'):  #do not print all gradient stats to console even if verbose
                        print(f'train_{key}: {value:.4f}')
                
                self._execute_callbacks('on_train_end', results = results, train_results = train_results)

                #validation
                self._execute_callbacks('on_val_begin')

                self.model.train(False) #self.model.eval()
                val_results = self._run_epoch(val_loader, verbose = verbose)
                for key, value in val_results.items():
                    results[f'val_{key}'] = value 
                    if verbose:
                        print(f'val_{key}: {value:.4f}')

                self._execute_callbacks('on_val_end', results = results, val_results = val_results)
        
                self._handle_lr_scheduler(results, metric_to_monitor)

                self._handle_history_and_log(results, epoch) #needs epoch becouse RL class works with episodes
                
                self._handle_best(results, metric_to_monitor, monitor_mode)

                if patience is not None and self._steps_without_improv >= patience: break #handle early stop
                
                self._execute_callbacks('on_epoch_end', results = results)
            
            end_time = time.time()
            tot_time = end_time - start_time
            print(f"training time: {tot_time:.2f}") #print even if not verbose
            
            self._execute_callbacks('on_fit_end', tot_time = tot_time, best_score = self.best_score, best_state = self.best_state)
        except KeyboardInterrupt:
            if self.checkpoint_path:
                self.save(True)
        finally:
            self._cleanup_and_reload_best()         

        return self.history

    def evaluate(self, test_loader, verbose = True):

        self._execute_callbacks('on_eval_begin')

        self.model.train(False)
        test_results = self._run_epoch(test_loader, verbose = verbose)

        results = {f'test_{key}': value for key, value in test_results.items()}
        self._execute_callbacks('on_eval_end')

        if verbose:
            for key, value in results.items():
                print(f'{key}: {value:.4f}')

        return results

    def _add_states(self, states):
        if hasattr(self.criterion, 'state_dict'):
            states['criterion_state_dict'] = self.criterion.state_dict()
    
    def _load_states(self, states):
        if 'criterion_state_dict' in states and hasattr(self.criterion, 'load_state_dict'):
            self.criterion.load_state_dict(states['criterion_state_dict'])