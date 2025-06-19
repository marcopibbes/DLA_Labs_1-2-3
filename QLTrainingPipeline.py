from BaseTrainingPipeline import BaseTrainingPipeline
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque
import inspect
import copy
import os
import time
import copy
import inspect
import numpy as np
import matplotlib.pyplot as plt
import random

class QLTrainingPipeline(BaseTrainingPipeline):
    def __init__(
            self,
            policy,
            buffer_size,
            *,
            select_action_fn = None,
            **base_kwargs
        ):

        self.callbacks = {
            'on_episode_begin': [],
            'on_episode_end': [],
            'on_step_begin': [],
            'on_step_end': [],
            'on_buffer_sample': [],
            'on_target_update': []
        }

        super().__init__(model = policy, **base_kwargs)
            
        self.select_action_fn = select_action_fn if select_action_fn is not None else self._default_select_action

        self.replay_buffer = deque(maxlen = buffer_size)
        self.eps = 1.

        self.target_net = copy.deepcopy(self.policy)
        self.target_net.train(False)

    #make the aliases
    #model will become policy
    @property
    def policy(self):
        return self.model

    @policy.setter
    def policy(self, value):
        self.model = value

    #current epoch will become current episode
    @property
    def current_episode(self):
        return self.current_epoch

    @current_episode.setter
    def current_episode(self, value):
        self.current_epoch = value

    #current batch will become current step
    @property 
    def current_step(self):
        return self.current_batch

    @current_step.setter
    def current_step(self, value):
        self.current_batch = value
    ###########################################

    def _init_history(self):
        self.history = {
            "train_raw_return" : [], 
            "train_episode_length" : [], 
            "q_loss" : [],
            "avg_q_loss": [],
            "eval_avg_return": [],
            "eval_avg_episode_length": [],
        }

        for m in self.metrics.keys(): 
            self.history[f'{m}'] = []

        super()._init_history()

    def _default_select_action(self, obs):
        q_values = self.policy(obs.to(self.device))

        p = np.random.rand()
        
        if p < self.eps:  # random sampling (exploration)
            act = np.random.randint(q_values.shape[-1])
            exp = q_values[act].item() if q_values.dim() == 1 else q_values[0, act].item()
        else:  # policy sampling (exploitation)
            act = q_values.argmax(dim = -1).item()
            exp = q_values[act].item() if q_values.dim() == 1 else q_values[0, act].item()
        
        return act, exp

    def _sample_batch(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones

    def _run_episode(self, env, batch_size = 32, gamma = 0.99, maxlen = 500):
        is_training = self.policy.training
        
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype = torch.float32)

        episode_return = 0
        episode_length = 0
        episode_loss = []
        
        # Set up gradient context based on training mode
        update_grad = torch.enable_grad()
        if not self.policy.training:
            update_grad = torch.no_grad()
        
        with update_grad:
            for i in range(maxlen):
                if is_training:
                    self.current_step += 1
                    self._execute_callbacks('on_step_begin')

                act, exp = self.select_action_fn(obs)

                n_obs, reward, term, trunc, info = env.step(act)
                n_obs = torch.tensor(n_obs, dtype = torch.float32)
                
                if term or trunc:
                    break

                #only add to replay buffer and train during training mode
                if is_training:
                    self.replay_buffer.append((obs, act, reward, n_obs, term or trunc))

                    #only train if we have enough samples in replay buffer
                    if len(self.replay_buffer) >= batch_size:
                        states, actions, rewards, next_states, dones = self._sample_batch(batch_size)
                        states = states.to(self.device)
                        actions = actions.to(self.device)
                        rewards = rewards.to(self.device)
                        next_states = next_states.to(self.device)
                        dones = dones.to(self.device)

                        self._execute_callbacks('on_buffer_sample', states = states, actions = actions, rewards = rewards, next_states = next_states, dones = dones)

                        current_q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                        with torch.no_grad():
                            next_q_values = self.target_net(next_states).max(1)[0]
                            target_q_values = rewards + (gamma * next_q_values * ~dones)
                        
                        loss = nn.MSELoss()(current_q_values, target_q_values)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        if loss is not None:
                            episode_loss.append(loss.item())
                    
                    self._execute_callbacks('on_step_end', action = act, reward = reward, next_obs = n_obs, done = term or trunc)

                obs = n_obs
                episode_return += reward
                episode_length += 1

        return episode_return, episode_length, episode_loss

    def evaluate(self, env, eval_episodes, maxlen = 500):
        self._execute_callbacks('on_eval_begin', env = env, n_eval_episodes = eval_episodes, maxlen = maxlen)

        self.policy.train(False) 

        self.old_eps = self.eps 
        self.eps = 0.  # disable exploration during evaluation

        total_rewards = []
        episode_lengths = []

        for _ in range(eval_episodes):
            episode_return, episode_length, _ = self._run_episode(env, maxlen = maxlen)
            total_rewards.append(episode_return)
            episode_lengths.append(episode_length)

        avg_reward = np.mean(total_rewards) if total_rewards else 0.0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
        
        self._execute_callbacks('on_eval_end', 
                                avg_reward = avg_reward, 
                                avg_length = avg_length,
                                eval_rewards = total_rewards, 
                                eval_lengths = episode_lengths)

        self.eps = self.old_eps  # restore exploration rate

        return avg_reward, avg_length

    def dqn(
        self,
        env,
        n_episodes,
        batch_size = 32, 
        gamma = 0.99,
        eps_decay = 0.9999,
        min_eps = 0.001,
        patience = None,
        metric_to_monitor = 'eval_avg_return',
        monitor_mode = 'max',
        eval_every_n_episodes = 1,
        eval_episodes = 1,
        target_update_frequency = 10,
        maxlen = 500,
        verbose = False
    ):
        assert monitor_mode in ['min', 'max'], "invalid monitor_mode"
        if eval_every_n_episodes is not None and eval_every_n_episodes <= 0:
            raise ValueError("eval_every_n_episodes must be None or a positive integer.")
        
        #bunch of warnings 
        if self.lr_scheduler and hasattr(self.lr_scheduler, 'mode') and self.lr_scheduler.mode != monitor_mode:
            self.lr_scheduler.mode = monitor_mode
            print("warning: lr scheduler mode was changed to align with monitor mode")

        if patience and eval_every_n_episodes > patience and metric_to_monitor in ['eval_avg_return', 'eval_avg_episode_length']:
            print("warning: patience is less than the evaluation frequency, therefore early stopping will be performed at the first evaluation,\nI suggest to use patience = actual_patience * eval_every_n_episode")
            
        if self.lr_scheduler and hasattr(self.lr_scheduler, 'patience') and self.lr_scheduler.patience < eval_every_n_episodes and metric_to_monitor in ['eval_avg_return', 'eval_avg_episode_length']:
            print("warning: lr scheduler patience is less than the evaluation frequency, therefore lr may converge to 0. too fast,\nI suggest to use patience = actual_patience * eval_every_n_episode")

        if not self.resume:
            self._init_history()
            self.eps = 1. #we always want to start with full exploration

        self.best_score = float('inf') if monitor_mode == 'min' else float('-inf')
        self.best_state = None

        if self.log_dir and not self.writer:
            self.writer = SummaryWriter(log_dir = self.log_dir)

        start_episode = self.current_episode + 1
        start_time = time.time()

        self._execute_callbacks('on_fit_begin', env = env, n_episodes = n_episodes, gamma = gamma, maxlen = maxlen)

        try:
            for episode in tqdm(range(start_episode, n_episodes), initial = start_episode, total = n_episodes, leave = True, disable = False):
                self.current_episode = episode
                results = {}

                self._execute_callbacks('on_episode_begin')

                #train
                self.policy.train(True)
                episode_return, episode_length, episode_loss = self._run_episode(env, batch_size, gamma, maxlen)
                results['train_raw_return'] = episode_return
                results['train_episode_length'] = episode_length
                results['q_loss'] = episode_loss[-1] if episode_loss else 0.0
                results['avg_q_loss'] = np.mean(episode_loss) if episode_loss else 0.0

                #update eps
                self.eps = max(self.eps * eps_decay, min_eps)

                #update target
                if episode % target_update_frequency == 0:
                    self.target_net.load_state_dict(self.policy.state_dict())
                    self._execute_callbacks('on_target_update')

                #evaluate
                results['eval_avg_return'] = np.nan
                results['eval_avg_episode_length'] = np.nan

                if 'eval_avg_return' in self.history and self.history['eval_avg_return']:
                    results['eval_avg_return'] = self.history['eval_avg_return'][-1]
                if 'eval_avg_episode_length' in self.history and self.history['eval_avg_episode_length']:
                    results['eval_avg_episode_length'] = self.history['eval_avg_episode_length'][-1]

                if eval_every_n_episodes is not None and ((episode + 1) % eval_every_n_episodes == 0 or episode == n_episodes - 1 or episode == 0):
                    avg_eval_reward, avg_eval_length = self.evaluate(env, eval_episodes, maxlen)

                    results['eval_avg_return'] = avg_eval_reward if avg_eval_reward is not None else np.nan
                    results['eval_avg_episode_length'] = avg_eval_length if avg_eval_length is not None else np.nan

                    if verbose:
                        print(f"avg_reward: {avg_eval_reward:.2f}, avg_length: {avg_eval_length:.2f}")

                self._handle_lr_scheduler(results, metric_to_monitor)

                self._handle_history_and_log(results, episode)
                
                self._handle_best(results, metric_to_monitor, monitor_mode)
 
                if patience is not None and self._steps_without_improv >= patience: break #handle early stop
                
                self._execute_callbacks('on_episode_end', results = results)

            end_time = time.time()
            tot_time = end_time - start_time
            print(f"training time: {tot_time:.2f}s")

            self._execute_callbacks('on_fit_end', tot_time = tot_time, best_score = self.best_score, best_state = self.best_state)

        except KeyboardInterrupt:
            if self.checkpoint_path:
                self.save(interrupted = True)
        finally:
            self._cleanup_and_reload_best()         

        return self.history

    fit = dqn

    def _add_states(self, states):
        states['eps'] = self.eps
        states['target_state_dict'] = self.target_net.state_dict()
        states['replay_buffer'] = list(self.replay_buffer)

    def _load_states(self, states):
        self.eps = states['eps']
        self.target_net.load_state_dict(states['target_state_dict'])
        self.replay_buffer = deque(states['replay_buffer'], maxlen = self.replay_buffer.maxlen)