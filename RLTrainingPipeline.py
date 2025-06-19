from BaseTrainingPipeline import BaseTrainingPipeline
import torch
import torch.nn as nn
from torch.distributions import Categorical
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


class RLTrainingPipeline(BaseTrainingPipeline):
    def __init__(
            self,
            policy,
            *,
            select_action_fn = None,
            **base_kwargs
        ):

        self.callbacks = {
            'on_episode_begin': [],
            'on_episode_end': [],
            'on_step_begin': [],
            'on_step_end': []
        }

        super().__init__(model = policy, **base_kwargs)
            
        self.select_action_fn = select_action_fn if select_action_fn is not None else self._default_select_action

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
            "policy_loss" : [],
            "eval_avg_return": [],
            "eval_avg_episode_length": []
        }

        for m in self.metrics.keys(): 
            self.history[f'{m}'] = []

        super()._init_history()

    def _default_select_action(self, obs):
        
        logits = self.policy(obs.to(self.device))
        dist = Categorical(logits = logits) #this implies that policy does not need/have to pass probabilities, maybe change into probs = ?
        act = dist.sample()

        #greedy exploration
        #act = torch.argmax(logits, dim = -1)
        #act = torch.tensor(act, dtype=torch.long).to(self.device)

        #epsilon greedy exploration
        #if np.random.rand() < eps (1.e-2):
        #   act = np.random.randint(0, self.policy.action_space.n)
        #else:
        #   act = torch.argmax(logits, dim = -1)
        #act = torch.tensor(act, dtype=torch.long).to(self.device)

        #simulated annealing
        #if np.random.rand() < eps (1.e-2):
        #   act = np.random.randint(0, self.policy.action_space.n)
        #else:
        #   act = torch.argmax(logits, dim = -1)
        #act = torch.tensor(act, dtype=torch.long).to(self.device)
        #eps = eps * decay_rate
        
        log_prob = dist.log_prob(act)

        return (act.item(), log_prob.reshape(1))

    def _compute_returns(self, rewards, gamma):
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        return np.array(returns).copy()

    def _run_episode(self, env, maxlen = 500):

        obss_list = [] 
        acts_list = []
        log_probs_list = []
        rewards_list = []

        obs, info = env.reset()

        update_grad = torch.enable_grad() #don't know whats the cost of the function but could be brought out if need to optimize
        if not self.policy.training: 
            update_grad = torch.no_grad()

        with update_grad:
            for i in range(maxlen):
                self.current_step = i
                obs_tensor = torch.tensor(obs, dtype = torch.float32).to(self.device)

                self._execute_callbacks('on_step_begin', obs = obs_tensor, info = info, current_step = self.current_step)

                act, log_prob = self.select_action_fn(obs_tensor)

                obss_list.append(obs_tensor)
                acts_list.append(act)
                log_probs_list.append(log_prob)

                obs, reward, term, trunc, info = env.step(act)
                rewards_list.append(reward)

                for m, func in self.metrics.items(): #not really sure which metrics would fit and therefore if i want to keep this here
                    self.history[m].append(func(obs, act, reward, term, trunc, info))

                self._execute_callbacks('on_step_end',
                                        obs = obs,
                                        info = info,
                                        act = act,
                                        log_prob = log_prob,
                                        reward = reward,
                                        term = term,
                                        trunc = trunc,
                                        current_step = self.current_step)

                if term or trunc:
                    break

        return obss_list, acts_list, torch.cat(log_probs_list) if log_probs_list else torch.empty(0), rewards_list

    def evaluate(self, env, eval_episodes, maxlen = 500):
        self._execute_callbacks('on_eval_begin', env = env, n_eval_episodes = eval_episodes, maxlen = maxlen)

        self.policy.train(False) #for evaluation during reinforcement this is called outside of the function too to uniform to SLTrainingPipeline

        total_rewards = []
        episode_lengths = []

        for _ in range(eval_episodes):
            _, _, _, rewards = self._run_episode(env, maxlen)
            total_rewards.append(sum(rewards))
            episode_lengths.append(len(rewards))

        avg_reward = np.mean(total_rewards) if total_rewards else 0.0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
        
        self._execute_callbacks('on_eval_end', 
                                avg_reward = avg_reward, 
                                avg_length = avg_length,
                                eval_rewards = total_rewards, 
                                eval_lengths = episode_lengths)

        return avg_reward, avg_length

    def reinforce(
        self,
        env,
        n_episodes,
        gamma = 0.99,
        baseline = None, 
        maxlen = 500,
        patience = None,
        metric_to_monitor = 'eval_avg_return',
        monitor_mode = 'max',
        verbose = True,
        eval_every_n_episodes = 1,  #N
        eval_episodes = 1           #M
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

        self.best_score = float('inf') if monitor_mode == 'min' else float('-inf')
        self.best_state = None

        if self.log_dir and not self.writer:
            self.writer = SummaryWriter(log_dir = self.log_dir)

        start_episode = self.current_episode + 1
        start_time = time.time()

        self._execute_callbacks('on_fit_begin', env = env, n_episodes = n_episodes, gamma = gamma, baseline = baseline, maxlen = maxlen)

        try:
            for episode in tqdm(range(start_episode, n_episodes), initial = start_episode, total = n_episodes, leave = True, disable = False): #not verbose):
                self.current_episode = episode
                results = {} 
                self._execute_callbacks('on_episode_begin')

                #training
                self._execute_callbacks('on_train_begin')
                
                self.policy.train(True)
                obss, acts, log_probs, rewards = self._run_episode(env, maxlen = maxlen)

                if not rewards: #episode might have ended immediately or had no rewards
                    policy_loss = torch.tensor(0.0, device = self.device)
                    results['train_raw_return'] = 0.0
                    results['train_episode_length'] = len(obss)
                else:
                    returns = self._compute_returns(rewards, gamma)
                    returns_tensor = torch.tensor(returns, dtype = torch.float32).to(self.device)

                    base_returns = returns_tensor

                    if baseline: base_returns = baseline(returns_tensor, obss)
                    else: base_returns = returns_tensor - returns_tensor.mean() 

                    self.optimizer.zero_grad()
                    policy_loss = (-log_probs * base_returns).mean() if log_probs.numel() > 0 else torch.tensor(0.0, device = self.device)
                    
                    if policy_loss.requires_grad: 
                        policy_loss.backward()
                        self.optimizer.step()

                    results['train_raw_return'] = sum(rewards)
                    results['train_episode_length'] = len(obss)
                
                results['policy_loss'] = policy_loss.item()
                
                self._execute_callbacks('on_train_end')

                #evaluating

                self._execute_callbacks('on_val_begin') #its not validation stage but kind of, no?

                self.policy.train(False)

                #since default value to monitor is eval_avg_return we cannot initialize it, lr scheduling and best states need it
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

                self._execute_callbacks('on_val_end', env = env, eval_every_n_episodes = eval_every_n_episodes, episode = episode, n_episodes = n_episodes)

                self._handle_lr_scheduler(results, metric_to_monitor)

                self._handle_history_and_log(results, episode)
                
                self._handle_best(results, metric_to_monitor, monitor_mode)

                if patience is not None and self._steps_without_improv >= patience: break #handle early stop
                
                self._execute_callbacks('on_episode_end', obss = obss, acts = acts, log_probs = log_probs, rewards = rewards, results=results)

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

    fit = reinforce #alias
