# aai4r-pouring-skill
On-going project for learning pouring skill.


## SPiRL on isaacgym environment
* Main train code
  - train.py
    

* Skill learning <br>
  1) Prepare demonstration dataset
  ```
  Run: skill_rl/run.py --> task_demonstration()
  ```

  2) Train skill embedding space
  ```
  Run: spirl/train.py (for pycharm debugging)
  Data Path: skill_rl/data/rollout_x.h5
  
  Logger: WandBLogger
  Model: ClSPiRLMdl <-- SkillPriorMdl
    * out = model(batch)
    * out.q     @ infer Mult.Gaussian(BaseProcessingLSTM/Linear(batch))
      -> BaseProcessingLSTM__CustomLSTM, CustomLSTMCell
    * out.p     @ fixed prior, Gaussian(out.q.mu0, out.q.log_sig0)
    * out.q_hat @ learned skill prior, prior_mdl(batch.states[:, 0])
      -> prior_mdl: Predictor__BaseProcessingNet
      -> List of FCBlock(Linear, Activation, Batchnorm)
    * out.z     @ latent variable sampling, out.q.sample() 
    * out.recon @ decoded action, decoder(out.z, learned_prior_input)
      -> List of FCBlock(Linear, Activation, Batchnorm)
  Train_Loader: RepeatedDataLoader
    * batch: [states, actions, pad_mask]
  ```
  
* Policy learning
  - spirl/rl/train.py (for pycharm debugging)
  - task_rl/run.py -> task_rl_train()
  ```
    warmup()    # gathering initial dataset and make replay buffer
      * warmup_batch <- sampler: HierarchicalSampler__Sampler
        ** out <- agent.act(obs) : FixedIntervalHierarchicalAgent__HierarchicalAgent__BaseAgent
          *** out.hl <- agent.hl_agent.act(obs) : ActionPriorSACAgent__SACAgent__ACAgent
            -> out.hl   @ hl_action(10-dim) & hl_log_prob(1-dim) 
          *** out.ll <- agent.ll_agent.act(obs & hl_action) : SACAgent__ACAgent
            -> out.ll   @ action(7-dim, motor) & log_prob(1-dim)
      * agent.add_exp(warmup_batch) : 
        ** replay_buffer.append(warmup_batch) : UniformReplayBuffer__ReplayBuffer
    
    train_epoch()
    while epoch_loop:
      * batch <- sampler: HierarchicalSampler__Sampler
      * agent.update(batch) : HierarchicalAgent__BaseAgent
        ** replay_buffer.append(batch) : UniformReplayBuffer__ReplayBuffer
        ** batch <- replay_buffer.sample()
        ** policy_out <- policy(batch.obs) : LearnedPriorAugmentedPIPolicy
                                             __PriorInitializedPolicy & LearnedPriorAugmentedPolicy
                                             __PriorAugmentedPolicy__Policy
          
          *** hl_policy_out <- hl_agent.update(batch.hl) : ActionPriorSACAgent__SACAgent__ACAgent
          -> hl_policy_out = {action(10-dim), log_prob(1-dim), dist(m.gauss, 10-dim), 
                              prior_dist(m.gauss, 10-dim), prior_divergence(1-dim)}

          # following ll policy out were not reached out during training...  
          *** ll_policy_out <- ll_agent.update(batch.ll) : SACAgent__ACAgent
          -> ll_policy_out = {action(7-dim), log_prob(1-dim), dist(m.gauss, 7-dim), 
                              prior_dist(m.gauss, 10-dim), prior_divergence(1-dim)}
        
        ** alpha_loss <- update_alpha(policy_out)
        ** policy_loss <- compute_policy_loss(batch, policy_out)
        ** target_Q
          *** policy_out_next <- policy(batch.obs_next)
          *** value_next <- compute_next_value(batch, policy_out_next)
          *** q_target <- batch.reward * scale + (1 - batch.done) * disc_fact * value_next 
          *** critic_loss <- compute_critic_loss(batch, q_target)
        
        ** update policy network
        ** update critic network
        ** update target network
  ```
  