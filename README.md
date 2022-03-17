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
      * agent.add_exp(warmup_batch) : 
      * 애초에 여기서 replay_buffer 만들 때 shape이 꼬임..
    train_epoch()     # 
    while epoch_loop:
      * batch <- sampler: HierarchicalSampler__Sampler
      * agent.update(batch) : HierarchicalAgent__BaseAgent
        ** hl_agent(hl_batch) : ActionPriorSACAgent__SACAgent__ACAgent
          *** replay_buffer.append(batch) : UniformReplayBuffer
        ** ll_agent(ll_batch) : SACAgent__ACAgent
  ```
  