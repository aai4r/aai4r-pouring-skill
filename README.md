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

  2) Train skill
  ```
  Run: spirl/train.py (for pycharm debugging)
  Data Path: skill_rl/data/rollout_x.h5
  
  Logger: WandBLogger
  Model: ClSPiRLMdl <-- SkillPriorMdl
    * out = model(batch)
    * out.q     # @infer Mult.Gaussian(BaseProcessingLSTM/Linear(batch))
      -> BaseProcessingLSTM <-- CustomLSTM, CustomLSTMCell
    * out.p     # @fixed prior, Gaussian(out.q.mu0, out.q.log_sig0)
    * out.q_hat # @learned skill prior, prior_mdl(batch.states[:, 0])
      -> prior_mdl: Predictor <-- BaseProcessingNet
      -> List of FCBlock(Linear, Activation, Batchnorm)
    * out.z     # @latent variable sampling, out.q.sample() 
    * out.recon # @decoded action, decoder(out.z, learned_prior_input)
      -> List of FCBlock(Linear, Activation, Batchnorm)
  Train_Loader: RepeatedDataLoader
    * batch: [states, actions, pad_mask]
  ```
  
  
  
  

  
* Policy learning
  - spirl/rl/train.py (for pycharm debugging)
  