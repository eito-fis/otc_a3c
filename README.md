# Unity Obstacle Tower Challenge
### This Repo
Hi! This was the 42 Robolab's work for Unity Obstacle Tower Challenge.
Parameters for each training and rendering script can be found within their
respective files.

To launch training run any of the `src/a2c/train_[Agent Name].py` scripts.<br/>For example, `python3 -m src.a2c.train_lstm --retro --render` to train an LSTM actor critic model on retro style observations from the environment.

To test an agent, run `src/a2c/eval/a2c_eval` (or  `src/a2c/eval/lstm_eval for lstm agents`).<br/>For example, ` python3 -m src.a2c.eval.lstm_eval --restore data/prierarchy_cycle_med_reg_EP15600.h5 --render`
