## OUT OF DATE
## WILL (?) BE UPDATED

# Usage
### * Collecting Human Replay
```
python3 -m src.human_replay --period: How many games to wait before saving memory
                            --episodes: Total amount of games to play
                            --save-obs: Whether to save images or not
                            --env-filepath: Path to environment if path isn't default
                            --input-filepath: Path to load memory from if adding onto a memory buffer
                            --output-filepath: Path to where the memory buffer will be stored
                            --floor: The floor the environment will start on
```
`python3 -m src.human_replay --period 1 --episodes 20 --save-obs --output-filepath human_input/output_file`

### Running Models
###### Specific hyperparameters like `learning rate` and `stack size` can be tuned in each respective commands `train/` file.
### * Training
```
python3 -m src.train.train_a3c --output-dir: Checkpoint and logging directory
                               --memory-dir: Memory logging directory. Memories / Images will only be saved when this flag is set
                               --restore: Model weights to restore from
                               --env-filename: Path to environment if path isn't default
                               --human-input: Path to human data. For training as a classifier before RL
                               --render: Render traning
                               --eval: Eval mode
                               --gray: Environment returns grayscale 84x84 images. For convolutions
                               --mobilenet: Environment returns 1280 embeddings. *SHOULD ALMOST ALWAYS BE SET*
```
`python3 -m src.train.train_a3c --mobilenet --render --restore PATH_TO_MODEL --output-dir data/test`

### * Multi-threaded Evaluation
```
pyton3 -m src.train.run_eval --env-filename: Path to environment if path isn't default
                             --memory-dir: Memory logging directory. Memories / Images will only be saved when this flag is set
                             --restore: Model weights to restore from
                             --render: Render evaluation
                             --gray: Environment returns grayscale 84x84 images. For convolutions
                             --mobilenet: Environment returns 1280 embeddings. *SHOULD ALMOST ALWAYS BE SET*
                             --curiosity: Set when evaluating a curiosity model
```
`python3 -m src.train.run_eval --render --mobilenet --restore PATH_TO_MODEL`
