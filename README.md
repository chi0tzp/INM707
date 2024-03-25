# INM707: Deep Reinforcement Learning



## Session 6: Introduction to DRL

### [Google Colab](https://colab.research.google.com/drive/142SLDs2LuuYQP50B3naQsrMDgy0e09Ps?usp=sharing)

### Local

#### Installation

We recommend using Python's virtual environment:

```bash
# CD to Lab6's dir
cd lab6/

# Create a virtual environment and activate it
python -m venv lab6-venv
source lab6-venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

For using the aforementioned virtual environment in a Jupyter Notebook, you need to manually add the kernel as follows:

```bash
python -m ipykernel install --user --name=lab6-venv
```



Train a Deep Q Learning (DQN) agent on the CartPole-v1 task from [Gymnasium](https://gymnasium.farama.org).

```bash
usage: main.py [-h] [-e NUM_EPISODES] [-b BATCH_SIZE] [-g GAMMA] [--eps-start EPS_START] [--eps-end EPS_END] [--eps-decay EPS_DECAY] [--tau TAU] [--lr LR]

options:
  -h, --help            show this help message and exit
  -e NUM_EPISODES, --num-episodes NUM_EPISODES
                        set number of training episodes
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        set training batch size (i.e., the number of experiences sampled from the replay memory)
  -g GAMMA, --gamma GAMMA
                        set the discount factor
  --eps-start EPS_START
                        set the initial value of epsilon
  --eps-end EPS_END     set the final value of epsilon
  --eps-decay EPS_DECAY
                        set the rate of exponential decay of epsilon (higher meaning a slower decay)
  --tau TAU             set the update rate of the target network
  --lr LR               set the learning rate
```





## Session 7: Policy- and Value-based Algorithms I (REINFORCE/SARSA)

### Google Colab: [REINFORCE](https://colab.research.google.com/drive/1627yl7ZZDu32hBKXhbcUAQUNXSR6TB41)
### Google Colab: [SARSA](https://colab.research.google.com/drive/13x6SSBBlEUyXn2omlJ_e1as8mozT_BT4#scrollTo=N69buoN0OUwj)



## Session 8: Policy- and Value-based Algorithms II (DQN)

### Google Colab: [DoubleDQN](https://colab.research.google.com/drive/1m9nCdIJJsmhin7WJrtvNgW9psibF62bl?usp=sharing)

