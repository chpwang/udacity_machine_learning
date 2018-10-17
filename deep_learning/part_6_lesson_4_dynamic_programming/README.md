你可以通过查看该 [GitHub 代码库](https://github.com/openai/gym.git) 详细了解 OpenAI Gym。

建议花时间查看 [leaderboard](https://github.com/openai/gym/wiki/Leaderboard)，其中包含每个任务的最佳解决方案。

请参阅此[博客帖子](https://blog.openai.com/openai-gym-beta/)，详细了解如何使用 OpenAI Gym 加速强化学习研究。

---
### 安装说明

如果你想在你的计算机上安装 OpenAI Gym，建议你完成以下简单安装过程：

``` bash
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```

安装 OpenAI Gym 后，请获取经典控制任务（例如`“CartPole-v0”`)的代码：

``` bash
pip install -e '.[classic_control]'
```

最后，通过运行在 `examples` 目录中提供的[简单的随机智能体](https://github.com/openai/gym/blob/master/examples/agents/random_agent.py)检查你的安装情况。

``` bash
cd examples/agents
python random_agent.py
```

（这些说明摘自该 [GitHub 代码库](https://github.com/openai/gym) 中的自述文件）