**环境配置**

cd到工作空间

```shell
uv sync

uv pip install -e .
```

**Finetune 步骤**

*TODO:*

1. **数据集：**包括state（observation.state, 自感知）和action（action），我们需要把state和action从joint改成tcp。

    state：1. 相对初始位置的位姿（pos + rot_vec, 6d * 2） 2. 夹爪距离（1d * 2） 3. 左夹爪相对右夹爪的位姿（pos + rot_vec, 6d），一共20d

    action：1. 末端执行器的位姿变化量（pos + 旋转矩阵前两列， 9d * 2）2. 夹爪距离变化量（1d * 2）

2. **修改policy：**写一个新的vb_policy，key和维度与我们自己的features符合

*注：*

1. config中，state和action的维度默认相同（pi0_config.py第71行），但可以修改. 本库中已经将state_dim与action_dim解耦

*STEPs:*

1. 数据格式转换
```shell
uv run python data/convert_zarr_to_lerobot.py
```

2. 计算归一化统计量
```shell
uv run python scripts/compute_norm_stats.py --config-name pi05_chaoyi
```
（用uv锁定所有库的版本，避免冲突）
其中pi05_chaoyi需要被修改为对应的config名，config在src/openpi/training/config.py中修改

3. 训练
```shell
uv run python scripts/train.py pi05_chaoyi --exp-name my_experiment --overwrite
```
同理