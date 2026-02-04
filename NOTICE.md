**环境配置**

0. 运行miniconda3/bin中的activate；创建conda环境

1. 在conda环境中pip install uv；cd到工作空间

2. 解决 PyAV/av 构建报错（uv sync 时 av 从源码编译，需要以下依赖）

    ```shell
    conda install -c conda-forge 'ffmpeg>=7'
    conda install -c conda-forge pkg-config cython c-compiler
    ```

3. 配置环境

    ```shell
    uv sync
    uv pip install -e .
    ```

4. 下载文件
    ```shell
    gdown "https://drive.google.com/drive/folders/1YP4H8W_4Cp12oZy38Am3YABi0s4k98Ib" -O ./data --folder
    ```
    其中，网址替换成自己的数据

**TODO**

1. **数据集**（已完成）：包括state（observation.state, 自感知）和action（action），我们需要把state和action从joint改成tcp。

    state：1. 相对初始位置的位姿（pos + rot_vec, 6d * 2） 2. 夹爪距离（1d * 2） 3. 左夹爪相对右夹爪的位姿（pos + rot_vec, 6d），一共20d

    action：1. 末端执行器的位姿变化量（pos + 旋转矩阵前两列， 9d * 2）2. 夹爪距离变化量（1d * 2）

2. **修改policy**（已完成）：写一个新的vb_policy，key和维度与我们自己的features符合

3. **修改网络**：将触觉信号加入

*注：*

1. config中，state和action的维度默认相同（pi0_config.py第71行），但可以修改. 本库中已经将state_dim与action_dim解耦

2. train.py中，原先的action为32维，需要在_load_weights_and_validate函数中添加自动适配action_dim的代码（为何load出来的action dim会和load之前有不同？？）

3. model.py中，原先的tuple IMAGE_KEYs与vb_policy中设定的图像keys不同，需要修改为对应名称

**Finetune 步骤**

1. 数据格式转换
```shell
bash scripts/run_convert.sh
```

2. 计算归一化统计量
```shell
bash scripts/compute_norm_stats.sh pi05_chaoyi
```
（用uv锁定所有库的版本，避免冲突）
其中，sh文件中pi05_chaoyi需要被修改为对应的config名；config本身在src/openpi/training/config.py中修改

3. 训练
```shell
bash scripts/train.sh pi05_chaoyi
```
同理