**环境配置**

0. 安装conda；运行miniconda3/bin中的activate；创建conda环境

    ```shell
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/miniconda3/bin/activate
    conda create -n vb python==3.11
    conda activate vb
    ```

1. 在conda环境中

    ```shell
    pip install uv
    ```

2. 解决 PyAV/av 构建报错（uv sync 时 av 从源码编译，需要以下依赖）

    ```shell
    conda install -c conda-forge 'ffmpeg>=7'
    conda install -c conda-forge pkg-config cython c-compiler
    ```

如果没有repo：

    ```shell
    cd ~
    git clone https://github.com/chaoyiL/openpi_chaoyi.git
    ```

3. 配置环境

    ```shell
    cd ~/openpi_chaoyi
    uv sync
    uv pip install -e .
    ```

4. 下载文件
    ```shell
    gdown "https://drive.google.com/drive/folders/1YP4H8W_4Cp12oZy38Am3YABi0s4k98Ib" -O ./data --folder
    ```
    其中，网址替换成自己的数据

**Finetune 步骤**

0. 修改使用的数据集：config.py 第564行 data_name 改为需要用的数据集名称；

1. 数据格式转换

```shell
bash scripts/run_convert.sh
```

2. 计算归一化统计量

```shell
bash scripts/compute_norm_stats.sh pi05_chaoyi_vitac
```
（用uv锁定所有库的版本，避免冲突）
其中，sh文件中pi05_chaoyi需要被修改为对应的config名；config本身在src/openpi/training/config.py中修改

3. 配置wandb

```shell
wandb login
```

4. 训练

```shell
bash scripts/train.sh pi05_chaoyi_vitac
```
同理

5. 下载ckpt

```shell
scp -P 22010 -i ~/.ssh/id_ed25519 -r root@194.68.245.213:~/openpi_chaoyi/checkpoints/pi05_chaoyi_vitac/my_experiment /home/liuchaoyi/openpi_chaoyi/openpi_chaoyi/checkpoints/pi05_chaoyi/
```
-P替换端口，root@194.68.245.213替换SSH over exposed TCP对应的账户和ip，后面的地址替换成自己的目标文件夹

6. 单步推理测试，给出一个随机输入，输出action chunk

```shell
bash scripts/test_single_inf.sh --config pi05_chaoyi_vitac --ckpt-dir checkpoints/pi05_chaoyi_vitac/my_experiment/50
```

action chunk 的长度可以在 config.py 中通过 action_horizon 参数修改

**TODO**

1. **数据集**（已完成）：包括state（observation.state, 自感知）和action（action），我们需要把state和action从joint改成tcp。

    state：1. 相对初始位置的位姿（pos + rot_vec, 6d * 2） 2. 夹爪距离（1d * 2） 3. 左夹爪相对右夹爪的位姿（pos + rot_vec, 6d），一共20d

    action：1. 末端执行器的位姿变化量（pos + 旋转矩阵前两列， 9d * 2）2. 夹爪距离变化量（1d * 2）

2. **修改policy**（已完成）：写一个新的vb_policy，key和维度与我们自己的features符合

3. **修改网络**（已完成）：将基于图像的触觉信号加入

4. **单步inference**（已完成）：将训练好的ckpt存储，并尝试用其推断

5. **多步inference**：尝试修改inference所读取的observation步数，尝试通过多步观测推出多步动作（如果可以的话）

*注：*

1. config中，state和action的维度默认相同（pi0_config.py第71行），但可以修改. 本库中已经将state_dim与action_dim解耦

2. train.py中，原先的action为32维，需要在_load_weights_and_validate函数中添加自动适配action_dim的代码（为何load出来的action dim会和load之前有不同？？）

3. model.py中，原先的tuple IMAGE_KEYs与vb_policy中设定的图像keys不同，需要修改为对应名称。**现已将不同模式的keys封装，只需在model文件中修改policy_type即可。**

4. 图像增强：在model.py第184行，原定的图像增强不对腕部相机进行（即wrist是否存在于key中），而只对外部相机进行。目前修改为对所有图像进行增强

5. 设置ckpt存储频率：在 config.py 中的 class TrainingConfig，有变量 save_interval，用来设置多少代存储一次ckpt

6. Git目前已经设置全局代理