# `launch_4gpu_reward_sweep.sh` 实验指南

## 1. 脚本用途

`launch_4gpu_reward_sweep.sh` 用于在一台多 GPU 主机上并行启动多组独立的强化学习实验，以比较不同末端执行器（EE）奖励权重。

它不是四卡 DDP，也不是四张卡共同优化一个模型。每张被选中的物理 GPU 都会启动一个独立的 `ios_train.py` 进程，并各自拥有：

- 独立的 Isaac Lab 仿真环境；
- 独立的 Actor/Critic、GRU、ContactNet 和 PPO optimizer；
- 独立的 checkpoint、TensorBoard events 和 W&B run；
- 独立的奖励权重和控制台日志。

因此，该脚本适合 reward sweep、消融实验和超参数对比。如果目标是让四张卡共同训练同一个 policy，则需要另外实现 `torchrun + NCCL + DDP + PPO 统计同步`，不能直接使用本脚本。

## 2. 当前默认实验配置

脚本当前配置如下：

| 项目 | 当前值 |
|---|---|
| Task | `Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-v0` |
| 每卡环境数 | 8192 |
| 最大迭代数 | 10000 |
| Seed | 42 |
| Logger | W&B |
| 运行模式 | Headless |
| GPU 映射 | 每个进程一张物理 GPU |

当前 reward sweep：

| 物理 GPU | Position | Orientation | PB | Reference |
|---:|---:|---:|---:|---:|
| 0 | 4.0 | 5.0 | 15.0 | 1.0 |
| 1 | 4.0 | 5.0 | 15.0 | 2.0 |
| 2 | 4.0 | 5.0 | 15.0 | 3.0 |
| 3 | 2.0 | 3.0 | 20.0 | 5.0 |

参数定义位于脚本中的数组：

```bash
#                        GPU 0  GPU 1  GPU 2  GPU 3
position_weights=(        4.0    4.0    4.0    2.0 )
orientation_weights=(     5.0    5.0    5.0    3.0 )
pb_weights=(             15.0   15.0   15.0   20.0 )
reference_weights=(       1.0    2.0    3.0    5.0 )
```

数组下标就是物理 GPU 编号。例如 `pb_weights[2]` 是物理 GPU 2 使用的 PB 权重。

## 3. 选择两张、三张或四张 GPU

实际启动哪些 GPU 由循环决定：

```bash
for gpu in 0 1 2 3; do
```

常见配置：

```bash
for gpu in 0 2; do       # 只启动物理 GPU 0、2
for gpu in 1 2 3; do     # 只启动物理 GPU 1、2、3
for gpu in 0 1 2 3; do   # 启动全部四张卡
```

没有被循环选中的 GPU 不会启动进程，其数组参数也不会被读取。参数数组可以始终保留四列。

脚本内部使用：

```bash
export CUDA_VISIBLE_DEVICES="${gpu}"
--device=cuda:0
```

这是正确映射。例如选择物理 GPU 2 后，该进程内部只看到一张卡，并将它编号为逻辑 `cuda:0`。不要把 `--device` 改成 `cuda:2`。

脚本中的 `Four-GPU` 和 `All four jobs` 是提示文字。选择少于四张卡时文字可能不准确，但不影响训练。

## 4. 启动训练

先激活能够导入 Isaac Lab 的 Conda 环境：

```bash
conda activate isaaclab_umi_on_tron
cd /data/jingchen/JC_umi-on-tron/IsaacLab_RFM/scripts/rsl_rl
./launch_4gpu_reward_sweep.sh
```

在普通终端中运行时，脚本会自动创建 detached tmux session，不需要手动先进入 tmux。脚本默认等待 60 秒进行启动存活检查；看到以下信息表示 tmux session 没有在初始化阶段立即退出：

```text
Four-GPU reward sweep started in detached tmux session: reward_sweep_<时间戳>
```

如果已经身处 tmux，脚本会直接在当前 tmux pane 内运行。若不希望使用自动 tmux，可运行：

```bash
./launch_4gpu_reward_sweep.sh --foreground
```

脚本采用 `set -euo pipefail`，错误变量、失败命令或管道错误会使 launcher 以非零状态结束。

## 5. tmux 操作

列出会话：

```bash
tmux ls
```

进入训练会话：

```bash
tmux attach -t reward_sweep_<时间戳>
```

退出查看但保持训练：按 `Ctrl-b`，松开后再按 `d`。

检查会话是否存在：

```bash
tmux has-session -t reward_sweep_<时间戳> && echo running
```

停止该次 sweep：

```bash
tmux kill-session -t reward_sweep_<时间戳>
```

停止会终止该 session 管理的训练进程。停止前应确认 checkpoint 已保存；非保存间隔内的进度可能丢失。

终端出现以下提示通常只是旧 session 已经不存在，不是新训练报错：

```text
no server running on /tmp/tmux-1000/default
```

## 6. 如何确认真正开始训练

仅看到 tmux session 被创建还不等同于 PPO 已开始迭代。应同时检查以下三项。

### 6.1 GPU 状态

```bash
watch -n 1 nvidia-smi
```

以 6144 environments 的当前实验为参考，每张 RTX 4090 稳态显存约为 18.6--18.7 GB。GPU 利用率会随 rollout、PPO update 和日志阶段波动。

### 6.2 控制台日志

```bash
tail -f \
  /data/jingchen/JC_umi-on-tron/IsaacLab_RFM/logs/rsl_rl/launcher_<时间戳>/reward_set_gpu0_<时间戳>.log
```

真正进入训练时应看到：

```text
Learning iteration N/10000
```

### 6.3 进程

```bash
ps -ef | grep '[i]os_train.py'
```

选了几张卡，就应存在几个 `ios_train.py` 进程。

## 7. 日志与 checkpoint 目录

根目录默认为：

```text
/data/jingchen/JC_umi-on-tron/IsaacLab_RFM/logs/rsl_rl
```

也可以在启动前通过 `WBC_LOG_ROOT` 修改：

```bash
export WBC_LOG_ROOT=/path/to/logs
```

一次 sweep 会产生两类目录。

### 7.1 Launcher 控制台日志

```text
launcher_<启动时间戳>/
├── reward_set_gpu0_<启动时间戳>.log
├── reward_set_gpu1_<启动时间戳>.log
├── reward_set_gpu2_<启动时间戳>.log
└── reward_set_gpu3_<启动时间戳>.log
```

这些文件记录各进程的 stdout/stderr，最适合排查 Traceback、OOM、资产转换错误和查看实时 iteration。

### 7.2 每张卡的训练目录

`ios_train.py` 会为每张卡创建独立目录：

```text
YYYY-MM-DD_HH-MM-SS_reward_set_gpuN_<启动时间戳>/
├── events.out.tfevents.*
├── model_*.pt
├── params/
│   ├── agent.yaml
│   ├── agent.pkl
│   ├── env.yaml
│   └── env.pkl
└── wandb/
```

- `model_*.pt`：checkpoint；
- `params/agent.yaml`：实际生效的 runner/PPO 配置；
- `params/env.yaml`：实际生效的环境、奖励和资产配置；
- `events.out.tfevents.*`：本地 TensorBoard 数据；
- `wandb/`：本地 W&B run 数据。

W&B 网页中会看到每张卡一个独立 run，这是预期行为，而不是四张卡合成一个 run。

实验复现时应以保存的 `params/*.yaml` 为准，而不只看启动脚本，因为 Conda editable installation、Hydra 和 CLI override 都可能改变最终配置。

## 8. 独立 USD 目录的原因

每个进程通过以下参数使用独立的机器人 USD 转换目录：

```bash
--asset_usd_dir="/tmp/IsaacLab/<时间戳>_gpuN"
```

这是必要的。多个相同 seed 的 Isaac Lab 进程若在同一秒并发转换 URDF，可能生成相同的随机临时目录名，互相覆盖 `assembly.usd`，最终出现：

```text
Unresolved reference prim path
No contact sensors added to the prim
```

`asset_usd_dir` 由 `ios_train.py` 在 Hydra 解析完成后写入配置。不要改回 Hydra 的 `env.scene.robot.spawn.usd_dir=...` override；当前 Isaac Lab 配置系统会将默认 `None` 严格识别为 `NoneType`，直接覆盖字符串会报类型错误。

## 9. 显存与环境数量经验

本机 RTX 4090 官方容量为 24 GB，`nvidia-smi` 可用约 24564 MiB。当前实测：

| 每卡环境数 | 实测/估计显存 | 建议 |
|---:|---:|---|
| 4096 | 约 14.6 GB | 余量充足 |
| 6144 | 约 18.6--18.7 GB | 当前推荐配置 |
| 7168 | 估计约 20--21 GB | 可作为上限测试 |
| 8192 | 估计约 22.8--25 GB | 可能 OOM，不建议直接四卡长训 |

RTX 5090 通常有 32 GB 显存，因此同样的 8192 environments 在 5090 上能运行，不代表 24 GB 的 RTX 4090 也具有足够峰值余量。

修改位置：

```bash
num_envs=8192
```

环境数增加不保证吞吐线性增加。如果 GPU 已接近满载，增加环境数可能只增加显存、iteration 时间和单轮 batch，而不显著提高 steps/s。修改后应先单卡短测，再扩大到全部 GPU。

## 10. 公平对比注意事项

- 对比 reward 权重时，除目标权重外应保持 task、seed、环境数、训练迭代数和代码版本一致。
- 当前各卡 seed 都为 42，便于降低实验间随机性差异。
- 不同 reward 参数会产生四个独立模型，不能将它们的梯度同步为一个 DDP 模型。
- 比较总 reward 时要谨慎：reward 权重不同会直接改变 reward 标度。优先比较位置误差、姿态误差、成功率、终止率和实机/回放表现。
- 编辑脚本不会改变已经启动的进程，只影响下一次启动。

## 11. 代码来源校验（重要）

启动前应检查 Conda 环境中的 editable installation：

```bash
python -m pip show ext_loco
python -m pip show rsl_rl
```

重点查看 `Editable project location`。训练后再检查：

```bash
grep -n 'asset_path:' \
  /data/jingchen/JC_umi-on-tron/IsaacLab_RFM/logs/rsl_rl/<run>/params/env.yaml
```

### 2026-07-18 环境路径事故与修复记录

`214347` 实验启动时，`isaaclab_umi_on_tron` 环境中的 `ext_loco` editable installation 错误指向：

```text
/home/ubuntu/Training_UMI-On-Tron/IsaacLab_RFM/source/ext_loco
```

而不是：

```text
/data/jingchen/JC_umi-on-tron/IsaacLab_RFM/source/ext_loco
```

`214347` 实验保存的 `params/env.yaml` 同样显示资产来自 `/home/ubuntu/Training_UMI-On-Tron/...`，并记录 `body_name: link6`。因此，该实验使用的是前一个 editable source 中的 `ext_loco` 环境配置，而不是 `/data/jingchen/JC_umi-on-tron` 工作区内最新的 `ext_loco` 配置。该实验已于 2026-07-18 终止，不应作为最新配置实验使用。

另一方面，`ios_train.py` 会主动把当前工作区的 `IsaacLab_RFM/rsl_rl` 插入 `sys.path`，所以自定义 runner 部分可能来自当前工作区。实验日志中必须记录这种代码来源组合，避免把结果错误归因于当前工作区全部源码。

随后使用以下方式将两个 editable package 切换到 `/data` 工作区：

```bash
python -m pip install --no-deps --no-build-isolation \
  -e /data/jingchen/JC_umi-on-tron/IsaacLab_RFM/source/ext_loco \
  -e /data/jingchen/JC_umi-on-tron/IsaacLab_RFM/rsl_rl
```

修复后的 `232018` 四卡实验已通过保存配置验收：

```text
asset_path: /data/jingchen/JC_umi-on-tron/IsaacLab_RFM/source/ext_loco/.../assembly.urdf
body_name: eef_link
foot_flat_l2: enabled
save_interval: 1000
```

截至该次修复，`pip show ext_loco rsl_rl` 的 `Editable project location` 均已指向 `/data/jingchen/JC_umi-on-tron/IsaacLab_RFM`。以后更换 Conda 环境或重新安装包时仍需重复检查，不能仅凭当前 shell 所在目录判断训练代码来源。

## 12. 日志清理原则

`.gitkeep` 用于让 Git 保留空日志目录，不参与训练，建议保留。

删除日志前先确认：

1. 目录时间戳不属于当前 tmux session；
2. 没有进程正在写入；
3. `model_*.pt` 已不再需要；
4. W&B/TensorBoard 数据无需保留；
5. 已记录必要的参数和结论。

训练目录不只是文本日志，还包含 checkpoint 和实际生效配置。删除后无法从本地恢复模型。清理时不要使用覆盖整个 `logs/rsl_rl` 的宽泛通配符，应逐个确认目标目录。

## 13. 常见故障排查

### tmux session 创建后立即消失

检查最新 launcher 目录：

```bash
tail -100 logs/rsl_rl/launcher_<时间戳>/*.log
```

搜索主要错误：

```bash
grep -nE 'Traceback|Error executing|CUDA out of memory|ValueError' \
  logs/rsl_rl/launcher_<时间戳>/*.log
```

### 只有部分 GPU 在运行

同时检查 `nvidia-smi` 和每张卡的 launcher log。历史上出现过并发 URDF 转换目录冲突；当前独立 `asset_usd_dir` 已用于解决该问题。

### CUDA OOM

降低 `num_envs`，优先回退到 6144 或 4096。不要只根据空闲状态下的显存判断，应观察进入 PPO iteration 后的稳定值和峰值。

### W&B 有四个 run

这是正常现象。当前设计是四个独立实验，每个训练进程对应一个 W&B run。

## 14. 实验日志模板

建议每次启动后复制以下模板，并以保存的 `params/*.yaml` 补全实际值。

```markdown
# Reward Sweep 实验记录

- 日期：
- 操作者：
- 主机：
- Git commit：
- Git working tree 是否干净：
- Conda 环境：
- ext_loco editable source：
- rsl_rl 实际 source：
- Task：
- 启动脚本：launch_4gpu_reward_sweep.sh
- Launch stamp：
- tmux session：
- WBC_LOG_ROOT：
- GPU 型号与数量：
- 选用物理 GPU：
- 每卡 num_envs：
- num_steps_per_env：
- max_iterations：
- save_interval：
- seed：

| GPU | Run name | Position | Orientation | PB | Reference | W&B URL |
|---:|---|---:|---:|---:|---:|---|
| 0 |  |  |  |  |  |  |
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |

## 启动验证

- [ ] tmux session 存活
- [ ] 选中的每张 GPU 均有 ios_train.py
- [ ] 各卡日志出现 Learning iteration
- [ ] 无 Traceback / OOM
- [ ] W&B run 均已创建
- [ ] params/env.yaml 与 params/agent.yaml 已保存
- [ ] asset_path/body_name 与预期代码来源一致

## 中间结果

- 记录迭代：
- Position error：
- Orientation error：
- Success/termination：
- Mean reward（仅同权重时直接比较）：
- 显存峰值：
- steps/s：

## 最终结论

- 最优配置：
- 判断依据：
- checkpoint：
- 后续动作：
- 异常与备注：
```
