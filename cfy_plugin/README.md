# CFY Plugin for Perturblib

## 概述

CFY (Classify then Forward Yield) 插件是一个模型无关的框架，可以为任何现有模型添加分类-前向-产出功能。该插件特别设计用于处理双重扰动场景，通过自适应MLP层维度和无缝集成，确保与原有模型的前向后向过程兼容。

## 核心设计原则

1. **模型无关性**: 可以应用到任何基础模型架构
2. **自适应MLP**: 根据输入参数自动调整层维度
3. **非侵入性集成**: 保留原始模型的前向后向流程
4. **梯度友好**: 确保反向传播过程中不会出现深度迭代问题

## 主要特性

### 🔄 自适应架构

- **动态MLP维度**: 根据模型复杂度自动调整隐藏层大小
- **层数自适应**: 智能确定最优层数以平衡性能和效率
- **组件特定配置**: 为编码器、分类器、专家网络分别优化

### 🚀 无缝集成

- **混合前向传播**: 智能路由双重扰动和其他类型输入
- **梯度保持**: 确保原始模型梯度流不受影响
- **状态保存/加载**: 完整支持模型检查点功能

### 🎯 专业化支持

- **LPM专用适配器**: 为Large Perturbation Model优化
- **交互特征工程**: 支持多种交互计算模式
- **类型感知路由**: 基于扰动数量自动分流处理

## LPM_CFY vs 原始LPM 对比分析

### 架构差异

| 特性         | 原始LPM       | LPM_CFY         | CFY Plugin     |
| ------------ | ------------- | --------------- | -------------- |
| **架构模式** | 单一路径回归  | 双路径分类+回归 | 可配置混合路径 |
| **特征工程** | 简单拼接 [3d] | 交互增强 [5d]   | 可配置交互模式 |
| **专家系统** | 无            | 固定4个专家     | 可配置专家数量 |
| **维度适配** | 固定维度      | 固定维度        | 自适应维度     |
| **模型耦合** | N/A           | 强耦合          | 松耦合插件     |

### 性能优势

1. **灵活性**: CFY插件可以应用到任何模型，不仅限于LPM
2. **可配置性**: 支持多种交互计算模式和自适应配置
3. **扩展性**: 易于添加新的专家网络和交互类型
4. **维护性**: 插件式设计便于独立测试和更新

## 近期改进详解（可直接用于讲述）

本节聚焦“这次到底改了什么、为什么这样改、效果如何、还差什么”。

### 1. 改进目标

本轮改进不只是调参，而是完成了三层升级：

1. **语义层升级**：将专家网络从“吃隐藏表示”改成“直接基于双扰动语义运算”。
2. **结构层升级**：从单一路由升级为“分类 + 四专家回归 + 融合门控/注意力”的可插拔结构。
3. **训练层升级**：补齐类别监督链路，确保分类辅助损失真正接入训练而不是空转。

### 2. 业务语义对齐（四类关系）

CFY 的四类专家语义对齐如下：

- **Additive**：接近两扰动加和效应。
- **Synergy**：优于（强于）加和预期。
- **Buffering**：弱于加和预期（缓冲/抵消一部分）。
- **Opposite**：与加和方向相反。

这一步的核心意义是：专家输出从“黑盒特征映射”转向“可解释的关系建模”。

### 3. 代码级改动总览（按文件职责）

#### 3.1 base.py：从通用 MLP 专家到语义专家 + 多头融合

主要变化：

1. **引入双扰动语义专家路径**
   - 专家不再仅依赖 hidden，改为可接收扰动对信息进行关系建模。

2. **引入 classify → experts 的联合推断结构**
   - 分类头输出类概率/类logits。
   - 四个专家并行回归。
   - 通过可学习融合（含 logit 引导、注意力映射、混合门控）合成最终预测。

3. **多头增强（保持插件可选启用）**
   - 加入 guidance/state/perturb/program/distribution 等可选分支。
   - 设计为“默认弱注入”，避免破坏原主干稳定性。

4. **关键工程修复：mask 后 ragged 张量一致性**
   - 在 masked batch 场景下，重建 perturbation_flat 与 perturbation_offset 的一致性。
   - 该修复直接改善 single/整体路径稳定性，属于高价值工程修复。

#### 3.2 adaptors.py：双扰动抽取和原模型桥接

主要变化：

1. 新增双扰动 pair 提取逻辑，支持插件语义专家输入。
2. 兼容 tuple 形式 embedded perturbations。
3. 统一 original forward 的 masked batch 处理，减少路径分叉和错位风险。

#### 3.3 training_comparison.py：训练目标与标签链路补齐

主要变化：

1. 训练目标扩展为“主回归 + 辅助分类 + 关系/分布正则”的组合方案。
2. 数据准备阶段将 co_effect_type 编码为 co_effect_type_id。
3. 标签提取逻辑优先读取 co_effect_type_id，缺失时回退 co_effect_type。

这一步解决了一个关键问题：
“类别监督可能拿不到标签”的隐形失效点，现已修复为显式优先数值标签。

### 4. 为什么这些改进是必要的

#### 问题1：专家与生物关系语义脱节

- 旧方案：expert(hidden) 更像通用回归器。
- 新方案：expert(perturb_pair, interaction) 对应明确关系语义。
- 收益：可解释性、迁移性和诊断能力更强。

#### 问题2：仅单路径回归在多关系数据上表达不足

- 旧方案：对所有关系统一回归，难以建模分布异质性。
- 新方案：先分类关系，再做关系专长回归并融合。
- 收益：将“判别”和“估计”解耦，提升多类型样本适配能力。

#### 问题3：训练辅助损失可能无效

- 旧方案：标签来源不稳定时，分类/专家监督可能空转。
- 新方案：固定优先 co_effect_type_id，保留字符串回退。
- 收益：辅助损失可控且可验证，训练信号更可靠。

### 5. 当前可讲述结果（口径建议）

建议对外口径：

1. **已经完成结构性升级**，不是简单参数搜索。
2. **整体与多个分项已具备竞争力**，但 **Buffering 仍是主要瓶颈**。
3. **已修复关键工程问题**（mask 下 flat/offset 一致性）和 **标签监督链路问题**，后续优化基线更可靠。

### 6. 风险与边界

1. Synergy 样本通常较少，分项波动会偏大。
2. 多头分支若权重注入过强，可能反向拉低主任务稳定性。
3. Buffering 与 Opposite 的决策边界更接近，易出现混淆，需要更定向监督策略。

### 7. 下一步优化建议（按优先级）

1. **优先级P0：Buffering定向优化**
   - 强化 buffering expert 监督权重（分阶段 warmup）。
   - 引入 buffering/opposite 边界约束损失。

2. **优先级P1：稀有类稳健性**
   - 对 synergy/opposite 使用 class-balanced focal 或重采样。

3. **优先级P2：可解释诊断工具**
   - 记录每批路由分布、专家激活强度、标签来源命中率，减少“看不到训练是否生效”的盲区。

### 8. 一段可直接汇报的话

“这次 CFY 不是调超参，而是做了三件关键事情：第一，把专家从隐藏特征回归改成双扰动语义回归，使 Additive、Synergy、Buffering、Opposite 关系可解释；第二，把结构升级为分类 + 四专家回归 + 门控融合，提高多关系建模能力；第三，补齐 co_effect_type_id 标签链路，确保辅助监督真实生效。工程上还修复了 masked batch 下的 flat/offset 一致性问题，这对 single 路径和整体稳定性影响很大。目前整体表现已稳定提升，下一阶段重点是继续压低 Buffering 误差。”

## 使用方法

### 1. 基础使用 - LPM模型增强

```python
from perturb_lib.cfy_plugin import LPMWithCFY
from perturb_lib.models.collection.lpm import LPM

class EnhancedLPM(LPMWithCFY, LPM):
    def __init__(self, *args, **kwargs):
        # 提取CFY配置
        cfy_config = kwargs.pop('cfy_config', {
            'num_classes': 4,
            'interaction_mode': 'elementwise',
            'adaptive_mlp_config': {
                'scale_factor': 1.2,
                'max_layers': 4,
                'encoder': {'num_layers': 3},
                'classifier': {'scale_factor': 0.8}
            }
        })

        super().__init__(*args, cfy_config=cfy_config, **kwargs)

        # 初始化CFY功能
        self.setup_cfy()

# 使用模型
model = EnhancedLPM(
    embedding_dim=128,
    hidden_dim=256,
    num_layers=3,
    # ... 其他LPM参数
)
```

### 2. 通用模型集成

```python
from perturb_lib.cfy_plugin import CFYPluginMixin

class MyModelWithCFY(CFYPluginMixin, MyBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 配置CFY功能
        self.setup_cfy(
            input_dim=self.calculate_interaction_dim(),
            hidden_dim=self.hidden_size,
            adaptive_mlp_config={
                'scale_factor': 1.0,
                'min_layers': 2,
                'max_layers': 5
            }
        )

    def extract_interaction_features(self, batch_data):
        # 实现您的特征提取逻辑
        pass

    def identify_dual_perturbations(self, batch_data):
        # 实现您的双重扰动识别逻辑
        pass
```

### 3. 自适应MLP配置

```python
# 基础配置
adaptive_config = {
    'scale_factor': 1.0,        # 全局缩放因子
    'layer_decay': 0.8,         # 层间维度衰减
    'min_layers': 1,            # 最少层数
    'max_layers': 5,            # 最多层数
    'min_hidden_dim': 64,       # 最小隐藏维度
    'max_hidden_dim': 512,      # 最大隐藏维度
}

# 组件特定配置
component_config = {
    'encoder': {
        'scale_factor': 1.2,    # 编码器略大
        'num_layers': 3,
    },
    'classifier': {
        'scale_factor': 0.8,    # 分类器较小
        'num_layers': 2,
    },
    'expert_0': {'scale_factor': 1.0},
    'expert_1': {'scale_factor': 1.1},  # 可为不同专家设置不同大小
}
```

## 技术实现细节

### 自适应MLP算法

```python
def _build_adaptive_mlp(self, input_dim, output_dim, config_key):
    # 1. 计算自适应维度
    scale = config.get('scale_factor', 1.0)
    adapted_hidden = int(base_hidden * scale)

    # 2. 确定层数
    complexity_factor = (input_dim + output_dim) / (2 * base_hidden)
    num_layers = max(min_layers, min(int(complexity_factor + 1), max_layers))

    # 3. 构建网络
    for i in range(num_layers):
        layer_scale = layer_decay ** i
        layer_dim = int(adapted_hidden * layer_scale)
        # ... 构建层
```

### 混合前向传播

```python
def hybrid_forward(self, batch_data, use_original=True):
    # 1. 识别扰动类型
    is_dual_mask, is_other_mask = self.identify_dual_perturbations(batch_data)

    # 2. 路由处理
    if is_other_mask.any() and use_original:
        # 非双重扰动 -> 原始模型
        other_preds = self._original_forward(batch_data, is_other_mask)

    if is_dual_mask.any():
        # 双重扰动 -> CFY处理
        interaction_features = self.extract_interaction_features(batch_data)
        dual_preds = self.cfy_forward(interaction_features[is_dual_mask])

    # 3. 合并结果
    return combined_predictions
```

### 梯度流保护

- **无侵入性**: 原始模型参数和计算图不受影响
- **选择性激活**: 只有双重扰动样本经过CFY路径
- **梯度连续性**: 确保所有路径的梯度正确传播
- **内存效率**: 避免不必要的张量复制和计算

## 配置选项详解

### 交互模式 (interaction_mode)

1. **elementwise**: `p1 ⊙ p2` (元素级乘积)
   - 特征: `[context, p1, p2, p1⊙p2, readout]`
   - 输入维度: 5 × embedding_dim

2. **concat**: 简单拼接
   - 特征: `[context, p1, p2, readout]`
   - 输入维度: 4 × embedding_dim

3. **bilinear**: 双线性交互
   - 特征: `[context, p1, p2, BiLinear(p1,p2), readout]`
   - 输入维度: 5 × embedding_dim
   - 额外参数: BiLinear层

### 自适应配置策略

- **保守型**: `scale_factor=0.8`, `max_layers=3` (减少过拟合)
- **标准型**: `scale_factor=1.0`, `max_layers=4` (平衡性能)
- **激进型**: `scale_factor=1.5`, `max_layers=6` (最大表达能力)

## 性能考虑

### 计算开销

- **额外参数**: 约20-30%增加（取决于配置）
- **推理时间**: 双重扰动样本约10-15%增加
- **内存使用**: 临时特征存储，影响较小

### 优化建议

1. 根据数据集大小调整 `scale_factor`
2. 使用较小的分类器 (`classifier.scale_factor < 1.0`)
3. 限制专家网络层数 (`expert_*.max_layers`)
4. 启用 dropout 防止过拟合

## 扩展和定制

### 添加新的交互模式

```python
class CustomLPMAdaptor(LPMCFYAdaptor):
    def __init__(self, *args, interaction_mode="custom", **kwargs):
        super().__init__(*args, interaction_mode=interaction_mode, **kwargs)
        if interaction_mode == "custom":
            self.custom_interaction = self._build_custom_interaction()

    def extract_interaction_features(self, batch_data):
        if self.interaction_mode == "custom":
            # 实现自定义交互计算
            interaction = self.custom_interaction(p1_emb, p2_emb)
            return torch.cat([context, p1_emb, p2_emb, interaction, readout], dim=1)
        return super().extract_interaction_features(batch_data)
```

### 自定义专家网络

```python
class SpecializedExperts(CFYMixinBase):
    def _build_cfy_components(self):
        super()._build_cfy_components()

        # 替换标准专家为专业化专家
        self.experts = nn.ModuleList([
            self._build_additive_expert(),     # 专门处理加性交互
            self._build_synergy_expert(),      # 专门处理协同作用
            self._build_buffering_expert(),    # 专门处理缓冲效应
            self._build_opposite_expert(),     # 专门处理拮抗作用
        ])
```

## 故障排除

### 常见问题

1. **CFY未初始化错误**

   ```python
   # 解决方案：确保调用setup_cfy()
   model.setup_cfy(input_dim=your_dim, hidden_dim=your_hidden_dim)
   ```

2. **维度不匹配**

   ```python
   # 检查交互特征维度
   features = model.extract_interaction_features(batch)
   assert features.shape[1] == model.input_dim
   ```

3. **梯度消失/爆炸**
   ```python
   # 调整自适应配置
   adaptive_config = {
       'scale_factor': 0.8,  # 减小网络
       'max_layers': 3,      # 减少层数
       'layer_decay': 0.9,   # 增大衰减
   }
   ```

### 调试工具

```python
# 检查模型状态
info = model.get_model_info()  # 如果实现了此方法
print(f"CFY enabled: {model.cfy_enabled}")
print(f"Components: {list(model._modules.keys())}")

# 检查前向传播
outputs = model.cfy_forward(features, return_intermediate=True)
print(f"Class probabilities: {outputs['class_probs']}")
print(f"Expert outputs: {outputs['expert_outputs']}")
```

## 贡献指南

欢迎贡献新的适配器、交互模式和优化改进！

### 开发新适配器

1. 继承 `CFYMixinBase` 或 `CFYPluginMixin`
2. 实现必要的抽象方法
3. 添加相应的测试用例
4. 更新文档和示例

### 测试

```bash
cd perturb_lib/cfy_plugin/
python examples.py  # 运行集成测试
```

## 许可证

Copyright (C) 2025 Contributors
Licensed under the Apache License, Version 2.0
