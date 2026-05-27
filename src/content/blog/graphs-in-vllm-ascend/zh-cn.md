---
title: "从 vLLM Ascend 的图模式聊开去：有限视野下的初窥门径"
description: "从官方图模式文档出发，先看 Ascend 在图模式体验背后补了哪些工程差异。"
pubDate: 2026-04-10
image: ""
slugId: "graphs-in-vllm-ascend"
category: "vLLM"
draft: false
---

- [第一篇：有限视野下的初窥门径](/blog/graphs-in-vllm-ascend/)
- [第二篇：扩大视野，来聊聊编译](/blog/graphs-in-vllm-ascend-1/)
- [第三篇：理解全局后，再看设计方案](/blog/graphs-in-vllm-ascend-2/)
- [第四篇：进入深水区，回看那些“不得已”](/blog/graphs-in-vllm-ascend-3/)
- [第五篇：跃出水面，高维视角下的回顾](/blog/graphs-in-vllm-ascend-4/)

# 按“图”索骥：从 vLLM Ascend 的图模式聊开去

## 太长不看

本文大致分五段：第一章先在 vLLM 官方图模式文档的视角下，解释在 Ascend 硬件上额外做了哪些工程补偿；第二章转去讲 `torch.compile`、Inductor、TorchAir 和 `npugraph_ex` 这些编译路径；第三章回到具体设计，拆开图模式里的动态性、分桶、不同 graph mode 与性能权衡；最后两章再把视角抬高，回头看这些技术细节背后的生态问题。

如果你想了解一些通用的知识和概念，那么看第一章就行；想了解编译的应用？看第二章；想知道具体的设计细节，第三章写得还算详细；第四章讲了框架之外的来龙去脉，而第五章则是扣题，真正的聊开去。

## 有限视野下的初窥门径
之前我也写过类似的技术文档，每当写到“第一章节”，蹦进我脑中的往往就是一些“下定义”的话题，像什么“图是什么”，“图的意义”等等，如同掉进兔子洞一般，开始追逐这些“正确但催眠”的答案。这样只会让文章的范围无形中扩大，让大家在阅读时不知所措。

**所以我们先将问题限定到足够小：** 假设我们目前视野有限，使用的语言是 Python，框架是 PyTorch，只讨论 vLLM 中图模式的方案，并不关心 PyTorch 层以下的调用链路。而且我们已经有一个模糊的印象，**图模式能够通过“一次下发多个任务”减少 CPU 下发耗时，提升推理性能**。

换言之，我们**熟悉且仅熟悉**这两篇官方文档：

- [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [CUDA Graphs in vLLM](https://docs.vllm.ai/en/latest/design/cuda_graphs/)

**注意，如果咱们不看上边儿这官方文档，那么后续的内容可能会不好理解，且沟通起来容易出现背景和预期不统一的问题，这样本文的价值也就大打折扣了。**

在这个假设前提之下，我们便可以好好的来聊聊 vLLM 中为图模式做的工程化设计，以及 vLLM Ascend 中额外新增的一些修改。

### 一切都为了“无感”
假设大家已经对图模式以及 vLLM 中使用有了模糊的了解，那说到 vLLM Ascend 里的工作：**如果做的够好，那么大家使用起来应该完全感受不到和 vLLM 中的差异。**

目前我们做的还是不错的，现在 vLLM Ascend 中使用图模式，已经非常接近上游 vLLM 的使用体验了，如果不追求极致优化，那么默认的配置就已经可以满足大多数需求。而关于一些很常见的问题，例如：
1. vLLM Ascend 怎么开启图模式？怎么验证图模式开启没有？
2. 图模式中的 `PIECEWISE` / `FULL` / ... 模式都是什么意思？怎么设置？
3. 不同的模型系列都支持 `PIECEWISE` / `FULL` 吗？怎么选择？
4. ...

这些使用层面的问题，可以说 95% 以上都可以在 vLLM 官方文档中可以找到对应的答案：[CUDA Graphs in vLLM](https://docs.vllm.ai/en/latest/design/cuda_graphs/)，vLLM Ascend 在这些问题中几乎是“隐形”的。

**如果你的问题可以在官方文档中得到解释，恭喜！下面的技术细节对于你来说可能无关紧要了。**

要是不乐意看英文文档，也可以试试直接向 DeepWiki 驱动的知识库进行提问，vLLM 和 vLLM Ascend 仓库均已支持该功能，分别在 [DeepWiki
vllm-project/vllm](https://deepwiki.com/vllm-project/vllm) 和 [DeepWiki vllm-project/vllm-ascend](https://deepwiki.com/vllm-project/vllm-ascend) 中进行提问即可。

### “到底在忙些什么？”

> “啊既然使用起来都差不多，那这么些个图模式相关的开发工作和 PR 到底是在忙些什么？”

诶，正是因为水面下是脚蹼拼命划，水面上小鸭才叫“嘎嘎”。

**我们一大部分工作量，本质上都在解决同一个问题：当下层软件栈 / 硬件的行为与外部生态出现不一致时，如何通过工程手段，抹平这种差异。**

**差异？有什么差异？** 所以接下来我们就要先有个大概的范围认知，看看“底层硬件 / 技术栈 / 各种实现”到底还有哪些坑：

1. 算子接口的参数定义与校验规则：

    **这一项可以说是多个问题的终极根源**，以至于我必须把它放在第一个来说，我们以 PFA/IFA 算子（也即 `torch_npu.npu_fused_infer_attention_score` 接口，通常我们称其为 `FIA`）为例。

    - 要求来自 HOST / CPU 的输入参数：

        在 CUDA Graph 的设计下，我们习惯的操作模式是：

        1. **捕获（capture）阶段**：定义好计算图，Device 记住所有的 Kernel Launch 序列和参数指针。
        2. **重放（replay）阶段**：我们只需要修改 Device Tensor 中的数值（比如更新 `token_ids` 与 `positions`）。
        3. **核心前提**：Host 侧不做任何复杂的逻辑计算，Host 仅仅是负责“按下重放键”。
    
        但是不同于 vLLM 原生的 Attention 实现（所有的可变参数均以 Device Tensor 形式传入），vLLM-Ascend 中的 FIA 接口要求 `actual_seq_lengths` 与 `actual_seq_lengths_kv` 两个参数（分别来自 vLLM 中的 `query_start_loc` 与 `seq_lens`，也就是前面说的 Device Tensor）必须是 Host List。

        这会带来什么问题呢？

        **Host List 的存在，破坏了 CUDA Graph 的静态假设。**

        让我们思考一下，一张已经捕获好的 CUDA Graph / ACL Graph，是不是理论上只会在 `graph.replay()` 时获取一次信息？而且这些信息都来自于提前设置好的 Device Memory 地址。所以在重放时，对于图内的算子来说它们没有办法获取到最新的 `actual_seq_lengths` 与 `actual_seq_lengths_kv` 信息。**那么我们最后模型输出出来的结果自然就会是重复的文本，因为图中只有捕获时的“过时信息”。**

    - 参数之间的校验：

        对于 FIA 来说，当其输入张量的排布为 `[T, N, D]`（参考[参数说明](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_fused_infer_attention_score.md#参数说明)）存在以下约束（参考[约束说明](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_fused_infer_attention_score.md#约束说明)）：

        > `actual_seq_lengths` 和 `actual_seq_lengths_kv` 必须传入，且以该入参元素数量作为Batch值（注意入参元素数量要小于等于4096）。该入参中每个元素的值表示当前Batch与之前所有Batch的Sequence Length和，因此后一个元素的值必须大于等于前一个元素的值；

        或许文档有一些抽象，我们来看一个算子报错会明显一些：

        ```
        When layout is TND, queryT(8) must be equal to the last element of actualSequenceLengthQ(5)[FUNC:CheckFAISeqlenDataInTND][FILE:fused_infer_attention_score_tiling.cpp][LINE:904]
        ```

        这下好理解了，也就是说存在类似这样一个校验（如果有感兴趣的话可以直接看[源代码](https://gitcode.com/cann/ops-transformer/blob/master/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp)，发挥开源社区的优势！）：

        ```python
        assert actualSequenceLengthQ[-1] == queryT.shape[0]
        ```

        尽管初看起来，这个校验可以说正常，无害，且合理，**但实际上在图模式中，它造成了很多次严重的问题**：原因就在于**这样的校验在 vLLM 的 Attention 算子中是没有的**，在图内流转的张量，第 0 维形状全是被填充到 `cudagraph_capture_sizes` 中的固定数值。那有人就要说，“啊，接口有差异这不很正常吗？不然你做框架适配是在做什么东西？”，当然没问题，稍后我们会来复盘为什么这样一个校验可能多次造成困扰，有哪些历史遗留问题，以及我们能从中吸取什么经验教训。

2. 捕获占用的资源与流规格的设计

    与 CUDA Graph 可以捕获任意数量的图不同，**ACL Graph 每捕获一张图都会占用一个流（Stream）资源**。然而，受限于驱动层限制，硬件可用的流资源总量有限，910 系列为 2048 条，310 及 710 系列据说为 1024 条，未经实测。需要注意的是，这几千条流是由整个软件栈共享的，不仅是上层的推理框架 vLLM，底层的 PyTorch 和 CANN 组件也都在“瓜分”这些有限的资源。

    看到这里大家可能会觉得，“哦好像还好啊？听起来两千个也挺多，难不成我们会需要捕获几千个图吗？”，诶，**还真是！** 不过我们先在这里卖个关子，至于为什么在推理场景中会触及到这个规格上限，我们先在此埋个伏笔。直接铺开讲可能会略显琐碎，等后文介绍完基础概念后，我们再回过头来深入拆解这个问题，便于理解，事半功倍！

3. 编译后端的支持完善程度

    写到这里的时候我突然想起来，咱们漏了一个关键背景： `torch.compile` 以及它在 vLLM 图模式中的定位，这一点确实是疏忽。这两者的联系并非那么紧密，完全可以各自单独使用的，所以之前我没有找到好的话头来提起。**别着急，我们马上在下一章就会讲到编译。**

    当前在昇腾硬件上，对于 `triton` 这样的编译后端（在编译链路中扮演多重角色，此处我们仅讨论其作为 Inductor 后端的功能）支持还并不完善。**vLLM Ascend 在很长一段时间是没有后端可用的（严格来说有 `npu_backend`，不过那是 TorchAir，暂不讨论）**，在这种情况下，我们想要对 `torch.compile` 出来的 FX 图进行高效的自动化转换与优化是有些困难的（并非不能做，现在已经实现了），这导致特性开发者被迫在模型定义脚本中硬编码性能优化逻辑。这种做法的弊端显而易见：每个模型都需要进行“霰弹式修改”，优化经验难以复用和迁移。
    
    此外，手动优化的另一大难题是**控制流**。记得我们之前说的吗？图模式不喜欢 `if-else` 这种动态的行为。如果拥有完善的 Inductor 后端，我们可以在 compile 阶段通过算子下发或图改写来拦截并处理控制流问题，同时针对不同的 Shape 生成对应的 FX 图，从而平衡通用性与执行效率。

写到这里我长舒一口气，终于在尽可能不扯远的情况下，写完了这一部分，尝试解答了“vLLM Ascend 中的额外设计来源”这个问题：**所有的“特殊设计”，最终都可以溯源到以上这几种差异，也就是：**
1. 算子接口以及内部校验带来的问题；
2. 硬件规格与 ACL Graph 本身设计与 Nvidia 生态的不同；
3. 软件栈依赖的缺失或者成熟度不够。

回头一看，为了解答这个问题，我们可能没头没脑地又引入了更多更深的概念，思之不免令人发笑，希望不要阻止大家继续往下看才好。

### 在官方文档上再进一步
在继续深入那些令人生畏的具体代码之前，我们还可以先确认下哪些东西是**不需要**我们操心的。

如果我们已经看过了之前提到的官方文档（[CUDA Graphs in vLLM](https://docs.vllm.ai/en/latest/design/cuda_graphs/)），那我们应该对其中的几个基础组件已经有了认知，我们来简单回顾一下：

官方文档中详细解释了 `CUDAGraphWrapper` / `CudagraphDispatcher` / `CUDAGraphMode` / `BatchDescriptor` 四个组件的设计，并且提供了下图来可视化调用流程：
![图模式路径](https://docs.vllm.ai/en/latest/assets/design/cuda_graphs/current_design.png)

**好消息是：在 vLLM-Ascend 中，这套流程的“骨架”被完整保留了下来。**

是这套抽象层足够稳健，所以我们才能直接复用吗？不，回想一下我们在上一节怎么说的？

> 所有的“特殊设计”，最终都可以溯源到以上差异

**啊哈！原来是因为这四个组件基本上都是 Python 原生的代码语法，甚至 PyTorch 的含量都不高，所以我们才能直接复用！**

唯一值得提一下的区别，是 `NPUModelRunner` 中对于 `CudagraphDispatcher` 的使用方法，稍微和 `GPUModelRunner` 中有一些不同：针对 `num_tokens` 的填充逻辑（这个填充逻辑会在后面详细展开，先不在这里岔开话题）有不同。

> **关于 V2 Model Runner：** 在本文撰写时，vLLM 的 V2 Model Runner 已经开始落地，但官方设计文档尚不完备，接口也可能随时调整。这里简单提几个结论供大家留意：
> 
> 1. V2 为主模型的 **FULL** 图模式引入了显式的管理，`ModelCudaGraphManager`，将 capture / dispatch / replay 收口到一处管理；
> 2. **PIECEWISE** 则没有被推倒重来，仍沿用 wrapper + forward\_context + batch descriptor 这套机制；
> 3. 此外，target model 和 EAGLE draft model 在 V2 中也各自有了清晰的 graph owner（`EagleCudaGraphManager`）。
>
> 如果你已经在跟进 V2 的代码，留意这些变化即可，如果还没有，先按本文介绍的 V1 体系理解，不影响全局认知。
