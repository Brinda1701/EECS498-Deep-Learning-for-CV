# 深度学习基础培养阶段：个人总计划与执行路线图 🚀

> **节点目标**：完成深度学习基础培养，熟练掌握神经网络实现、训练、性能调优与训练数据准备，通过编程考核，顺利进入正式科研训练。

---

## 📌 导师原始要求与指导 (Peng's Directives)

> 💬 **彭老师要求原文**：
> 
> 接下来是深度学习基础培养阶段。
> 
> 你先完成一个课程的学习。学习《Deep Learning for Computer Vision》课程（完成里面编程作业即可。懂得做编程作业，也就懂课程内容了。在GitHub上维护一个repo，存放作业）
> 
> 课程链接: https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/schedule.html
> 
> 还有简单看看Deep Learning的Notebook：https://uvadlc-notebooks.readthedocs.io/en/latest/ 网站里的Deep Learning 1 (PyTorch)部分。
> 
> 你在钉钉上每周一给我同步自己的进度和下一周的计划。如果遇到节假日就skip。
> 
> 等这一个课程学完，有一个编程考核，如果考核通过，就能开始接触正式的科研训练。
> 
> 如果对哪些算法或哪些代码有问题，都可以向我提问，比较建议先谷歌搜索或者问deepseek。
> 
> 同时仔细学习两个编程技术：
> 1. AI Coding入门指南：https://pengsida.notion.site/AI-Coding-2f83fe292ff18059a970fd9323b782a9
> 2. 《调试九法》这本书：https://www.notion.so/pengsida/debug-1b69debf803a4c268fc8a09a9a748bbf
> 
> 这本书里面的思想在平时写代码和做实验都非常有用，是我一直以来做科研非常需要的一本书，对我来说是一本神书。
> 
> 这本书的中文翻译让人读起来比较费劲，可以直接看我写的总结。相信这本书列出的九个原则会在你的编程和科研中经常用到。
> 
> 还有学习这门课程，这个课程不用在节点1学完：https://pengsida.net/games003/
> 
> 如果你课程学完了，找我说一下。我会安排接下来的学习内容。

---

## 🛠️ 任务拆解与学习路线图

### 一、 工具与编程方法论（实战前置与全程贯穿）

- [ ] **1. 精读方法论指南**
  - **《调试九法》总结**（[Notion 链接](https://www.notion.so/pengsida/debug-1b69debf803a4c268fc8a09a9a748bbf)）：认真领会九个调试原则，在后续 EECS498 的所有 Bug 排查中刻意练习，并记录调试日志。
  - **AI Coding 入门指南**（[Notion 链接](https://pengsida.notion.site/AI-Coding-2f83fe292ff18059a970fd9323b782a9)）：掌握如何将 AI（DeepSeek / Cursor 等）作为代码助手，提升开发与 Debug 效率。
- [ ] **2. 基础框架与 Notebook**
  - 学习 [UVA Deep Learning 1 (PyTorch)](https://uvadlc-notebooks.readthedocs.io/en/latest/) 基础部分，跑通 Tensor 运算与 Autograd 计算图。
- [ ] **3. GitHub 专属仓库运维**
  - 维持专属仓库 `EECS498-Deep-Learning-for-CV`，按规范提交代码、编写 README 进度和 Debug Log。

---

### 二、 核心课程与编程作业攻坚（EECS 498/598）

**课程主页**：[EECS 498/598 WI2022 Schedule](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/schedule.html)  
**核心策略**：以**编程作业（Assignments）**为驱动，做懂作业即搞懂课程。

- [ ] **Assignment 1: PyTorch 101 & k-NN**
  - 完成 PyTorch 基础张量操作。
  - 实现 k-Nearest Neighbors 分类器及交叉验证。
- [ ] **Assignment 2: Linear Classifiers & Two-Layer Neural Networks**
  - 线性分类器（SVM, Softmax loss）实现。
  - 从零手动实现双层神经网络的前向传播与反向传播（Backpropagation）。
- [ ] **Assignment 3: Fully-Connected Nets, Optimization & CNNs**
  - 实现全连接网络与常用优化器（SGD, Momentum, RMSProp, Adam）。
  - 手写实现卷积神经网络（CNN）基础层。
- [ ] **Assignment 4: Batch Normalization, Dropout & PyTorch ResNet**
  - 实现 Batch Normalization 与 Dropout 防过拟合层。
  - 使用 PyTorch 框架搭建 ResNet 残差网络。
- [ ] **Assignment 5: Object Detection & Image Captioning**
  - 单阶段目标检测器（YOLO 机制）。
  - 基于 Vanilla RNN 与 LSTM 的图像描述生成（Image Captioning）。
- [ ] **Assignment 6: Transformers & Generative Models**
  - Transformer 架构与 Self-Attention 机制。
  - 变分自编码器（VAE）与生成对抗网络（GAN）。

---

### 三、 长线拓展课程

- [ ] **GAMES003：计算机图形学与动画**
  - **课程链接**：[GAMES003 课程主页](https://pengsida.net/games003/)
  - **定位**：不要求在“节点1（深度学习基础培养）”学完，作为周记/空闲时间的长线储备课程，为后续 3D 重建与 VR 等科研做铺垫。

---

## 📅 过程管理与沟通机制

1. **钉钉每周一例行汇报**
   - **时间**：每周一准时发送。
   - **内容**：
     1. 上周进度复盘（完成的 Assignment 节点、GitHub Commit 说明）。
     2. 本周计划与推进目标。
     3. 遇到的核心难点及思考。
   - **特殊情况**：遇法定节假日自动 skip。
2. **提问与解决问题原则**
   - 遇到代码或算法问题，**优先使用 Google 搜索或询问 DeepSeek** 解决。
   - 思考并排查后仍无法解决的核心算法或逻辑困惑，整理好上下文向彭老师提问。
3. **节点通关申请**
   - EECS 498 编程作业全部完成并维护好 GitHub 仓库后，向彭老师报备，申请参加**实验室编程考核**。