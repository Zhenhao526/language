# Slot-local receiver representation study

本实验检验 staged dual2 协议的组合留出失败是否来自接收者状态表示。父代 sender、消息容量、任务、伙伴轮换、训练支持和随机流固定，只替换 worker 0 的表示：

- `history`：接收者把两个到达的 token 编成完整消息历史；
- `slot_local`：第一个子任务只读 slot 0，第二个子任务只读 slot 1。

两种表示都在 leave-one-goal-out 支持上训练，并各自配 live/silent。主指标是缺失目标组合的 held-out live−silent、slot-local−history 差值和 0.60 零样本阈值；它是接收者架构干预，不是给主体提供语义字典。

正式归档在 `results/formal_20260917/`，原始 child execution、日志和 checkpoint 不保留。
