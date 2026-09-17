# Slot-local representation plan

在冻结 staged dual2 factorized parent sender 后，替换 worker 0，并删除一个目标组合的 child 训练支持。`history` 与 `slot_local` 使用同一 seed、同一 episode stream、同一 live/silent 对照和 3000 updates；唯一操纵是 worker 的消息状态输入。预注册零样本标准为缺失组合 held-out natural ≥ 0.60。若 slot-local 恢复而 history 失败，说明行动时机需要配合接收者的因子化状态；若两者都失败，组合协议仍缺少跨组合学习压力。
