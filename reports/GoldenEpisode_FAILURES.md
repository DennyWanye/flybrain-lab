# Golden Episode 当前失败项与边界

GOLDEN_EPISODE_READY = YES
MODEL_READY_FOR_NEXT_STAGE = NO

浏览器Golden回放和GE-01～GE-15验收已完成，证据见GoldenEpisode_TEST_RESULTS.md及GoldenEpisode_ACCEPTANCE.json。

模型评估未通过：固定32场景仅8个成功（25%），24个越界（75%），0碰撞、0超时。训练分布附近7/8，反向0/8，横跨房间0/8，短距离1/8。失败明细和动作轨迹见golden_evaluation_final/episodes.jsonl。

未达到预先固定的诊断成功率>=90%、碰撞/越界率<=1%。更不能宣称通过TS1三训练seed×300封存case及随机基线比较的正式门槛。本轮不重训，不为凑成功率修改测试场景。

已知边界：固定高度、空房间、模拟pose输入、首个动作直接设置airborne；只记录最后一个神经子步。未测试噪声、障碍、起飞流程、真实硬件或Live训练期间断开浏览器。这些没有被伪装为通过。
