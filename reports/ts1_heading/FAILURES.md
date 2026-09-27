# H1 封存失败清单

900 场中有 20 场失败，均为 60 秒任务超时；碰撞或越界为 0。失败记录末尾存在顺时针与逆时针反复切换。部分场景瞬时角度已在阈值内，但没有同时满足连续保持 2 秒，因此仍正确计为失败。该观察来自已有记录，不构成本轮调参或重新训练。

| 种子 | 场景 | 终止误差 | 连续保持 | 最后五个动作 |
|---|---|---|---|---|
| 11 | h1-test-050 | 22.48° | 0.0s | CCW30 / CW30 / CW30 / CCW30 / CCW30 |
| 11 | h1-test-081 | 11.88° | 0.1s | CW30 / CCW30 / CCW30 / CW30 / CCW30 |
| 11 | h1-test-087 | 13.18° | 0.0s | CW30 / CCW30 / CW30 / CCW30 / CW30 |
| 11 | h1-test-126 | 14.00° | 0.0s | CW30 / CCW30 / STOP / CW30 / CCW30 |
| 11 | h1-test-142 | 14.21° | 0.0s | CCW30 / CW30 / STOP / CCW30 / CW30 |
| 11 | h1-test-152 | 22.96° | 0.0s | CW30 / CCW30 / CW30 / CCW30 / CW30 |
| 11 | h1-test-225 | 26.36° | 0.0s | CCW30 / CW30 / CCW30 / CW30 / CCW30 |
| 11 | h1-test-266 | 14.00° | 0.0s | CW30 / CCW30 / STOP / CW30 / CCW30 |
| 11 | h1-test-276 | 26.36° | 0.0s | CCW30 / CW30 / CCW30 / CW30 / CCW30 |
| 11 | h1-test-283 | 28.48° | 0.0s | STOP / CCW30 / CW30 / CW30 / CCW30 |
| 11 | h1-test-298 | 34.39° | 0.0s | CW30 / CW30 / STOP / CCW30 / CW30 |
| 22 | h1-test-085 | 12.61° | 0.1s | CW30 / CW30 / CCW30 / CCW30 / CW30 |
| 22 | h1-test-109 | 8.59° | 0.0s | CW30 / STOP / CCW30 / CW30 / CCW30 |
| 22 | h1-test-150 | 0.91° | 0.0s | CCW30 / CW30 / CCW30 / STOP / CW30 |
| 22 | h1-test-194 | 10.28° | 0.1s | CW30 / STOP / CCW30 / CW30 / CCW30 |
| 22 | h1-test-245 | 40.63° | 0.0s | CCW30 / CW30 / CW30 / CCW30 / CCW30 |
| 22 | h1-test-256 | 46.16° | 0.0s | CW30 / CCW30 / CCW30 / CW30 / CCW30 |
| 33 | h1-test-142 | 40.40° | 0.0s | CCW30 / CW30 / CW30 / STOP / CCW30 |
| 33 | h1-test-145 | 42.20° | 0.0s | CCW30 / CW30 / CCW30 / CW30 / CCW30 |
| 33 | h1-test-154 | 13.15° | 0.0s | STOP / CCW30 / CCW30 / CW30 / CW30 |
