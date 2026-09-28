# 本轮代码与报告范围

- V1R：training/altitude_refined.py、altitude_refined_campaign.py；平滑多尺度高度编码、边界附近对称标注、实际稳定状态标签。
- C2：training/spatial.py、spatial_campaign.py 保留首轮失败；C2R 使用 spatial_refined.py、spatial_refined_campaign.py，新划分并修复180秒模拟器关闭传播和重复关闭。
- 取消语义：sdk/async_client.py，取消Future只取消等待；观察器结束响应明确simulation_ended与device_stop_confirmed。
- 配额：recording_v3.py，用新类记录缺失区间并清空缓冲，旧冻结训练Recording代码保留。
- 对照：training/direct_mlp.py，原始26维普通MLP单种子pilot，不使用神经连接组。
- 页面：flyview/tellosim_api.py、static/tellosim.html/js，加入V1R/C2/MLP/统一验收结果、画布帧率和明确结束语义。
- 主体新增测试：test_altitude_refined_skill、test_spatial_skill、test_spatial_refined_skill、test_execution_wait、test_recording_v3。
- 报告：ts1_altitude_refined、ts1_spatial（失败保留）、ts1_spatial_refined、ts1_mlp_pilot、ts1_completion。

源码、权重、训练出处、评估、回放及历史失败均通过独立新目录交付，Git HEAD未被用来伪称包含未提交文件；implementation-hashes.json与包清单定义当前实际版本。

- C2P：spatial_precision.py、spatial_precision_campaign.py；新开发案例确认旧编码近目标分辨率不足，导航改用连续距离通道并独立初始化读出；高度/朝向从保留的C2R训练权重继续，所有划分重新生成。首轮smoke 0/8及独立补充拟合7/8证据分别保存。

- C2Q：spatial_orientation.py、spatial_orientation_campaign.py；新开发记录发现位置已合格时朝向错误转向/停止循环，新增多尺度相对角编码及独立朝向读出，导航和高度从C2P继续。C2P边界失败及其全部数据保留。


C2S/T/U: isolated rejected phase-measurement development candidates. S119/128 failed development; T12/16 failed targeted development; U16/16 targeted but88/90 partial full development, making127/128 impossible, so exact child stopped before neural development or formal evaluation. S/U contain parameter-identical frozen C2Q transfers, no new optimization. Default delivered formal model remains C2Q with readiness NO. All candidate source, partial data and cleanup evidence retained.
