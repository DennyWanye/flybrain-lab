# 数据合同

五份JSON Schema使用Draft 2020-12；事件包含transition、neural_sample、training_metric、session_status四类。

`view_manifest.schema.json`用于不可变离线导出；`live_descriptor.schema.json`用于活动源，不对每次变化的快照沿用旧文件hash。后两种manifest不得混用。

字段含义和跨字段规则见包根目录`05_数据合同与接入说明.md`。`tools/validate_contracts.py`是可运行的样例/基础语义校验，不是完整服务实现；Web端还要用同schema做运行时校验与自己的交互测试。
