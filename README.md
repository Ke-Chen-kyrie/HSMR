# HSMR

**Reconstructing Humans with a Biomechanically Accurate Skeleton** —— 人体动态重建 / 姿态估计(Human Mesh & Skeleton Reconstruction)项目。

> 个人学习 / 练手项目,仅供研究、教学与个人演示使用。

## 目录结构

```
HSMR/
├── HSMR-main/     核心源码(训练 / 推理 / 评估)
│   ├── configs/            实验配置
│   ├── lib/                核心库
│   ├── docs/              文档
│   ├── exp/ / tests/       实验与测试
│   └── README.md           主项目说明(含许可证与引用)
├── HSMR-infer/   独立推理工程(含 Docker 部署)
├── HSMR-render/  渲染 / 可视化工程
├── HSMR-sdk/     推理 SDK + 示例
└── *.py / *.sh    根级入口脚本(bench / test / deploy 等)
```

## 说明

- 仓库只含**源码 / 配置 / 文档 / 脚本**层;数据(data_inputs)、模型权重与部署产物(deploy)、第三方依赖(thirdparty,以 submodule 形式由 `HSMR-main/.gitmodules` 声明)均不入库。
- 完整项目说明、环境搭建与训练 / 推理方法见 [`HSMR-main/README.md`](HSMR-main/README.md)。
- 请勿将本项目用于简历 / 履历注水或任何欺骗性用途。