"""GuardAI · 教材为准的教育答疑系统 —— 核心模块

数据流（一条问答的完整路径）：
    提问
     └─ safety.py     输入侧风险分级（L0~L3）+ 会话累积 + 危机分流
     └─ router.py     分诊：这题落在哪本教材（single / consult / none）
     └─ retrieve.py   dense + BM25 → RRF 融合 →（可选重排）→ fill-to-budget 组装上下文
     └─ generate.py   调用生成模型（glm / kimi / deepseek，OpenAI 兼容接口）
     └─ present.py    呈现层：置信分级 + 兜底话术（拒答/会诊并列）
     └─ nli.py        本地 NLI 核对（答案是否被原文支持）

模块概览：
    common.py    配置加载 / 路径解析 / 磁盘缓存 / token 估算 / 日志
    corpus.py    教材读取 + 切片（固定 token 窗口）
    index.py     稠密嵌入（bge）+ BM25 索引（numpy 后端，可切 faiss）
    retrieve.py  检索 + RRF 融合 + 可选重排 + fill-to-budget
    router.py    分诊—会诊路由（单源 / 会诊 / 超纲拒答）
    generate.py  三家国产模型统一调用 + PK/CK 双路径 + 策略 S1/S2/S4
    evaluate.py  四格标签 + 归因 + 冲突 / RAG 指标（离线评测用）
    present.py   呈现层：把不确定性变成用户看得见的东西
    prompts.py   所有提示词模板集中管理（含安全护栏三段）
    safety.py    安全护栏：风险分级 / 会话累积 / 危机分流 / 告警 / 输出复核
    nli.py       本地 NLI 判据（DeBERTa-v3 MNLI，逐句核对答案是否被原文支持）
    qwen3_reranker.py  Qwen3-Reranker 本地包装器（可选重排后端）
"""
