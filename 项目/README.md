# RAG 智能客服自动化测试与质量评测系统

本项目面向测试开发 / AI 测试场景，基于 FastAPI 搭建轻量 RAG 智能客服，并使用 Python + Pytest 构建自动化测试与质量评测框架。项目核心思路是：传统接口测试只能验证 HTTP 状态码和返回结构，无法判断 RAG 是否检索到正确知识、回答是否正确以及是否产生幻觉。

因此，我将 RAG 链路拆分为 Retriever 和 Generator 两个阶段，分别评测 Hit@K、Recall@K、MRR、Answer Correctness、Faithfulness、Refusal Accuracy 等指标，并实现检索失败、生成失败、幻觉、拒答失败等错误归因。同时加入 Prompt Injection 安全测试、Baseline/Candidate 版本回归和 Quality Gate。

当前共完成 83 条 Pytest 自动化测试，全部通过；60 条质量评测用例中，Hit@3 和 Recall@3 均为 90%，MRR 为 0.86。

## 运行

```bash
pip install -r requirements.txt
python -m pytest -v
python scripts/run_evaluation.py
python scripts/run_regression.py
```
