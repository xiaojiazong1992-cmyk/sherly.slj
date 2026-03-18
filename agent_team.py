# -*- coding: utf-8 -*-
"""
agent_team.py
住宅工程建筑安装工程造价预测 —— Multi-Agent Team
----------------------------------------------------
四个专职 Agent 协作完成项目全流程：

  - 架构师 (architect)   : 设计整体算法架构与改进方向
  - 数据专员 (data)      : 数据采集、清洗、特征分析
  - 代码工程师 (coder)   : 运行模型代码、发现并修复 Bug
  - 结果总结员 (reporter): 汇总评估指标并生成报告

用法：
    export ANTHROPIC_API_KEY=<your-key>
    python agent_team.py [--task <task_description>]

默认 task：运行完整 pipeline 并汇总结果。
"""

import os
import sys
import argparse
import asyncio
import anyio

from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition, ResultMessage, SystemMessage

# ─── 项目根目录 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_DIR = os.path.join(PROJECT_ROOT, "construction_cost_predictor")

# ─── Agent 定义 ────────────────────────────────────────────────────────────────

ARCHITECT_AGENT = AgentDefinition(
    description=(
        "总体算法架构师。负责分析当前预测模型的整体框架，"
        "提出算法改进方向、超参数搜索策略、特征工程优化建议，"
        "以及 Blending/Stacking 融合方案的设计思路。"
        "只做分析与建议，不直接执行代码。"
    ),
    prompt=(
        "你是一位机器学习算法架构师，专注于建筑工程造价预测领域。\n"
        "项目位于 " + PREDICTOR_DIR + "。\n\n"
        "你的职责：\n"
        "1. 阅读 src/models.py、src/preprocessing.py、src/feature_engineering.py，"
        "   理解当前 Ridge → RF → XGBoost → Blending → Stacking 的五模型架构。\n"
        "2. 分析各模型的超参数搜索策略（GridSearch / RandomizedSearch）是否合理。\n"
        "3. 评估 Expanding Window CV 时序交叉验证的设计是否能防止数据泄露。\n"
        "4. 提出可量化的改进方向（如加入 LightGBM、贝叶斯优化、时序特征等）。\n"
        "5. 给出 Blending 权重优化和 Stacking 元学习器的改进建议。\n\n"
        "请用中文输出结构化的架构分析报告（Markdown 格式）。"
    ),
    tools=["Read", "Glob", "Grep"],
)

DATA_AGENT = AgentDefinition(
    description=(
        "数据专员。负责检查数据文件质量、分析特征分布、"
        "评估缺失值/异常值处理方案，以及核查 outputs/ 目录中"
        "已生成的 Excel 数据文件是否完整、合理。"
    ),
    prompt=(
        "你是一位数据分析专员，专注于建筑工程造价数据质量评估。\n"
        "项目位于 " + PREDICTOR_DIR + "。\n\n"
        "你的职责：\n"
        "1. 检查 outputs/01_raw_data.xlsx 中原始数据的字段完整性和数据量。\n"
        "2. 读取 outputs/02_clean_data.xlsx 评估清洗后数据的统计特征。\n"
        "3. 读取 outputs/03_feature_stats.xlsx 分析特征相关性和 VIF 多重共线性。\n"
        "4. 阅读 src/data_scraper.py 了解数据采集策略（NBS/GGZY/合成）。\n"
        "5. 阅读 src/preprocessing.py 评估特征工程（对数变换、派生特征）的合理性。\n"
        "6. 指出数据层面可能存在的问题（样本偏差、时间跨度、省份覆盖等）。\n\n"
        "请用中文输出结构化的数据质量报告（Markdown 格式）。"
    ),
    tools=["Read", "Glob", "Grep", "Bash"],
)

CODER_AGENT = AgentDefinition(
    description=(
        "代码工程师。负责运行模型训练脚本、捕获错误信息、"
        "定位并修复 Bug，确保 run_all.py 能够端到端成功执行。"
        "当发现代码问题时直接修改相关文件。"
    ),
    prompt=(
        "你是一位 Python 机器学习工程师，负责保证项目代码正常运行。\n"
        "项目位于 " + PREDICTOR_DIR + "。\n\n"
        "你的职责：\n"
        "1. 在 " + PREDICTOR_DIR + " 目录下运行 `python run_all.py` 并捕获输出。\n"
        "2. 检查各 Step 是否全部 [OK]，若有 [FAIL] 则定位根因。\n"
        "3. 阅读相关源文件（src/*.py）理解失败原因。\n"
        "4. 修复发现的 Bug（直接编辑文件），并重新运行验证。\n"
        "5. 确认 outputs/ 目录下所有 .xlsx 和 figures/*.png 均已生成。\n"
        "6. 输出每个修复点的说明（改了什么、为什么这样改）。\n\n"
        "注意：\n"
        "- 运行命令时始终在 " + PREDICTOR_DIR + " 目录下执行。\n"
        "- 如果训练时间过长，记录进度并汇报中间状态。\n"
        "- 请用中文输出代码运行报告（Markdown 格式）。"
    ),
    tools=["Read", "Glob", "Grep", "Bash", "Edit", "Write"],
)

REPORTER_AGENT = AgentDefinition(
    description=(
        "结果总结员。负责读取模型评估结果、SHAP 特征重要性，"
        "汇总各模型的 MAE/MAPE/R² 指标，生成最终的实验结论与论文摘要建议。"
    ),
    prompt=(
        "你是一位科研结果分析专员，负责汇总建筑工程造价预测实验的最终结论。\n"
        "项目位于 " + PREDICTOR_DIR + "。\n\n"
        "你的职责：\n"
        "1. 读取 outputs/04_model_results.xlsx，整理五个模型"
        "   （Ridge/RF/XGBoost/Blending/Stacking）的性能指标表格。\n"
        "2. 读取 outputs/05_shap_importance.xlsx，列出 Top-10 重要特征。\n"
        "3. 分析 outputs/figures/ 中的图表文件列表，确认图表完整性。\n"
        "4. 对比各模型性能，指出最优模型及其优势原因。\n"
        "5. 给出论文'实验结论'章节的摘要建议（200-300字）。\n"
        "6. 给出未来改进方向的简短建议（3-5条）。\n\n"
        "请用中文输出结构化的实验结果报告（Markdown 格式）。"
    ),
    tools=["Read", "Glob", "Grep", "Bash"],
)

# ─── 主任务编排 ────────────────────────────────────────────────────────────────

MAIN_TASK = """
你是一位项目总协调人，负责调度以下四位专家 Agent 完成住宅工程建筑安装工程造价预测项目：

1. **architect**（架构师）：分析当前算法架构，提出改进建议
2. **data**（数据专员）：评估数据质量，检查特征工程
3. **coder**（代码工程师）：运行代码、修复 Bug，确保 pipeline 顺利完成
4. **reporter**（结果总结员）：汇总模型评估指标，生成实验报告

**执行顺序建议：**
- 先让 architect 做架构分析
- 同时让 data 做数据质量评估
- 让 coder 运行完整 pipeline（如有错误则修复）
- 最后让 reporter 汇总结果

请依次调用各 Agent，收集它们的输出，最终生成一份综合报告，
包含：架构评估、数据质量评估、代码运行状态、模型性能对比、最优模型结论。

项目工作目录：""" + PREDICTOR_DIR


async def run_agent_team(task: str, verbose: bool = True) -> str:
    """运行 Multi-Agent Team，返回最终综合报告"""

    session_id = None
    final_result = ""

    print("\n" + "=" * 70)
    print("  住宅工程建筑安装工程造价预测 — Agent Team 启动")
    print("=" * 70)
    print(f"工作目录: {PREDICTOR_DIR}")
    print(f"任务: {task[:100]}...")
    print("=" * 70 + "\n")

    options = ClaudeAgentOptions(
        cwd=PREDICTOR_DIR,
        allowed_tools=["Read", "Glob", "Grep", "Bash", "Edit", "Write", "Agent"],
        permission_mode="acceptEdits",
        model="claude-opus-4-6",
        agents={
            "architect": ARCHITECT_AGENT,
            "data": DATA_AGENT,
            "coder": CODER_AGENT,
            "reporter": REPORTER_AGENT,
        },
        system_prompt=(
            "你是项目总协调人，使用中文与用户沟通。"
            "调用各专家 Agent 时使用 Agent 工具，并将各 Agent 的输出整合为最终报告。"
            "请确保各 Agent 的工作目录都设置为：" + PREDICTOR_DIR
        ),
        max_turns=50,
    )

    async for message in query(prompt=task, options=options):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            session_id = message.data.get("session_id")
            if verbose:
                print(f"[Session] ID: {session_id}")

        elif isinstance(message, ResultMessage):
            final_result = message.result
            if verbose:
                print("\n" + "=" * 70)
                print("  最终综合报告")
                print("=" * 70)
                print(final_result)

        elif verbose:
            # 显示中间消息类型
            msg_type = getattr(message, "type", type(message).__name__)
            if hasattr(message, "content"):
                content = message.content
                if isinstance(content, list):
                    for block in content:
                        if hasattr(block, "text") and block.text:
                            print(block.text, end="", flush=True)
            elif hasattr(message, "subtype"):
                print(f"[{msg_type}/{message.subtype}]", flush=True)

    return final_result


def main():
    parser = argparse.ArgumentParser(
        description="住宅工程造价预测 Multi-Agent Team"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=MAIN_TASK,
        help="自定义任务描述（默认：运行完整 pipeline 并汇总结果）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式：只输出最终报告",
    )
    args = parser.parse_args()

    # 检查 API Key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("错误：请先设置 ANTHROPIC_API_KEY 环境变量")
        print("  export ANTHROPIC_API_KEY=<your-key>")
        sys.exit(1)

    anyio.run(run_agent_team, args.task, not args.quiet)


if __name__ == "__main__":
    main()
