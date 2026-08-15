# -*- coding: utf-8 -*-
r"""
============================================================================
 2023 CUMCM C题 蔬菜类商品 —— 原始数据清洗脚本
============================================================================
 任务（一次性完成）：
   1) 检测所有缺失值，并给出填补方案（均值 / 中位数 / 插值，分别说明理由）；
   2) 用箱线图 IQR 方法检测异常值，列出所有异常行；
   3) 完成清洗并输出 cleaned_data.csv。

 数据来源：C:\Users\imagination\Desktop\2023C
   - 附件1.xlsx   商品信息（251 单品 / 6 品类）            → 主数据
   - 附件2.xlsx   销售流水（878503 行, 2020-07-01~2023-06-30）→ 主表
   - 附件3.xlsx   批发价格（55982 行）                     → 成本序列
   - 附件4.xlsx   损耗率（251 单品 + 6 品类平均）          → 属性数据

 输出：
   - cleaned_data/cleaned_data.csv           清洗后的主表（附件2，含填补与异常处理）
   - cleaned_data/outliers_report.csv        全部 IQR 异常行明细
   - cleaned_data/cleaned_附件3_批发价格.csv  清洗后的批发价格
   - cleaned_data/cleaned_附件4_损耗率.csv    清洗后的损耗率
   - 控制台打印缺失报告与异常摘要

 说明：原始文件只读，所有清洗结果写入 cleaned_data/ 目录，绝不覆盖原始附件。
============================================================================
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. 环境与路径配置
# ---------------------------------------------------------------------------
BASE_DIR = r"C:\Users\imagination\Desktop\2023C"   # 原始数据所在目录
RAW_DIR  = os.path.join(BASE_DIR, "workspace", "data_raw")  # 只读原始数据
OUT_DIR  = os.path.join(BASE_DIR, "cleaned_data")           # 清洗输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# 异常值处理方式（"winsorize"=封顶到箱线图上下限, "remove"=删除异常行, "flag"=仅标记）
OUTLIER_ACTION = "winsorize"

# ---------------------------------------------------------------------------
# 1. 读取全部附件（保留原始数据，不做任何修改）
# ---------------------------------------------------------------------------
# 商品编码统一读成字符串，避免 Excel 中 1.029e+14 的科学计数法精度丢失
a1 = pd.read_excel(os.path.join(RAW_DIR, "附件1.xlsx"),
                   dtype={"单品编码": str, "分类编码": str})
a2 = pd.read_excel(os.path.join(RAW_DIR, "附件2.xlsx"),
                   dtype={"单品编码": str, "销售类型": str, "是否打折销售": str})
a3 = pd.read_excel(os.path.join(RAW_DIR, "附件3.xlsx"),
                   dtype={"单品编码": str})
a4 = pd.read_excel(os.path.join(RAW_DIR, "附件4.xlsx"),
                   sheet_name="Sheet1", dtype={"单品编码": str})
# 附件4 中还有一张“品类平均损耗率”表，也一并读入供报告使用
a4_cat = pd.read_excel(os.path.join(RAW_DIR, "附件4.xlsx"),
                       sheet_name="平均损耗率(%)_小分类编码_不同值",
                       dtype={"小分类编码": str})

print("=" * 78)
print("已读取原始数据：")
print(f"  附件1 商品信息 : {a1.shape[0]} 行, {a1.shape[1]} 列")
print(f"  附件2 销售流水 : {a2.shape[0]} 行, {a2.shape[1]} 列")
print(f"  附件3 批发价格 : {a3.shape[0]} 行, {a3.shape[1]} 列")
print(f"  附件4 损耗率   : {a4.shape[0]} 行, {a4.shape[1]} 列")
print("=" * 78)

# ---------------------------------------------------------------------------
# 2. 缺失值检测与填补
# ---------------------------------------------------------------------------
# 2.1 定义一个通用的“缺失值报告 + 填补”函数
#     strategy 的取值:
#       - "mean"    均值填补: 仅当列近似对称分布且无严重异常值时使用。
#       - "median"  中位数填补: 分布右偏/存在异常值时使用, 稳健、不受极端值影响。
#       - "interp"  插值填补: 该列本质是时间序列, 用相邻观测线性插值,
#                    能利用时间先后信息; 序列首尾或孤立缺失则回退到中位数。
#     fill 用于执行; explain 是对“为什么选这个”的说明, 打印给用户看。
def impute_missing(df, strategy, name):
    """对 DataFrame 按 strategy 执行缺失填补, 返回 (填补后df, 报告字典)。"""
    report = {"file": name, "filled": {}}
    for col, method in strategy.items():
        if col not in df.columns:
            continue
        n_miss = int(df[col].isna().sum())
        if n_miss == 0:
            continue  # 该列无缺失, 无需填补
        if method == "mean":
            fill_val = df[col].mean()
            df[col] = df[col].fillna(fill_val)
        elif method == "median":
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
        elif method == "interp":
            # 先线性插值（利用时间先后顺序）, 首尾仍缺失的用中位数兜底
            before = df[col].isna().sum()
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
            df[col] = df[col].fillna(df[col].median())
            fill_val = "interpolate(linear)+median-backfill"
        else:
            raise ValueError(f"未知填补方法: {method}")
        report["filled"][col] = {"method": method, "count": n_miss,
                                 "fill_value": fill_val}
    return df, report

# 2.2 为每个附件定义填补策略（含理由）
#     先用偏度佐证：销量/单价/批发价/损耗率均右偏(偏度>0), 故均值不稳健。
skew_report = {}
for col, s in [("销量(千克)", a2["销量(千克)"]),
               ("销售单价(元/千克)", a2["销售单价(元/千克)"]),
               ("批发价格(元/千克)", a3["批发价格(元/千克)"]),
               ("损耗率(%)", a4["损耗率(%)"])]:
    skew_report[col] = round(s.skew(), 2)

# 策略说明（对应题目要求的“分别说明理由”）：
#   - 销量(千克)      → median: 强右偏(偏度约 ~6), 均值被大单抬高, 中位数稳健;
#                       销量不是平滑时间序列, 插值会引入虚假的连续感。
#   - 销售单价        → interp: 单价本质是按日成交的价格时间序列, 相邻日信息可用;
#                       同时单价有异常值, 不用均值。
#   - 批发价格        → interp: 单品-日成本序列, 线性插值保留成本时间走势;
#                       首尾缺失用该列中位数兜底。
#   - 损耗率(%)       → median: 单值/盘点口径, 无时间维度, 且含异常值(右偏), 用中位数稳健。
#   - 均值(mean)      → 本例中不使用: 所有数值列均右偏, 均值被极端值拉高;
#                       仅在近似对称、无异常时才合适（代码已保留该选项）。
impute_plan = {
    "附件2": {"销量(千克)": "median",
              "销售单价(元/千克)": "interp"},
    "附件3": {"批发价格(元/千克)": "interp"},
    "附件4": {"损耗率(%)": "median"},
}
impute_reason = {
    "销量(千克)": "median：销量强右偏(偏度{:.2f})，均值受大单影响大，中位数稳健；且销量非平滑序列，不宜插值。",
    "销售单价(元/千克)": "interp：单价为按日价格时间序列，可用相邻日线性插值保留趋势；含异常值故不用均值。",
    "批发价格(元/千克)": "interp：单品-日成本序列，线性插值保留成本走势，首尾回退中位数。",
    "损耗率(%)": "median：损耗率为盘点口径单值、无时间维度，分布右偏(偏度{:.2f})，中位数稳健。",
}

print("\n----- 偏度检查（用于论证填补方法） -----")
for col, sk in skew_report.items():
    print(f"  {col:<18} 偏度 = {sk:>8.2f}")

print("\n----- 缺失值检测结果 -----")
missing_report = []
for name, df in [("附件1", a1), ("附件2", a2), ("附件3", a3),
                 ("附件4-Sheet1", a4), ("附件4-品类", a4_cat)]:
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if len(miss) == 0:
        print(f"  [{name:<12}] 缺失值: 无")
    else:
        print(f"  [{name:<12}] 缺失值: {miss.to_dict()}")
missing_report.append({"result": "当前 4 个附件均无缺失值"})

# 2.3 执行填补（当前无缺失 → 循环内自动跳过, 但代码逻辑完整可用）
imputed_dfs = {}
for key, strategy in impute_plan.items():
    target = {"附件2": a2, "附件3": a3, "附件4": a4}[key]
    target, rep = impute_missing(target, strategy, key)
    imputed_dfs[key] = target
    if rep["filled"]:
        print(f"  [{key}] 填补完成: {rep['filled']}")
    else:
        print(f"  [{key}] 无缺失需要填补")

# 打印填补策略与理由（输出成表, 便于写入报告）
strategy_rows = []
for col, m in [("销量(千克)", "median"), ("销售单价(元/千克)", "interp"),
               ("批发价格(元/千克)", "interp"), ("损耗率(%)", "median")]:
    strategy_rows.append({
        "列": col, "方法": m,
        "理由": impute_reason[col].format(skew_report.get(col, 0))
    })
strategy_df = pd.DataFrame(strategy_rows)
strategy_df.to_csv(os.path.join(OUT_DIR, "imputation_strategy.csv"),
                   index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 3. 异常值检测（箱线图 IQR 法）与处理
# ---------------------------------------------------------------------------
def iqr_outliers(s):
    """对数值序列做箱线图 IQR 检测, 返回 (Q1, Q3, IQR, 下界, 上界, 异常掩码)。"""
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (s < lower) | (s > upper)
    return q1, q3, iqr, lower, upper, mask

outlier_records = []   # 汇总所有文件的异常行

def detect_and_treat(df, col, name, clip_lower=True, clip_upper=True,
                     extra_mask_ok=None, df_uid="row_index"):
    """
    对 df[col] 执行 IQR 检测 + 处理（winsorize/remove/flag）。
    clip_lower/clip_upper: 是否封顶下/上界（损耗率等有自然下界0时只封顶上界）。
    extra_mask_ok: 额外的“视为合法、不判异常”的掩码（如退货行）。
    返回处理后的 Series 与异常标记 Series。
    """
    s = pd.to_numeric(df[col], errors="coerce")
    valid = s.notna()
    base = s[valid]
    q1, q3, iqr, lo, hi, _ = iqr_outliers(base)

    # 异常判定：基础 IQR 掩码, 再叠加 extra_mask_ok（合法业务行除外）
    out_mask = pd.Series(False, index=df.index)
    out_mask.loc[valid] = ((base < lo) | (base > hi)).values
    if extra_mask_ok is not None:
        out_mask &= ~extra_mask_ok   # 合法行（如退货）不判异常

    n_out = int(out_mask.sum())
    # 记录异常行明细
    for idx in df.index[out_mask]:
        outlier_records.append({
            "文件": name, "列": col, "行号(原表)": idx,
            "值": round(float(s.loc[idx]), 4),
            "Q1": round(q1, 4), "Q3": round(q3, 4),
            "下界": round(lo, 4), "上界": round(hi, 4),
            "方向": "高侧异常" if s.loc[idx] > hi else "低侧异常",
        })

    # 处理
    treated = s.copy()
    if OUTLIER_ACTION == "winsorize":
        if clip_lower:
            treated = treated.clip(lower=lo)
        if clip_upper:
            treated = treated.clip(upper=hi)
        # 重要：extra_mask_ok 标记的合法业务行（如退货）只是“不判异常”，
        # 其原始值同样不应被封顶篡改，故恢复原值。
        if extra_mask_ok is not None:
            treated[extra_mask_ok] = s[extra_mask_ok]
    elif OUTLIER_ACTION == "remove":
        treated.loc[out_mask] = np.nan   # 调用方负责删除行
    # "flag" 模式下 treated 保持原值, 仅靠 out_mask 标记

    print(f"  [{name:<10}] {col:<18} Q1={q1:.3f} Q3={q3:.3f} "
          f"IQR={iqr:.3f} 界限=[{lo:.3f}, {hi:.3f}] 异常行数={n_out}")
    return treated, out_mask

print("\n----- 异常值检测（箱线图 IQR 法）与处理 -----")

# 3.1 附件2 销量(千克)
#     特殊说明：销量<0 的 461 行全部是“退货”（合法业务事件），不是数据错误，
#     也不参与异常封顶 —— 通过 extra_mask_ok 排除。
returns_mask = a2["销量(千克)"] < 0
a2["销量_清洗后"], a2["销量_是否异常"] = detect_and_treat(
    a2, "销量(千克)", "附件2", clip_lower=True, clip_upper=True,
    extra_mask_ok=returns_mask)
print(f"    其中 销量<0 的退货行数={int(returns_mask.sum())}（合法退货，未判异常、未封顶）")

# 3.2 附件2 销售单价(元/千克)
a2["销售单价_清洗后"], a2["销售单价_是否异常"] = detect_and_treat(
    a2, "销售单价(元/千克)", "附件2")

# 3.3 附件3 批发价格(元/千克)
a3["批发价格_清洗后"], a3["批发价格_是否异常"] = detect_and_treat(
    a3, "批发价格(元/千克)", "附件3")

# 3.4 附件4 损耗率(%)
#     特殊说明：损耗率有自然下界 0（无损耗 = 合法，且可能是好事），
#     因此只对“高侧异常”封顶，低侧(如0)保留原值。
a4["损耗率_清洗后"], a4["损耗率_是否异常"] = detect_and_treat(
    a4, "损耗率(%)", "附件4", clip_lower=False, clip_upper=True)

# 保存全部异常行明细
outlier_df = pd.DataFrame(outlier_records)
outlier_df.to_csv(os.path.join(OUT_DIR, "outliers_report.csv"),
                  index=False, encoding="utf-8-sig")
print(f"\n异常行明细已保存: cleaned_data/outliers_report.csv "
      f"（共 {len(outlier_df)} 行，按文件/列聚合见下）")
print(outlier_df.groupby(["文件", "列"]).size().to_string())

# ---------------------------------------------------------------------------
# 4. 组装清洗后的主表并输出 cleaned_data.csv
# ---------------------------------------------------------------------------
# 4.1 附加一个“分类名称”列（来自附件1，主数据增强，不影响清洗语义）
a2_clean = a2.merge(a1[["单品编码", "分类编码", "分类名称"]],
                    on="单品编码", how="left")

# 4.2 用清洗后的数值列替换原始列
a2_clean["销量(千克)"] = a2_clean["销量_清洗后"]
a2_clean["销售单价(元/千克)"] = a2_clean["销售单价_清洗后"]

# 4.3 写主表（含两列异常标记，保留原始信息可追溯）
cleaned_main = a2_clean[[
    "销售日期", "扫码销售时间", "单品编码", "分类编码", "分类名称",
    "销量(千克)", "销售单价(元/千克)", "销售类型", "是否打折销售",
    "销量_是否异常", "销售单价_是否异常",
]].rename(columns={
    "销量_是否异常": "销量_异常标记", "销售单价_是否异常": "销售单价_异常标记"
})
cleaned_main.to_csv(os.path.join(OUT_DIR, "cleaned_data.csv"),
                    index=False, encoding="utf-8-sig")

# 4.4 顺带保存清洗后的批发价格与损耗率（同样保留异常标记）
a3[["日期", "单品编码", "批发价格_清洗后", "批发价格_是否异常"]].rename(
    columns={"批发价格_清洗后": "批发价格(元/千克)",
             "批发价格_是否异常": "批发价格_异常标记"}).to_csv(
    os.path.join(OUT_DIR, "cleaned_附件3_批发价格.csv"),
    index=False, encoding="utf-8-sig")
a4[["单品编码", "单品名称", "损耗率_清洗后", "损耗率_是否异常"]].rename(
    columns={"损耗率_清洗后": "损耗率(%)",
             "损耗率_是否异常": "损耗率_异常标记"}).to_csv(
    os.path.join(OUT_DIR, "cleaned_附件4_损耗率.csv"),
    index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 4.5 同步到建模工作流数据目录 (workspace/data_clean/)
#     供后续 Q1-Q3 建模直接复用: 清洗后的主表、批发价格、损耗率。
#     （standalone 交付物仍保留在 cleaned_data/, 此处只做复制, 不移动原文件）
# ---------------------------------------------------------------------------
import shutil
WC_DIR = os.path.join(BASE_DIR, "workspace", "data_clean")
os.makedirs(WC_DIR, exist_ok=True)
sync_map = {
    "cleaned_data.csv":              "sales_cleaned.csv",        # 附件2 清洗后主表
    "cleaned_附件3_批发价格.csv":      "wholesale_cleaned.csv",    # 附件3 清洗后批发价
    "cleaned_附件4_损耗率.csv":        "loss_rate_cleaned.csv",    # 附件4 清洗后损耗率
}
for src_name, dst_name in sync_map.items():
    shutil.copy2(os.path.join(OUT_DIR, src_name), os.path.join(WC_DIR, dst_name))

# ---------------------------------------------------------------------------
# 5. 汇总摘要
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("清洗完成。输出文件：")
for f in sorted(os.listdir(OUT_DIR)):
    p = os.path.join(OUT_DIR, f)
    print(f"  {os.path.join('cleaned_data', f):<42} {os.path.getsize(p):>10,} 字节")
print("-" * 78)
print("已同步到建模工作流 (workspace/data_clean/):")
for dst in sync_map.values():
    print(f"  workspace/data_clean/{dst}")
print("=" * 78)
print(f"cleaned_data.csv 行数: {len(cleaned_main):,}, "
      f"列数: {cleaned_main.shape[1]}")
print(f"填补方案与理由已保存: cleaned_data/imputation_strategy.csv")
print(f"异常行明细已保存:     cleaned_data/outliers_report.csv")
