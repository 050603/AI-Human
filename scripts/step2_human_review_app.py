#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

st.title("Stage 2 - 人工靶向微干预")

stage1_path = st.text_input("stage1_result.json 路径", value="outputs/stage1_result.json")
out_path = st.text_input("输出 feedback 路径", value="outputs/stage2_human_feedback.json")

if st.button("加载 Stage1"):
    data = json.loads(Path(stage1_path).read_text(encoding="utf-8"))
    st.session_state["stage1"] = data

stage1 = st.session_state.get("stage1")
feedback = []
if stage1:
    for result in stage1.get("results", []):
        for dim, detail in result.get("dimensions", {}).items():
            if detail.get("decision") != "human_review":
                continue
            st.subheader(f"{result['slice_id']} / {dim}")
            st.write(detail.get("diagnostic", {}))
            need = st.checkbox("需要人工介入", key=f"need-{result['slice_id']}-{dim}", value=True)
            choice = st.text_input("专家选择ID", key=f"choice-{result['slice_id']}-{dim}", value="A")
            custom = st.text_area("自定义意见(选 CUSTOM 时填写)", key=f"custom-{result['slice_id']}-{dim}")
            question_payload = detail.get("diagnostic", {})
            feedback.append({
                "slice_id": result["slice_id"],
                "dimension": dim,
                "needs_human_intervention": need,
                "expert_choice_id": choice,
                "expert_custom_text": custom,
                "core_conflict": question_payload.get("high_score_view", {}).get("reason", "") + " VS " + question_payload.get("low_score_view", {}).get("reason", ""),
                "question_payload": question_payload,
            })

if st.button("保存 Stage2 反馈"):
    payload = {"feedback": feedback}
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    st.success(f"已保存: {p}")
