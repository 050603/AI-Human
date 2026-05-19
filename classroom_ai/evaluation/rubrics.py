from __future__ import annotations

UNESCO_RUBRIC: dict[str, dict[str, object]] = {
    "human_centered": {
        "name": "以人为本 (Human-Centered)",
        "levels": {
            "2-3分 (理解层)": "培养对AI由人类主导的认知(HAL)；理解人类管控的必要性(CHEC)；培养对人类与机器能动性关系的批判性思维(DAMA)。",
            "4-5分 (应用层)": "树立人类问责是法定义务的观念(LAW-AI)；理解使用AI决策时人类的社会责任(HARM)；具备引导AI目的性使用的态度与能力(CAP-AI)。",
            "6-7分 (创建层)": "培养具备批判性思维的AI公民意识(CAC)；培养AI社会中的个人与社会责任(PSR-AI)；培养作为AI公民的自我实现感与终身学习态度(ALL-AI)。",
        },
    },
    "ai_ethics": {
        "name": "AI伦理 (AI Ethics)",
        "levels": {
            "2-3分 (理解层)": "阐释AI相关困境及伦理冲突背后的原因(AIDEC)；理解基于场景的AI伦理原则及其影响(SCEP-AI)；对AI伦理原则进行具身反思(ERIC-AI)。",
            "4-5分 (应用层)": "培养责任使用AI的自我意识与伦理自律(SACE-AI, ORD-AI)；深化安全使用实操知识及本地法规认知(PRAI-Safe)。",
            "6-7分 (创建层)": "认知并理解“设计即伦理”理念(AUBE)；对现有算法背后的设计即伦理原则保持批判(CRIT-EBD)；在AI监管中维护此理念的社会责任(SURE-AI)。",
        },
    },
    "ai_tech_and_app": {
        "name": "AI技术与应用 (AI Tech & Application)",
        "levels": {
            "2-3分 (理解层)": "举例说明AI的定义与范畴(EDSAI)；构建AI如何基于数据与算法训练的概念(DAKTA)；培养跨学科开放思维(FOMIA)。",
            "4-5分 (应用层)": "强化数据建模/工程/分析技能(SKEMA)；习得适宜的AI编程技能(ATAP)；有效利用开源数据集与AI工具(ASSET)。",
            "6-7分 (创建层)": "掌握开发面向任务的AI工具的高阶技能(CEDAT)；在定制工具包及编程方面展现创造力(CREAIC)；测试与优化自主开发的AI工具(ESTO)。",
        },
    },
    "ai_system_design": {
        "name": "AI系统设计 (AI System Design)",
        "levels": {
            "2-3分 (理解层)": "培养“AI不应被使用的场景”的批判思维(AIDES)；界定待由AI解决的问题范围(SPAR)；评估AI对数据、算法与算力的需求(DACR)。",
            "4-5分 (应用层)": "掌握AI架构的方法论知识与技能(SAMKA)；储备构建AI系统的高阶技术技能与项目管理能力(SPARC)。",
            "6-7分 (创建层)": "批判性评价AI系统(SCAN)；在优化/重构/关停AI系统中建立技术与社会责任(BROS)；塑造AI时代共同创造者的身份认同(FACE)。",
        },
    },
}


def get_default_dimensions() -> list[str]:
    return list(UNESCO_RUBRIC.keys())
