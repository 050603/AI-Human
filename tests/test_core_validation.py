from __future__ import annotations

from classroom_ai.pipeline.core_validation import run_core_validation
from classroom_ai.schemas.transcript import Transcript
from classroom_ai.slicing.time_slicer import slice_transcript_by_time
from classroom_ai.uncertainty.entropy import shannon_entropy
from classroom_ai.uncertainty.semantic_entropy import compute_semantic_entropy


def test_time_slicer_creates_overlapping_slices():
    transcript = Transcript.from_dict(
        {
            "lesson_id": "demo",
            "segments": [
                {"segment_id": "a", "start": 0, "end": 10, "speaker": "S0", "text": "开始"},
                {"segment_id": "b", "start": 90, "end": 100, "speaker": "S0", "text": "结束"},
            ],
        }
    )

    slices = slice_transcript_by_time(transcript, window_seconds=60, overlap_seconds=10)

    assert [item.slice_id for item in slices] == ["demo_slice_0001", "demo_slice_0002"]
    assert slices[0].start == 0
    assert slices[1].start == 50


def test_entropy_is_zero_for_identical_values():
    assert shannon_entropy([5, 5, 5]) < 1e-9


def test_semantic_entropy_clusters_similar_reasons():
    entropy, labels, clusters = compute_semantic_entropy(
        [
            "教师通过追问促进学生解释原因。",
            "教师通过追问促进学生解释原因。",
            "教师反馈偏控制，缺少支架。",
        ],
    )

    assert entropy > 0
    assert labels[0] == labels[1]
    assert len(clusters) >= 2


def test_core_validation_pipeline_runs_with_mock_local_provider():
    result = run_core_validation("data/sample/lesson_001.json", "configs/local_mock.yaml")

    assert result["lesson_id"] == "lesson_001"
    assert result["slice_count"] >= 1
    first = result["results"][0]
    assert first["monte_carlo_samples"] == 20
    assert first["decision"] in {"auto_accept", "human_review"}
    assert 1 <= first["majority_score"] <= 7
