from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.result).read_text(encoding='utf-8'))
    packets = []
    for r in data.get('results', []):
        if r.get('decision') != 'human_review':
            continue
        packet = {
            'slice_id': r.get('slice_id'),
            'start': r.get('start'),
            'end': r.get('end'),
            'majority_score': r.get('majority_score'),
            'score_distribution': r.get('score_distribution'),
            'semantic_entropy': r.get('semantic_entropy'),
            'score_entropy': r.get('score_entropy'),
            'diagnostic': r.get('diagnostic', {}),
            'samples': [
                {
                    'score': s.get('score'),
                    'reason': s.get('reason'),
                    'evidence': s.get('evidence'),
                }
                for s in r.get('samples', [])
            ],
        }
        packets.append(packet)

    output = {
        'title': 'AI 分歧诊断书',
        'human_review_count': len(packets),
        'records': packets,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
