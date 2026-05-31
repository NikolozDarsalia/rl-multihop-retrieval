"""A tiny synthetic corpus shaped exactly like MuSiQue-Ans records, so the same
foundation (and later graph) code runs end to end without downloading anything.

It deliberately reuses distractor paragraphs across questions (to exercise
dedup) and plants a known bridge structure: the previous hop's answer appears
verbatim in the next hop's gold paragraph, mirroring the ~80% bridge coverage
you measured in the EDA.
"""

from __future__ import annotations
import random


def _para(idx: int, title: str, text: str, supporting: bool = False) -> dict:
    return {
        "idx": idx,
        "title": title,
        "paragraph_text": text,
        "is_supporting": supporting,
    }


def make_fixture(
    n_questions: int = 12, n_paragraphs: int = 20, seed: int = 0
) -> dict[str, list]:
    rng = random.Random(seed)

    # a small shared pool of distractor paragraphs, reused across questions
    pool = [
        {"title": f"Topic {c}", "text": f"Generic paragraph about topic {c}. " * 3}
        for c in range(30)
    ]

    records = []
    for q in range(n_questions):
        n_hops = rng.choice([2, 2, 2, 3, 4])
        entities = [f"Entity_{q}_{h}" for h in range(n_hops)]

        gold_texts = []
        for h in range(n_hops):
            t = f"{entities[h]} is a subject in scenario S{q}."
            if h >= 1:
                t += f" It is linked to {entities[h - 1]}."
            t += f" It was founded in {1900 + h}."
            gold_texts.append(t)

        # intermediate answers are the next bridge entity; final answer is a value
        answers = [
            entities[h + 1] if h + 1 < n_hops else f"{1900 + (n_hops - 1)}"
            for h in range(n_hops)
        ]

        gold_indices = sorted(rng.sample(range(n_paragraphs), n_hops))

        paragraphs, gi = [], 0
        for idx in range(n_paragraphs):
            if gi < n_hops and idx == gold_indices[gi]:
                paragraphs.append(
                    _para(idx, entities[gi], gold_texts[gi], supporting=True)
                )
                gi += 1
            else:
                d = rng.choice(pool)
                paragraphs.append(_para(idx, d["title"], d["text"], supporting=False))

        decomp = [
            {
                "id": h,
                "question": f"sub-question {h}",
                "answer": answers[h],
                "paragraph_support_idx": gold_indices[h],
            }
            for h in range(n_hops)
        ]

        records.append(
            {
                "id": f"{n_hops}hop__fixture_{q}",
                "question": f"Question {q} about {entities[0]}?",
                "answer": answers[-1],
                "answer_aliases": [],
                "paragraphs": paragraphs,
                "question_decomposition": decomp,
                "answerable": True,
            }
        )

    rng.shuffle(records)
    split = max(1, int(0.8 * n_questions))
    return {"train": records[:split], "validation": records[split:]}
