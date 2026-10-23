"""
eval/metrics/business_rules_metrics.py — Métricas de clasificación y tasa de falsos positivos (FPR).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class RuleConfusionMatrix:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def record(self, predicted: bool, actual: bool) -> None:
        if predicted and actual:
            self.tp += 1
        elif predicted and not actual:
            self.fp += 1
        elif not predicted and not actual:
            self.tn += 1
        else:
            self.fn += 1

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return (self.tp / denom) if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return (self.tp / denom) if denom > 0 else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    @property
    def fpr(self) -> float:
        """False Positive Rate = FP / (FP + TN)"""
        denom = self.fp + self.tn
        return (self.fp / denom) if denom > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fpr": round(self.fpr, 4),
        }


class BenchmarkEvaluator:
    def __init__(self) -> None:
        self.rules: Dict[str, RuleConfusionMatrix] = {}

    def get_rule_matrix(self, rule_tag: str) -> RuleConfusionMatrix:
        if rule_tag not in self.rules:
            self.rules[rule_tag] = RuleConfusionMatrix()
        return self.rules[rule_tag]

    def record_prediction(self, rule_tag: str, predicted: bool, ground_truth: bool) -> None:
        matrix = self.get_rule_matrix(rule_tag)
        matrix.record(predicted, ground_truth)

    def evaluate_thresholds(
        self,
        max_p95_ms: float = 65.0,
        max_fpr: float = 0.05,
        min_recall: float = 0.90,
        p95_ms: float = 0.0,
    ) -> Dict[str, Any]:
        passed = True
        reasons = []

        if p95_ms > max_p95_ms:
            passed = False
            reasons.append(f"Latencia p95 ({p95_ms}ms) excede umbral ({max_p95_ms}ms)")

        for tag, mat in self.rules.items():
            if mat.fpr > max_fpr:
                passed = False
                reasons.append(f"Regla '{tag}' FPR ({mat.fpr*100:.1f}%) excede umbral ({max_fpr*100:.1f}%)")
            if (mat.tp + mat.fn > 0) and (mat.recall < min_recall):
                passed = False
                reasons.append(f"Regla '{tag}' Recall ({mat.recall*100:.1f}%) menor a ({min_recall*100:.1f}%)")

        return {
            "passed": passed,
            "verdict": "PASS" if passed else "FAIL",
            "failures": reasons,
            "rules": {tag: mat.to_dict() for tag, mat in self.rules.items()},
        }
