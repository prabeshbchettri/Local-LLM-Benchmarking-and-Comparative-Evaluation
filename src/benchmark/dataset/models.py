"""
File: models.py

Description:
Pydantic models for the versioned benchmark dataset, evaluation configurations, and probes.

Responsibilities:
- Define categories, difficulties, evaluation methods, and instruction constraints.
- Validate item-level evaluation requirements and stable identifiers.
- Validate dataset counts, distributions, and unanswerable-item placement.
- Compute canonical SHA-256 hashes and deterministic item orderings.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Iterable, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from benchmark.models.base import BaseSchema

ConstraintText = Annotated[str, StringConstraints(min_length=1, strip_whitespace=False)]

DATASET_NAME = "benchmark-v1.0.0"
DATASET_VERSION = "1.0.0"
DATASET_SCHEMA_VERSION = "1.0.0"
DATASET_RANDOM_SEED = 42
DATASET_TOTAL_ITEMS = 100

CATEGORY_COUNTS: dict[str, int] = {
    "general_knowledge": 15,
    "logical_reasoning": 15,
    "mathematical_reasoning": 15,
    "coding": 15,
    "instruction_following": 15,
    "structured_json": 15,
    "summarization": 10,
}

DIFFICULTY_COUNTS: dict[str, int] = {
    "easy": 30,
    "medium": 47,
    "hard": 23,
}

UNANSWERABLE_COUNTS: dict[str, int] = {
    "general_knowledge": 2,
    "logical_reasoning": 1,
    "mathematical_reasoning": 1,
    "instruction_following": 1,
    "summarization": 1,
}

CATEGORY_ID_PREFIX: dict[str, str] = {
    "general_knowledge": "gk",
    "logical_reasoning": "lr",
    "mathematical_reasoning": "mr",
    "coding": "code",
    "instruction_following": "if",
    "structured_json": "json",
    "summarization": "sum",
}

PERFORMANCE_PROBE_VERSION = "1.0.0"
CONSISTENCY_PROBE_VERSION = "1.0.0"
PROBE_SCHEMA_VERSION = "1.0.0"
PERFORMANCE_REPEATS = 5
CONSISTENCY_REPEATS = 3


class Category(StrEnum):
    """Benchmark quality categories."""

    GENERAL_KNOWLEDGE = "general_knowledge"
    LOGICAL_REASONING = "logical_reasoning"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    CODING = "coding"
    INSTRUCTION_FOLLOWING = "instruction_following"
    STRUCTURED_JSON = "structured_json"
    SUMMARIZATION = "summarization"


class Difficulty(StrEnum):
    """Item difficulty labels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvaluationType(StrEnum):
    """Supported deterministic evaluation methods."""

    EXACT_MATCH = "exact_match"
    NUMERIC = "numeric"
    CODE_TESTS = "code_tests"
    JSON_SCHEMA = "json_schema"
    INSTRUCTION_CONSTRAINTS = "instruction_constraints"
    REFERENCE_OVERLAP = "reference_overlap"
    REFUSAL_CHECK = "refusal_check"
    COMPOSITE = "composite"


class ProbeGroup(StrEnum):
    """Performance probe prompt groups."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class WordCountConstraint(BaseSchema):
    """Require a word count within an inclusive range."""

    type: Literal["word_count"]
    minimum: int = Field(ge=1)
    maximum: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> "WordCountConstraint":
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("maximum must be greater than or equal to minimum")
        return self


class SentenceCountConstraint(BaseSchema):
    """Require a sentence count within an inclusive range."""

    type: Literal["sentence_count"]
    minimum: int = Field(ge=1)
    maximum: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> "SentenceCountConstraint":
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("maximum must be greater than or equal to minimum")
        return self


class BulletCountConstraint(BaseSchema):
    """Require an exact number of bullet lines."""

    type: Literal["bullet_count"]
    count: int = Field(ge=1)


class LineCountConstraint(BaseSchema):
    """Require an exact number of non-empty lines."""

    type: Literal["line_count"]
    count: int = Field(ge=1)


class RequiredPhraseConstraint(BaseSchema):
    """Require a case-insensitive phrase."""

    type: Literal["required_phrase"]
    phrase: ConstraintText


class ForbiddenPhraseConstraint(BaseSchema):
    """Forbid a case-insensitive phrase."""

    type: Literal["forbidden_phrase"]
    phrase: ConstraintText


class OccurrenceCountConstraint(BaseSchema):
    """Require an exact phrase occurrence count."""

    type: Literal["occurrence_count"]
    phrase: ConstraintText
    count: int = Field(ge=0)


class StartsWithConstraint(BaseSchema):
    """Require a response prefix."""

    type: Literal["starts_with"]
    text: ConstraintText


class EndsWithConstraint(BaseSchema):
    """Require a response suffix."""

    type: Literal["ends_with"]
    text: ConstraintText


class LowercaseConstraint(BaseSchema):
    """Require lowercase alphabetic characters."""

    type: Literal["lowercase"]


class EachLineStartsWithConstraint(BaseSchema):
    """Require every non-empty line to start with text."""

    type: Literal["each_line_starts_with"]
    text: ConstraintText


class EachLineEndsWithConstraint(BaseSchema):
    """Require every non-empty line to end with text."""

    type: Literal["each_line_ends_with"]
    text: ConstraintText


class EachLineContainsConstraint(BaseSchema):
    """Require every non-empty line to contain text."""

    type: Literal["each_line_contains"]
    text: ConstraintText


class ContainsAllConstraint(BaseSchema):
    """Require all listed terms to appear."""

    type: Literal["contains_all"]
    terms: list[ConstraintText] = Field(min_length=1)


class OrderedTermsConstraint(BaseSchema):
    """Require listed terms to appear in order."""

    type: Literal["ordered_terms"]
    terms: list[ConstraintText] = Field(min_length=1)


class ExactTextConstraint(BaseSchema):
    """Require an exact response string."""

    type: Literal["exact_text"]
    text: ConstraintText


InstructionConstraint = Annotated[
    WordCountConstraint
    | SentenceCountConstraint
    | BulletCountConstraint
    | LineCountConstraint
    | RequiredPhraseConstraint
    | ForbiddenPhraseConstraint
    | OccurrenceCountConstraint
    | StartsWithConstraint
    | EndsWithConstraint
    | LowercaseConstraint
    | EachLineStartsWithConstraint
    | EachLineEndsWithConstraint
    | EachLineContainsConstraint
    | ContainsAllConstraint
    | OrderedTermsConstraint
    | ExactTextConstraint,
    Field(discriminator="type"),
]


class CodeTestCase(BaseSchema):
    """One authored unit test for a coding item."""

    name: str = Field(min_length=1)
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    expected: Any
    description: str | None = None


class DatasetItem(BaseSchema):
    """One validated quality benchmark item."""

    id: str = Field(pattern=r"^[a-z]{2,4}-[0-9]{3}$")
    category: Category
    difficulty: Difficulty
    prompt: str = Field(min_length=20)
    system_instruction: str | None = None
    expected_answer: str | None = None
    evaluation_type: EvaluationType
    reference_answer: str | None = None
    test_cases: list[CodeTestCase] = Field(default_factory=list)
    json_schema: dict[str, Any] | None = None
    required_constraints: list[InstructionConstraint] = Field(default_factory=list)
    maximum_output_tokens: int = Field(default=512, ge=1, le=4096)
    tags: list[str] = Field(default_factory=list)
    is_unanswerable: bool = False
    judge_eligible: bool = False
    max_score: float = Field(default=1.0, gt=0.0)
    source: str = Field(default="authored", min_length=1)
    notes: str | None = None
    accepted_answers: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    function_name: str | None = None
    function_signature: str | None = None
    numeric_tolerance: float | None = Field(default=None, ge=0.0)
    expected_unit: str | None = None

    @model_validator(mode="after")
    def validate_evaluation_requirements(self) -> "DatasetItem":
        prefix = CATEGORY_ID_PREFIX[self.category.value]
        if not self.id.startswith(f"{prefix}-"):
            raise ValueError(
                f"item id {self.id!r} must use the {prefix}- prefix for category "
                f"{self.category.value}"
            )

        if self.evaluation_type is EvaluationType.EXACT_MATCH:
            if not self.expected_answer:
                raise ValueError("exact_match items require expected_answer")
        elif self.evaluation_type is EvaluationType.NUMERIC:
            if not self.expected_answer:
                raise ValueError("numeric items require expected_answer")
            try:
                float(self.expected_answer)
            except ValueError as exc:
                raise ValueError("numeric expected_answer must be a number") from exc
        elif self.evaluation_type is EvaluationType.CODE_TESTS:
            if not self.function_name:
                raise ValueError("code_tests items require function_name")
            if not self.function_signature:
                raise ValueError("code_tests items require function_signature")
            if not self.test_cases:
                raise ValueError("code_tests items require at least one test case")
            if not self.reference_answer:
                raise ValueError("code_tests items require reference_answer")
            names = [case.name for case in self.test_cases]
            if len(set(names)) != len(names):
                raise ValueError("code_tests item test case names must be unique")
        elif self.evaluation_type is EvaluationType.JSON_SCHEMA:
            if not self.json_schema:
                raise ValueError("json_schema items require json_schema")
            if not self.reference_answer:
                raise ValueError("json_schema items require a valid reference_answer")
        elif self.evaluation_type is EvaluationType.INSTRUCTION_CONSTRAINTS:
            if not self.required_constraints:
                raise ValueError(
                    "instruction_constraints items require required_constraints"
                )
        elif self.evaluation_type is EvaluationType.REFERENCE_OVERLAP:
            if not self.reference_answer:
                raise ValueError("reference_overlap items require reference_answer")
            if not self.key_facts:
                raise ValueError("reference_overlap items require key_facts")
        elif self.evaluation_type is EvaluationType.REFUSAL_CHECK:
            if not self.reference_answer:
                raise ValueError("refusal_check items require reference_answer")
            if not self.forbidden_claims:
                raise ValueError("refusal_check items require forbidden_claims")
        elif self.evaluation_type is EvaluationType.COMPOSITE:
            if not self.reference_answer:
                raise ValueError("composite items require reference_answer")

        if self.is_unanswerable and self.evaluation_type is not EvaluationType.REFUSAL_CHECK:
            raise ValueError("unanswerable items must use refusal_check")
        if self.evaluation_type is EvaluationType.REFUSAL_CHECK and not self.is_unanswerable:
            raise ValueError("refusal_check items must set is_unanswerable")

        if self.evaluation_type is not EvaluationType.CODE_TESTS:
            if self.test_cases:
                raise ValueError("test_cases are only valid for code_tests items")
            if self.function_name or self.function_signature:
                raise ValueError(
                    "function_name and function_signature are only valid for code_tests items"
                )
        if self.evaluation_type is not EvaluationType.JSON_SCHEMA and self.json_schema is not None:
            raise ValueError("json_schema is only valid for json_schema items")
        if self.evaluation_type is not EvaluationType.NUMERIC:
            if self.numeric_tolerance is not None:
                raise ValueError("numeric_tolerance is only valid for numeric items")
            if self.expected_unit is not None:
                raise ValueError("expected_unit is only valid for numeric items")
        return self

    def evaluation_config(self) -> dict[str, Any]:
        """Return the method-specific configuration used by later evaluators."""
        config: dict[str, Any] = {"evaluation_type": self.evaluation_type.value}
        if self.accepted_answers:
            config["accepted_answers"] = list(self.accepted_answers)
        if self.numeric_tolerance is not None:
            config["tolerance"] = self.numeric_tolerance
        if self.expected_unit is not None:
            config["expected_unit"] = self.expected_unit
        if self.test_cases:
            config["test_cases"] = [case.model_dump(mode="json") for case in self.test_cases]
        if self.json_schema is not None:
            config["json_schema"] = self.json_schema
        if self.required_constraints:
            config["constraints"] = [
                constraint.model_dump(mode="json") for constraint in self.required_constraints
            ]
        if self.key_facts:
            config["key_facts"] = list(self.key_facts)
        if self.forbidden_claims:
            config["forbidden_claims"] = list(self.forbidden_claims)
        return config


class BenchmarkDataset(BaseSchema):
    """Versioned container for the complete 100-item quality dataset."""

    name: str = Field(min_length=1)
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    schema_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    created_at: datetime
    random_seed: int = DATASET_RANDOM_SEED
    description: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    items: list[DatasetItem] = Field(
        min_length=DATASET_TOTAL_ITEMS,
        max_length=DATASET_TOTAL_ITEMS,
    )

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_distributions(self) -> "BenchmarkDataset":
        if len(self.items) != DATASET_TOTAL_ITEMS:
            raise ValueError(f"dataset must contain exactly {DATASET_TOTAL_ITEMS} items")

        ids = [item.id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("dataset item ids must be unique")

        prompts = [" ".join(item.prompt.lower().split()) for item in self.items]
        if len(set(prompts)) != len(prompts):
            raise ValueError("dataset prompts must be unique")

        category_counts = self.category_distribution()
        if category_counts != CATEGORY_COUNTS:
            raise ValueError(
                f"category distribution must be {CATEGORY_COUNTS}, got {category_counts}"
            )

        difficulty_counts = self.difficulty_distribution()
        if difficulty_counts != DIFFICULTY_COUNTS:
            raise ValueError(
                f"difficulty distribution must be {DIFFICULTY_COUNTS}, got {difficulty_counts}"
            )

        unanswerable = self.unanswerable_distribution()
        expected_unanswerable = {
            category: count
            for category, count in UNANSWERABLE_COUNTS.items()
            if count
        }
        actual_unanswerable = {
            category: count for category, count in unanswerable.items() if count
        }
        if actual_unanswerable != expected_unanswerable:
            raise ValueError(
                "unanswerable distribution must be "
                f"{expected_unanswerable}, got {actual_unanswerable}"
            )

        if self.random_seed != DATASET_RANDOM_SEED:
            raise ValueError(f"random_seed must be {DATASET_RANDOM_SEED}")
        return self

    def category_distribution(self) -> dict[str, int]:
        """Return item counts for every category in canonical order."""
        counts = {category.value: 0 for category in Category}
        for item in self.items:
            counts[item.category.value] += 1
        return counts

    def difficulty_distribution(self) -> dict[str, int]:
        """Return item counts for every difficulty in canonical order."""
        counts = {difficulty.value: 0 for difficulty in Difficulty}
        for item in self.items:
            counts[item.difficulty.value] += 1
        return counts

    def evaluation_type_distribution(self) -> dict[str, int]:
        """Return item counts for every evaluation type in canonical order."""
        counts = {evaluation.value: 0 for evaluation in EvaluationType}
        for item in self.items:
            counts[item.evaluation_type.value] += 1
        return counts

    def unanswerable_distribution(self) -> dict[str, int]:
        """Return unanswerable item counts for every category in canonical order."""
        counts = {category.value: 0 for category in Category}
        for item in self.items:
            if item.is_unanswerable:
                counts[item.category.value] += 1
        return counts

    def ordered_items(self, seed: int | None = None) -> list[DatasetItem]:
        """Return a deterministic seeded shuffle of the dataset items."""
        items = list(self.items)
        random.Random(self.random_seed if seed is None else seed).shuffle(items)
        return items


class PerformancePrompt(BaseSchema):
    """One fixed performance probe prompt."""

    id: str = Field(pattern=r"^P[1-9]$")
    group: ProbeGroup
    prompt: str = Field(min_length=20)
    target_output_words: int = Field(ge=1)
    maximum_output_tokens: int = Field(default=512, ge=1, le=4096)

    @model_validator(mode="after")
    def validate_group(self) -> "PerformancePrompt":
        number = int(self.id[1:])
        expected = (
            ProbeGroup.SHORT
            if number <= 3
            else ProbeGroup.MEDIUM
            if number <= 6
            else ProbeGroup.LONG
        )
        if self.group is not expected:
            raise ValueError(f"{self.id} must belong to the {expected.value} group")
        return self


class PerformanceProbe(BaseSchema):
    """Versioned performance probe containing nine fixed prompts."""

    name: str = Field(min_length=1)
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    schema_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    created_at: datetime
    random_seed: int = DATASET_RANDOM_SEED
    repeats: int = Field(default=PERFORMANCE_REPEATS, ge=1)
    description: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompts: list[PerformancePrompt] = Field(min_length=9, max_length=9)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_prompts(self) -> "PerformanceProbe":
        ids = [prompt.id for prompt in self.prompts]
        if len(set(ids)) != len(ids):
            raise ValueError("performance prompt ids must be unique")
        for group in ProbeGroup:
            count = sum(1 for prompt in self.prompts if prompt.group is group)
            if count != 3:
                raise ValueError(f"performance probe must contain three {group.value} prompts")
        return self


class ConsistencyProbeItem(BaseSchema):
    """One dataset item selected for repeated consistency measurement."""

    item_id: str = Field(pattern=r"^[a-z]{2,4}-[0-9]{3}$")
    category: Category
    prompt: str = Field(min_length=20)
    repeats: int = Field(default=CONSISTENCY_REPEATS, ge=2)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1)
    seed: int = DATASET_RANDOM_SEED
    repeat_penalty: float = Field(default=1.1, gt=0.0)
    num_ctx: int = Field(default=4096, ge=512)
    maximum_output_tokens: int = Field(default=512, ge=1, le=4096)


class ConsistencyProbe(BaseSchema):
    """Versioned consistency probe with ten stratified dataset items."""

    name: str = Field(min_length=1)
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    schema_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    created_at: datetime
    random_seed: int = DATASET_RANDOM_SEED
    selection_method: str = Field(default="seeded_stratified", min_length=1)
    repeats: int = Field(default=CONSISTENCY_REPEATS, ge=2)
    description: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    items: list[ConsistencyProbeItem] = Field(min_length=10, max_length=10)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_items(self) -> "ConsistencyProbe":
        ids = [item.item_id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("consistency probe item ids must be unique")
        categories = {item.category for item in self.items}
        if categories != set(Category):
            raise ValueError("consistency probe must cover every category")
        return self


def canonical_payload(value: Any) -> str:
    """Serialize a value with stable key ordering and compact separators."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_payload_hash(payloads: Iterable[Any]) -> str:
    """Compute a SHA-256 hash over a sequence of canonical payloads."""
    return hashlib.sha256(canonical_payload(list(payloads)).encode("utf-8")).hexdigest()


def item_payload(item: DatasetItem) -> dict[str, Any]:
    """Return the canonical JSON payload for one item.

    Top-level null fields are omitted, but a code test's ``expected`` value is
    always retained so ``None`` remains a distinct, hash-covered expected result.
    """
    payload = item.model_dump(mode="json", exclude_none=True)
    if item.test_cases:
        test_cases = []
        for case in item.test_cases:
            case_payload = case.model_dump(mode="json", exclude_none=True)
            case_payload["expected"] = case.expected
            test_cases.append(case_payload)
        payload["test_cases"] = test_cases
    return payload


def compute_dataset_content_hash(items: Iterable[DatasetItem]) -> str:
    """Compute the dataset hash over items sorted by stable identifier."""
    payloads = [item_payload(item) for item in sorted(items, key=lambda item: item.id)]
    return compute_payload_hash(payloads)


def performance_prompt_payload(prompt: PerformancePrompt) -> dict[str, Any]:
    """Return the canonical JSON payload for one performance prompt."""
    return prompt.model_dump(mode="json", exclude_none=True)


def compute_performance_probe_hash(prompts: Iterable[PerformancePrompt]) -> str:
    """Compute the performance probe hash over prompts sorted by identifier."""
    payloads = [
        performance_prompt_payload(prompt)
        for prompt in sorted(prompts, key=lambda prompt: prompt.id)
    ]
    return compute_payload_hash(payloads)


def consistency_item_payload(item: ConsistencyProbeItem) -> dict[str, Any]:
    """Return the canonical JSON payload for one consistency item."""
    return item.model_dump(mode="json", exclude_none=True)


def compute_consistency_probe_hash(items: Iterable[ConsistencyProbeItem]) -> str:
    """Compute the consistency probe hash over items sorted by dataset identifier."""
    payloads = [
        consistency_item_payload(item)
        for item in sorted(items, key=lambda item: item.item_id)
    ]
    return compute_payload_hash(payloads)
