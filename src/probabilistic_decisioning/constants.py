"""Shared constants for the Bank Marketing offline pipeline."""

BANK_MARKETING_INPUT_FIELDS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]
BANK_MARKETING_NUMERIC_FEATURE_NAMES = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
]
BANK_MARKETING_CATEGORICAL_FEATURE_NAMES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
BANK_MARKETING_EXPECTED_FIELD_COUNT = len(BANK_MARKETING_INPUT_FIELDS) + 1
BANK_MARKETING_LEAKAGE_FEATURE_NAME = "duration"
DEFAULT_DATASET_NAME = "bank_marketing"
DEFAULT_FEATURE_SET_VERSION = "bank_marketing_v1"
DEFAULT_HASH_DIMENSION = 1_048_576
DEFAULT_START_TIMESTAMP = "2026-01-01T00:00:00+00:00"
