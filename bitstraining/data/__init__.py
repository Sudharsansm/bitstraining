"""Dataset pipeline: ingest, schema detection, cleaning, dedup, splitting, packing, statistics."""

from bitstraining.data.clean import CleanReport, clean_rows, near_duplicate_groups
from bitstraining.data.loaders import read_rows
from bitstraining.data.pack import ffd_pack, packing_efficiency, padding_efficiency
from bitstraining.data.schema import detect_schema, normalize_row, row_sequences
from bitstraining.data.split import leakage, split_rows
from bitstraining.data.stats import compute_stats, inspect_report

__all__ = [
    "CleanReport",
    "clean_rows",
    "compute_stats",
    "detect_schema",
    "ffd_pack",
    "inspect_report",
    "leakage",
    "near_duplicate_groups",
    "normalize_row",
    "packing_efficiency",
    "padding_efficiency",
    "read_rows",
    "row_sequences",
    "split_rows",
]
