"""
Text normalization, metadata standardization, and feature soup construction for LitLens.
Preserves scientific terminology, acronyms, and original fields while constructing
weighted representations for TF-IDF indexing.
"""

import html
import os
import re
from typing import Optional, Any, List
import pandas as pd
from src.data_loader import extract_base_id_and_version


# Regex patterns for scientific text normalization
RE_LATEX_COMMANDS = re.compile(r"\\[a-zA-Z]+(?:\{([^}]*)\})?")
RE_MATH_BLOCKS = re.compile(r"\${1,2}(.*?)\${1,2}")
RE_XML_TAGS = re.compile(r"<[^>]+>")
RE_WHITESPACE = re.compile(r"\s+")
# Token pattern preserving technical expressions, hyphenated words, acronyms, numbers (e.g., resnet-50, 3d, l1-norm)
TOKEN_PATTERN = r"(?u)\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*\b"


def normalize_text(text: Optional[str]) -> str:
    """
    Normalizes scientific text while preserving technical terminology:
    - Decodes HTML/XML entities
    - Strips LaTeX command markup while keeping the inner expression
    - Retains hyphenated technical terms, acronyms, and alphanumeric identifiers
    - Converts to lowercase and collapses whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    # Decode HTML entities (e.g., &amp;, &lt;, &gt;)
    t = html.unescape(text)

    # Remove XML/HTML tags
    t = RE_XML_TAGS.sub(" ", t)

    # Flatten LaTeX math blocks to the contained text
    t = RE_MATH_BLOCKS.sub(r" \1 ", t)

    # Strip LaTeX commands like \textbf{term} -> term, \alpha -> alpha
    t = RE_LATEX_COMMANDS.sub(r" \1 ", t)

    # Standardize curly quotes and unicode dashes
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("`", "'")
    t = t.replace("—", "-").replace("–", "-")

    # Lowercase
    t = t.lower()

    # Collapse excess whitespace
    t = RE_WHITESPACE.sub(" ", t).strip()

    return t


def standardize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the metadata schema while strictly preserving original title,
    summary, authors, and date fields.
    Adds parsed dates, base_id, version, and clean author display fields.
    """
    # Work on a shallow copy
    proc = df.copy()

    # Parse dates safely
    proc["published_date_parsed"] = pd.to_datetime(
        proc["published_date"], format="mixed", errors="coerce"
    )
    proc["updated_date_parsed"] = pd.to_datetime(
        proc["updated_date"], format="mixed", errors="coerce"
    )
    proc["published_year"] = proc["published_date_parsed"].dt.year.fillna(0).astype(int)

    # Extract base_id and version
    base_versions = proc["id"].apply(extract_base_id_and_version)
    proc["base_id"] = [bv[0] for bv in base_versions]
    proc["version"] = [bv[1] for bv in base_versions]

    # Flag latest version per base_id
    max_versions = proc.groupby("base_id")["version"].transform("max")
    proc["is_latest_version"] = proc["version"] == max_versions

    # Ensure clean string representation for authors display
    def _clean_authors_str(authors_val: Any) -> str:
        s = str(authors_val).strip()
        if s.startswith("[") and s.endswith("]"):
            # Clean list format string: "['Author 1', 'Author 2']"
            cleaned = re.sub(r"[\'\"\[\]]", "", s)
            return ", ".join([a.strip() for a in cleaned.split(",") if a.strip()])
        return s

    proc["authors_display"] = proc["authors"].apply(_clean_authors_str)

    return proc


def build_weighted_soup(
    df: pd.DataFrame,
    title_weight: int = 3,
    include_category: bool = True,
) -> pd.Series:
    """
    Creates a weighted text representation for scientific papers.
    Titles are given greater importance by repeating them `title_weight` times.
    Categories provide explicit domain context.
    Abstract summaries provide the rich semantic grounding.
    """
    titles_norm = df["title"].fillna("").apply(normalize_text)
    summaries_norm = df["summary"].fillna("").apply(normalize_text)

    # Construct weighted soup: (title * title_weight) + category + summary
    weighted_title = (titles_norm + " ") * max(1, title_weight)

    if include_category and "category" in df.columns:
        categories_norm = df["category"].fillna("").apply(normalize_text)
        soup = weighted_title + categories_norm + " " + summaries_norm
    else:
        soup = weighted_title + summaries_norm

    return soup.str.strip()


def save_processed_catalog(
    df: pd.DataFrame,
    output_path: str = "models/papers_catalog.parquet",
) -> None:
    """
    Saves the standardized catalog as a compressed Parquet file for high-performance retrieval.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)


def load_processed_catalog(
    catalog_path: str = "models/papers_catalog.parquet",
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Loads the processed catalog from Parquet.
    """
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Processed catalog not found at: {catalog_path}")
    return pd.read_parquet(catalog_path, engine="pyarrow", columns=columns)
