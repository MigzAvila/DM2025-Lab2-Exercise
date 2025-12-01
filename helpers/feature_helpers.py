"""

A fully parameterized text cleaning module for pandas DataFrames.

Features:
- Normalize and sanitize text (HTML unescape, whitespace, emojis)
- Replace URLs, mentions, and name placeholders with standard tokens
- Remove leading quote lines
- Truncate text at edit markers ("edit:", "tl;dr")
- Include missing hashtags if needed
- Fully configurable through function parameters
- Apply cleaning to entire DataFrames with `clean_dataframe`

Usage Example:
----------------
from text_cleaning import clean_dataframe

train_df = clean_dataframe(train_df, text_col="post", hashtag_col="hashtags")
"""

import re
import html
import emoji
import pandas as pd
from typing import Optional, List, Dict, Pattern, Tuple

def split_train_test_data(source_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Splits the source data into training and testing sets based on the presence of labels (split).

    Args:
        source_data (pd.DataFrame): The source DataFrame containing the data with the field split with train and test data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]
        train_df, test_df DataFrames.
    """
    train_df = source_data.query("split == 'train'").dropna(subset=["emotion"]).reset_index(drop=True)
    test_df = source_data.query("split == 'test'").reset_index(drop=True)
    return train_df, test_df

# -------------------------------------------------------------
# Default regex patterns (in dictionary form)
# -------------------------------------------------------------
DEFAULT_PATTERNS: Dict[str, Pattern] = {
    "compact_ws": re.compile(r"\s+"),
    "leading_quote": re.compile(r"^\s*>+\s*"),
    "name_token": re.compile(r"\[NAME\]", flags=re.IGNORECASE),
    "url": re.compile(r"http\S+|www\.\S+"),
    "mention": re.compile(r"@\w+"),
    "edit_mark": re.compile(r"(edit:|tl;dr)", flags=re.IGNORECASE)
}


# -------------------------------------------------------------
# Helper: add hashtag words not found in text
# -------------------------------------------------------------
def add_missing_hashtags(text: str, hashtags: List[str]) -> str:
    """
    Add missing hashtags from the list if they are not already present in the text.

    Parameters
    ----------
    text : str
        Original text.
    hashtags : list of str
        List of hashtags to check and add if missing.

    Returns
    -------
    str
        Text with missing hashtags appended.
    """
    if not hashtags:
        return text

    tokens = set(map(str.lower, re.findall(r"\b\w+\b", text)))

    missing = [
        tag.lstrip("#").strip()
        for tag in hashtags
        if isinstance(tag, str)
        and tag.lstrip("#").strip().lower() not in tokens
    ]

    return f"{text} {' '.join(missing)}" if missing else text


# -------------------------------------------------------------
# Helper: replace tokens using regex patterns
# -------------------------------------------------------------
def replace_tokens(
    text: str,
    patterns: Dict[str, Pattern],
    replace_urls: bool,
    replace_mentions: bool,
    replace_name_tokens: bool
) -> str:
    """
    Replace mentions, URLs, and name placeholders with standardized tokens.

    Parameters
    ----------
    text : str
        Text to process.
    patterns : dict
        Dictionary of regex patterns.
    replace_urls : bool
        Replace URLs with <URL> if True.
    replace_mentions : bool
        Replace @mentions with <USER> if True.
    replace_name_tokens : bool
        Replace [NAME] with <NAME> if True.

    Returns
    -------
    str
        Text with tokens replaced.
    """
    if replace_mentions:
        text = patterns["mention"].sub(" <USER> ", text)

    if replace_urls:
        text = patterns["url"].sub(" <URL> ", text)

    if replace_name_tokens:
        text = patterns["name_token"].sub(" <NAME> ", text)

    return text


# -------------------------------------------------------------
# Helper: truncate at edit/tldr markers
# -------------------------------------------------------------
def slice_at_edit_mark(text: str, patterns: Dict[str, Pattern], enable: bool) -> str:
    """
    Truncate text at the first occurrence of an edit marker (edit: or tl;dr).

    Parameters
    ----------
    text : str
        Text to truncate.
    patterns : dict
        Dictionary of regex patterns.
    enable : bool
        If True, truncation is applied; otherwise, text is returned unchanged.

    Returns
    -------
    str
        Possibly truncated text.
    """
    if not enable:
        return text

    match = patterns["edit_mark"].search(text)
    return text[:match.start()] if match else text


# -------------------------------------------------------------
# Main cleaning function (fully configurable)
# -------------------------------------------------------------
def clean_text(
    text: str,
    hashtags: Optional[List[str]] = None,
    *,
    patterns: Dict[str, Pattern] = DEFAULT_PATTERNS,
    html_unescape_enabled: bool = True,
    normalize_whitespace: bool = True,
    demojize_enabled: bool = True,
    replace_urls: bool = True,
    replace_mentions: bool = True,
    replace_name_tokens: bool = True,
    remove_leading_quotes: bool = True,
    truncate_edit_mark: bool = True,
    add_hashtags: bool = True
) -> str:
    """
    Clean a single text string based on configurable options.

    Parameters
    ----------
    text : str
        The raw text to clean.
    hashtags : list of str, optional
        List of hashtags associated with the text.
    patterns : dict
        Dictionary of compiled regex patterns.
    html_unescape_enabled : bool
        If True, decode HTML entities.
    normalize_whitespace : bool
        If True, normalize whitespace to single spaces.
    demojize_enabled : bool
        If True, convert emojis to textual tokens.
    replace_urls : bool
        If True, replace URLs with <URL>.
    replace_mentions : bool
        If True, replace mentions with <USER>.
    replace_name_tokens : bool
        If True, replace [NAME] tokens with <NAME>.
    remove_leading_quotes : bool
        If True, remove lines starting with '>'.
    truncate_edit_mark : bool
        If True, truncate text at edit markers.
    add_hashtags : bool
        If True, append missing hashtags to the text.

    Returns
    -------
    str
        Cleaned text string.
    """

    if html_unescape_enabled:
        text = html.unescape(text)

    if normalize_whitespace:
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    if demojize_enabled:
        text = emoji.demojize(text, language="en").replace(":", " ")

    text = replace_tokens(
        text,
        patterns=patterns,
        replace_urls=replace_urls,
        replace_mentions=replace_mentions,
        replace_name_tokens=replace_name_tokens
    )

    if remove_leading_quotes:
        text = patterns["leading_quote"].sub("", text)

    text = slice_at_edit_mark(text, patterns, truncate_edit_mark)

    if add_hashtags and isinstance(hashtags, list):
        text = add_missing_hashtags(text, hashtags)

    if normalize_whitespace:
        text = patterns["compact_ws"].sub(" ", text).strip()

    return text


# -------------------------------------------------------------
# DataFrame cleaner (parameterized)
# -------------------------------------------------------------
def clean_dataframe(
    df: pd.DataFrame,
    *,
    text_col: str = "post",
    hashtag_col: str = "hashtags",
    patterns: Dict[str, Pattern] = DEFAULT_PATTERNS,
    **clean_text_kwargs
) -> pd.DataFrame:
    """
    Clean a dataframe by applying the `clean_text` function row-by-row.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the text data to clean.
    text_col : str, default "post"
        Column containing the raw text.
    hashtag_col : str, default "hashtags"
        Column containing a list of hashtags.
    patterns : dict, optional
        Dictionary of compiled regex patterns to use.
    **clean_text_kwargs :
        Additional keyword arguments passed to `clean_text`
        (e.g., demojize_enabled=True, add_hashtags=True).

    Returns
    -------
    pd.DataFrame
        A new dataframe with cleaned text in the specified column.
    """
    result = df.copy()

    result[text_col] = [
        clean_text(
            row[text_col],
            row.get(hashtag_col),
            patterns=patterns,
            **clean_text_kwargs
        )
        for _, row in result.iterrows()
    ]

    return result

__all__ = [
    "clean_text",
    "clean_dataframe",
    "add_missing_hashtags",
    "replace_tokens",
    "slice_at_edit_mark",
    "split_train_test_data"
]
