import os
import pickle
import pandas as pd
from typing import Any, Dict, List, Optional, Union


class InstructRecDataset:
    """
    A unified wrapper for InstructRec datasets (Books, MovieTV, Yelp, Reads).

    This class normalizes all dataset formats (list, dict, DataFrame)
    into a consistent pandas DataFrame and provides convenient accessors
    for sampling, column inspection, item retrieval, and list-based fields
    such as `asin` and `ranked_lists`.

    Each dataset entry typically includes:
        - reviewText      (list of review snippets)
        - title           (list of item titles)
        - description     (list of item descriptions)
        - instruction     (user instruction/query)
        - persona         (persona description)
        - asin            (list of candidate item IDs)
        - ranked_lists    (ranked list of recommendation labels)
    """

    def __init__(self, path: str):
        """
        Load and normalize a .pkl InstructRec dataset.

        Automatically detects whether the underlying object is a list,
        dict, or DataFrame and converts it into a standard DataFrame.

        Parameters
        ----------
        path : str
            Path to the .pkl dataset file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        self.path = path

        # Load raw object
        with open(path, "rb") as f:
            raw = pickle.load(f)

        # Normalize into DataFrame
        if isinstance(raw, pd.DataFrame):
            self.data = raw
        elif isinstance(raw, list):
            self.data = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            self.data = pd.DataFrame(raw)
        else:
            raise ValueError(
                f"Unexpected dataset type ({type(raw)}). "
                "Expected list, dict, or pandas DataFrame."
            )

        # Clean index
        self.data.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Access a full dataset entry by index.

        Parameters
        ----------
        idx : int
            Row index.

        Returns
        -------
        dict
            A dictionary representing the dataset entry.
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range.")
        return self.data.iloc[idx].to_dict()

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_column(self, name: str) -> List[Any]:
        """
        Return a full column as a list.

        Useful for text fields, list-based fields,
        or supervision labels (ranked lists, personas, etc.).
        """
        if name not in self.data.columns:
            raise KeyError(f"Column '{name}' not found.")
        return self.data[name].tolist()

    def get_item_field(self, idx: int, field: str) -> Any:
        """
        Return a specific field from a given entry.

        Parameters
        ----------
        idx : int
            Row index.
        field : str
            Column name.

        Returns
        -------
        Any
            Field value.
        """
        item = self.__getitem__(idx)
        if field not in item:
            raise KeyError(f"Field '{field}' does not exist in this dataset.")
        return item[field]

    def list_columns(self) -> List[str]:
        """Return all available dataset columns."""
        return list(self.data.columns)

    def to_pandas(self) -> pd.DataFrame:
        """
        Return a copy of the underlying pandas DataFrame.

        This guarantees that external edits do not mutate
        the dataset’s internal state.
        """
        return self.data.copy()

    def filter_by_keyword(self, field: str, keyword: str) -> pd.DataFrame:
        """
        Filter entries where the provided field contains a given keyword.

        Useful for exploratory analysis or building subsets.

        Parameters
        ----------
        field : str
            Column to search.
        keyword : str
            Substring to match.

        Returns
        -------
        DataFrame
            Filtered subset.
        """
        if field not in self.data:
            raise KeyError(f"Field '{field}' not found.")
        return self.data[self.data[field].astype(str)
                         .str.contains(keyword, case=False)]

    # -------------------------------------------------------------
    # Field getters (ALL columns)
    # -------------------------------------------------------------

    def get_review_text(self, idx: int) -> List[str]:
        """
        Return the list of review text snippets for the given entry.
        """
        return self.get_item_field(idx, "reviewText")

    def get_titles(self, idx: int) -> List[str]:
        """
        Return the list of candidate item titles for the given entry.
        """
        return self.get_item_field(idx, "title")

    def get_descriptions(self, idx: int) -> List[str]:
        """
        Return the list of candidate item descriptions for the given entry.
        """
        return self.get_item_field(idx, "description")

    def get_instruction(self, idx: int) -> str:
        """
        Return the user's instruction/query for the given entry.
        """
        return self.get_item_field(idx, "instruction")

    def get_persona(self, idx: int) -> str:
        """
        Return the persona description for the given entry.
        """
        return self.get_item_field(idx, "persona")

    def get_asins(self, idx: int) -> List[str]:
        """
        Return the list of candidate item identifiers (ASIN IDs).
        """
        return self.get_item_field(idx, "asin")

    def get_ranked_list(self, idx: int) -> List[Any]:
        """
        Return the ranked list (pseudo-ground truth ordering)
        associated with the given entry.
        """
        return self.get_item_field(idx, "ranked_lists")


    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        """Readable representation showing path, size, and columns."""
        return (
            "InstructRecDataset(\n"
            f"  path = '{self.path}',\n"
            f"  size = {len(self)},\n"
            f"  columns = {self.list_columns()}\n"
            ")"
        )

import pandas as pd


class ItemIndex:
    """
    Index-based lookup for Yelp items used in the InstructRec dataset.

    This class loads the CSV file `combined_yelp_asin_mapping.csv`,
    which contains real item metadata (index, asin, title, description),
    and exposes accessor methods based on the value in the `index` column,
    not the DataFrame row position.
    """

    def __init__(self, csv_path: str):
        """
        Load the mapping file into memory and build an index lookup table.

        Expected columns:
            - index (or Index)
            - asin
            - title
            - description
        """
        df = pd.read_csv(csv_path)

        # Normalize the name of the index column
        if "index" in df.columns:
            index_col = "index"
        elif "Index" in df.columns:
            df = df.rename(columns={"Index": "index"})
            index_col = "index"
        else:
            raise ValueError("CSV file must contain an 'index' or 'Index' column.")

        required = {"index", "asin", "title", "description"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV file is missing required columns: {missing}")

        self.df = df
        self.index_col = index_col

        # Build a dictionary: index_value -> metadata
        self._by_index = {}
        for _, row in df.iterrows():
            key = int(row[self.index_col])
            self._by_index[key] = {
                "asin": row["asin"],
                "title": row["title"],
                "description": row["description"],
            }

    # ---------------------------------------------------------
    # Index-based getters (using the value in the `index` column)
    # ---------------------------------------------------------

    def get_item_by_index(self, index_value: int) -> dict:
        """
        Return the full item metadata for a given `index` value
        (as stored in the CSV column `index`), not the row position.

        Parameters
        ----------
        index_value : int
            Value from the `index` column.

        Returns
        -------
        dict
            Example:
            {
                "asin": "...",
                "title": "...",
                "description": "..."
            }

        Raises
        ------
        KeyError
            If no item exists for the given index.
        """
        index_value = int(index_value)
        if index_value not in self._by_index:
            raise KeyError(f"No item found for index {index_value}.")
        return self._by_index[index_value]

    def get_asin(self, index_value: int) -> str:
        """Return the ASIN for the given `index` value."""
        return self.get_item_by_index(index_value)["asin"]

    def get_title(self, index_value: int) -> str:
        """Return the item title for the given `index` value."""
        return self.get_item_by_index(index_value)["title"]

    def get_description(self, index_value: int) -> str:
        """Return the item description for the given `index` value."""
        return self.get_item_by_index(index_value)["description"]

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of items in the mapping."""
        return len(self.df)

    def __repr__(self):
        return (
            "ItemIndex(\n"
            f"  size={len(self)},\n"
            f"  columns={list(self.df.columns)}\n"
            ")"
        )
