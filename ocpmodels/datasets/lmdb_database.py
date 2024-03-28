"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is modified from the ASE db json backend
and is thus licensed under the corresponding LGPL2.1 license

The ASE notice for the LGPL2.1 license is available here:
https://gitlab.com/ase/ase/-/blob/master/LICENSE
"""

from __future__ import annotations

import os
import typing
import zlib
from pathlib import Path
from typing import Optional

import lmdb
import numpy as np
import orjson
from ase.db.core import Database, now, ops
from ase.db.row import AtomsRow
from typing_extensions import Self

if typing.TYPE_CHECKING:
    from ase import Atoms


# These are special keys in the ASE LMDB that hold
# metadata and other info
RESERVED_KEYS = ["nextid", "metadata", "deleted_ids"]


class LMDBDatabase(Database):
    def __init__(
        self,
        filename: Optional[str | Path] = None,
        create_indices: bool = True,
        use_lock_file: bool = False,
        serial: bool = False,
        readonly: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        For the most part, this is identical to the standard ase db initiation
        arguments, except that we add a readonly flag.
        """
        super().__init__(
            Path(filename),
            create_indices,
            use_lock_file,
            serial,
            *args,
            **kwargs,
        )

        # Add a readonly mode for when we're only training
        # to make sure there's no parallel locks
        self.readonly = readonly

        if self.readonly:
            # Open a new env
            self.env = lmdb.open(
                str(self.filename),
                subdir=False,
                meminit=False,
                map_async=True,
                readonly=True,
                lock=False,
            )

            # Open a transaction and keep it open for fast read/writes!
            self.txn = self.env.begin(write=False)

        else:
            # Open a new env with write access
            self.env = lmdb.open(
                str(self.filename),
                map_size=1099511627776 * 2,
                subdir=False,
                meminit=False,
                map_async=True,
            )

            self.txn = self.env.begin(write=True)

        # Load all ids based on keys in the DB.
        self.ids = []
        self.deleted_ids = []
        self._load_ids()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, tb) -> None:
        self.close()

    def close(self) -> None:
        # Close the lmdb environment and transaction
        self.txn.commit()
        self.env.close()

    def _write(
        self,
        atoms: Atoms | AtomsRow,
        key_value_pairs: dict,
        data: Optional[dict],
        idx: Optional[int] = None,
    ) -> None:
        Database._write(self, atoms, key_value_pairs, data)

        mtime = now()

        if isinstance(atoms, AtomsRow):
            row = atoms
        else:
            row = AtomsRow(atoms)
            row.ctime = mtime
            row.user = os.getenv("USER")

        dct = {}
        for key in row.__dict__:
            if key[0] == "_" or key in row._keys or key == "id":
                continue
            dct[key] = row[key]

        dct["mtime"] = mtime

        if key_value_pairs:
            dct["key_value_pairs"] = key_value_pairs

        if data:
            dct["data"] = data

        constraints = row.get("constraints")
        if constraints:
            dct["constraints"] = [
                constraint.todict() for constraint in constraints
            ]

        # json doesn't like Cell objects, so make it an array
        dct["cell"] = np.asarray(dct["cell"])

        if idx is None:
            idx = self._nextid
            nextid = idx + 1
        else:
            data = self.txn.get(f"{idx}".encode("ascii"))
            assert data is not None

        # Add the new entry
        self.txn.put(
            f"{idx}".encode("ascii"),
            zlib.compress(
                orjson.dumps(dct, option=orjson.OPT_SERIALIZE_NUMPY)
            ),
        )
        # only append if idx is not in ids
        if idx not in self.ids:
            self.ids.append(idx)
            self.txn.put(
                "nextid".encode("ascii"),
                zlib.compress(
                    orjson.dumps(nextid, option=orjson.OPT_SERIALIZE_NUMPY)
                ),
            )
        # check if id is in removed ids and remove accordingly
        if idx in self.deleted_ids:
            self.deleted_ids.remove(idx)
            self._write_deleted_ids()

        return idx

    def _update(
        self,
        idx: int,
        key_value_pairs: Optional[dict] = None,
        data: Optional[dict] = None,
    ):
        # hack this to play nicely with ASE code
        row = self._get_row(idx, include_data=True)
        if data is not None or key_value_pairs is not None:
            self._write(
                atoms=row, idx=idx, key_value_pairs=key_value_pairs, data=data
            )

    def _write_deleted_ids(self):
        self.txn.put(
            "deleted_ids".encode("ascii"),
            zlib.compress(
                orjson.dumps(
                    self.deleted_ids, option=orjson.OPT_SERIALIZE_NUMPY
                )
            ),
        )

    def delete(self, ids: list[int]) -> None:
        for idx in ids:
            self.txn.delete(f"{idx}".encode("ascii"))
            self.ids.remove(idx)

        self.deleted_ids += ids
        self._write_deleted_ids()

    def _get_row(self, idx: int, include_data: bool = True):
        if idx is None:
            assert len(self.ids) == 1
            idx = self.ids[0]
        data = self.txn.get(f"{idx}".encode("ascii"))

        if data is not None:
            dct = orjson.loads(zlib.decompress(data))
        else:
            raise KeyError(f"Id {idx} missing from the database!")

        if not include_data:
            dct.pop("data", None)

        dct["id"] = idx
        return AtomsRow(dct)

    def _get_row_by_index(self, index: int, include_data: bool = True):
        """Auxiliary function to get the ith entry, rather than a specific id"""
        data = self.txn.get(f"{self.ids[index]}".encode("ascii"))

        if data is not None:
            dct = orjson.loads(zlib.decompress(data))
        else:
            raise KeyError(f"Id {id} missing from the database!")

        if not include_data:
            dct.pop("data", None)

        dct["id"] = id
        return AtomsRow(dct)

    def _select(
        self,
        keys,
        cmps: list[tuple[str, str, str]],
        explain: bool = False,
        verbosity: int = 0,
        limit: Optional[int] = None,
        offset: int = 0,
        sort: Optional[str] = None,
        include_data: bool = True,
        columns: str = "all",
    ):
        if explain:
            yield {"explain": (0, 0, 0, "scan table")}
            return

        if sort is not None:
            if sort[0] == "-":
                reverse = True
                sort = sort[1:]
            else:
                reverse = False

            rows = []
            missing = []
            for row in self._select(keys, cmps):
                key = row.get(sort)
                if key is None:
                    missing.append((0, row))
                else:
                    rows.append((key, row))

            rows.sort(reverse=reverse, key=lambda x: x[0])
            rows += missing

            if limit:
                rows = rows[offset : offset + limit]
            for key, row in rows:
                yield row
            return

        if not limit:
            limit = -offset - 1

        cmps = [(key, ops[op], val) for key, op, val in cmps]
        n = 0
        for idx in self.ids:
            if n - offset == limit:
                return
            row = self._get_row(idx, include_data=include_data)

            for key in keys:
                if key not in row:
                    break
            else:
                for key, op, val in cmps:
                    if isinstance(key, int):
                        value = np.equal(row.numbers, key).sum()
                    else:
                        value = row.get(key)
                        if key == "pbc":
                            assert op in [ops["="], ops["!="]]
                            value = "".join("FT"[x] for x in value)
                    if value is None or not op(value, val):
                        break
                else:
                    if n >= offset:
                        yield row
                    n += 1

    @property
    def metadata(self):
        """Load the metadata from the DB if present"""
        if self._metadata is None:
            metadata = self.txn.get("metadata".encode("ascii"))
            if metadata is None:
                self._metadata = {}
            else:
                self._metadata = orjson.loads(zlib.decompress(metadata))

        return self._metadata.copy()

    @metadata.setter
    def metadata(self, dct):
        self._metadata = dct

        # Put the updated metadata dictionary
        self.txn.put(
            "metadata".encode("ascii"),
            zlib.compress(
                orjson.dumps(dct, option=orjson.OPT_SERIALIZE_NUMPY)
            ),
        )

    @property
    def _nextid(self):
        """Get the id of the next row to be written"""
        # Get the nextid
        nextid_data = self.txn.get("nextid".encode("ascii"))
        nextid = (
            orjson.loads(zlib.decompress(nextid_data)) if nextid_data else 1
        )
        return nextid

    def count(self, selection=None, **kwargs) -> int:
        """Count rows.

        See the select() method for the selection syntax.  Use db.count() or
        len(db) to count all rows.
        """
        if selection is not None:
            n = 0
            for row in self.select(selection, **kwargs):
                n += 1
            return n
        else:
            # Fast count if there's no queries! Just get number of ids
            return len(self.ids)

    def _load_ids(self) -> None:
        """Load ids from the DB

        Since ASE db ids are mostly 1-N integers, but can be missing entries
        if ids have been deleted. To save space and operating under the assumption
        that there will probably not be many deletions in most OCP datasets,
        we just store the deleted ids.
        """

        # Load the deleted ids
        deleted_ids_data = self.txn.get("deleted_ids".encode("ascii"))
        if deleted_ids_data is not None:
            self.deleted_ids = orjson.loads(zlib.decompress(deleted_ids_data))

        # Reconstruct the full id list
        self.ids = [
            i for i in range(1, self._nextid) if i not in set(self.deleted_ids)
        ]
