##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from __future__ import annotations

import glob
import gzip
import hashlib
import itertools
import json
import os
import pickle
import random
import re
import struct
import tarfile
import warnings
from collections.abc import Sequence
from operator import itemgetter
from typing import Any, Callable, Optional, Union

import nibabel
import numpy as np
from torch.utils.data import Dataset

MODALITIES = ("t1", "t2", "flair")
_INDEX_VERSION = 3
_DATASET_RE = re.compile(r"ds\d+")
_MEMBER_RE = re.compile(
    r"(?P<key>.+)\.(?P<label>meta(?=\.json$)|(?:t1|t2|flair)(?:_mask)?"
    r"(?=\.nii(?:\.gz)?$))"
)


class OpenMind(Dataset):
    """OpenMind dataset [1]_, read from WebDataset shards.

    The OpenMind dataset is a large-scale 3D MRI dataset of the head and neck
    region featuring 114k MRI Images. This data was pooled from exactly 800
    datasets from the OpenNeuro platform and provides 23 different MRI
    modalities from over 30 different scanners, representing a highly variable
    pre-training dataset.

    The current version includes 43001 T1w, 23012 T2w and 4614 FLAIR images,
    preprocessed with the quasi-raw `brainprep
    <https://github.com/brainprepdesk/brainprep>`_ workflow (skull-stripping,
    N4 bias field correction, 1mm isotropic resampling and linear registration
    to MNI space).

    The shards are written by ``bids_to_webdataset``: one sample per
    (subject, session, modality, instance), where an instance is a distinct
    combination of the BIDS entities ``run``, ``acq``, ``rec``, ``dir`` and
    ``echo``. Shards where several modalities are bundled in a single sample
    (older layout) are read as well. Deciding which images go together is
    done here, at load time, with `group_by`, `require_all` and `instance`,
    so it can be changed without rewriting the shards.

    Parameters
    ----------
    root : str
        Directory containing the ``.tar`` shards.

    modality : None, str or sequence of str, default=None
        Modalities to keep among "t1", "t2" and "flair". None keeps all of
        them. A string returns a single image per item; a sequence returns a
        dict {modality: image} per item, see `group_by`.

    group_by : None, {"session", "subject"} or callable, default=None
        How images are regrouped into items:

        - None: one item per image, whatever its modality.
        - "session": one item per (dataset, subject, session), holding one
          image per requested modality (e.g. T1w + T2w acquired in the same
          session).
        - "subject": same, across all the sessions of a subject. Subject
          labels are only unique within a dataset, so the dataset id (e.g.
          "ds000030", parsed from the shard name) is part of the group.
        - callable: takes the metadata dict of an image (see `return_meta`)
          and returns a hashable group key.

        Without grouping, an item is always a single array, even if several
        modalities are requested. With grouping, an item is a dict
        {modality: image} unless `modality` is a single string.

    require_all : bool, default=True
        Only relevant with `group_by`. If True, groups that miss one of the
        requested modalities are dropped. If False, they are kept and the
        missing modalities are set to None (use a custom `collate_fn` with a
        `DataLoader`).

    instance : {"first", "random", "all"}, default="first"
        Only relevant with `group_by`. When a group holds several images of
        the same modality (several runs, acquisitions...):

        - "first": keep the first one in sorted key order (the image
          without run/acq/... entities comes first).
        - "random": draw one at each `__getitem__` call.
        - "all": one item per combination (cartesian product).

    subjects : sequence of str or (str, str), default=None
        Keep only these subjects, given either as "sub-01" (matches this
        label in every dataset) or as ("ds000030", "sub-10159"). Subject
        labels are only unique within a dataset. Use it to define
        train/validation splits at the subject level, which avoids leakage
        between sessions or modalities of the same subject.

    apply_mask : bool, default=False
        If True, multiply each image by its brain mask. Images without mask
        are returned unchanged.

    return_meta : bool, default=False
        If True, `__getitem__` returns ``(data, meta)`` where `meta` is the
        ``meta.json`` dict of the sample (subject, session, modality,
        entities, source paths) plus its ``key``. For grouped dict items,
        `meta` is a dict {modality: meta}.

    shards : str, default="*.tar"
        Glob pattern (relative to `root`) selecting the shards. A warning is
        raised if the same image appears in several of them.

    index_path : str or None, default=None
        Where the shard index is cached. Building it requires a pass over the
        tar headers, but is done only once. Default is
        ``<root>/openmind_index_<hash of shards>.pkl``. If not writable, the
        index is simply rebuilt at each initialization.

    transforms : callable or None, default=None
        A function/transform that takes in an image (or a dict of images) and
        returns a transformed version.

    Examples
    --------
    One T1w image per item:

    >>> dataset = OpenMind(root="data/openmind_wds", modality="t1")
    >>> t1 = dataset[0]
    >>> print(t1.shape)
    (1, 182, 218, 182)

    Paired T1w/T2w acquired in the same session:

    >>> dataset = OpenMind(
    ...     root="data/openmind_wds",
    ...     modality=("t1", "t2"),
    ...     group_by="session",
    ... )
    >>> sample = dataset[0]
    >>> print(sample["t1"].shape, sample["t2"].shape)
    (1, 182, 218, 182) (1, 182, 218, 182)

    One image per subject, drawn among all modalities:

    >>> dataset = OpenMind(
    ...     root="data/openmind_wds",
    ...     group_by=lambda meta: meta["subject"],
    ...     require_all=False,
    ...     instance="random",
    ... )

    Notes
    -----
    Shards are indexed with the standard library (``tarfile``) and images are
    read by byte offsets, so random access works with any sampler and any
    number of `DataLoader` workers. The ``webdataset`` package is not needed
    to read the data.

    References
    ----------
    .. [1] Wald et al., An OpenMind for 3D medical vision self-supervised
           learning, arXiv 2025
    """

    def __init__(
        self,
        root: str,
        modality: Optional[Union[str, Sequence[str]]] = None,
        group_by: Optional[Union[str, Callable[[dict], Any]]] = None,
        require_all: bool = True,
        instance: str = "first",
        subjects: Optional[Sequence[str]] = None,
        apply_mask: bool = False,
        return_meta: bool = False,
        shards: str = "*.tar",
        index_path: Optional[str] = None,
        transforms: Optional[Callable] = None,
    ):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.modalities = self._parse_modality(modality)
        self.group_by = self._parse_group_by(group_by)
        self.require_all = require_all
        if instance not in ("first", "random", "all"):
            raise ValueError(
                f"instance must be 'first', 'random' or 'all', got "
                f"{instance!r}"
            )
        self.instance = instance
        self.apply_mask = apply_mask
        self.return_meta = return_meta
        self.transforms = transforms
        # A dict per item only makes sense when regrouping several modalities
        self._as_dict = group_by is not None and not isinstance(modality, str)

        self.shards = sorted(glob.glob(os.path.join(self.root, shards)))
        if not self.shards:
            raise FileNotFoundError(
                f"No shard matching {shards!r} found in {self.root}"
            )
        if index_path is None:
            tag = hashlib.md5(shards.encode()).hexdigest()[:8]
            index_path = os.path.join(self.root, f"openmind_index_{tag}.pkl")
        self.entries = self._load_index(index_path)

        keep = set(subjects) if subjects is not None else None
        self._warn_duplicates()
        self._selected = [
            i
            for i, e in enumerate(self.entries)
            if e["meta"]["modality"] in self.modalities
            and (keep is None or self._is_kept(e["meta"], keep))
        ]
        self.items = self._make_items()
        self._handles: dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        """Load one item.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        data : array or dict of array
            A float32 array of shape `(1, X, Y, Z)`, or a dict
            {modality: array} for grouped multimodal items.
        meta : dict, optional
            Only if `return_meta` is True.
        """
        chosen = {}
        for modality, candidates in self.items[idx].items():
            if not candidates:
                chosen[modality] = None
            elif self.instance == "random" and len(candidates) > 1:
                chosen[modality] = random.choice(candidates)
            else:
                chosen[modality] = candidates[0]

        images, metas = {}, {}
        for modality, entry_idx in chosen.items():
            if entry_idx is None:
                images[modality], metas[modality] = None, None
                continue
            entry = self.entries[entry_idx]
            images[modality] = self._load_image(entry)
            metas[modality] = {**entry["meta"], "key": entry["key"]}

        if self._as_dict:
            data, meta = images, metas
        else:
            # Exactly one modality per item (either ungrouped or a single
            # requested modality).
            (data,), (meta,) = images.values(), metas.values()
        if self.transforms is not None:
            data = self.transforms(data)
        return (data, meta) if self.return_meta else data

    def __str__(self) -> str:
        return (
            f"OpenMind(root={self.root!r}, modalities={self.modalities}, "
            f"group_by={self.group_by!r}, instance={self.instance!r}, "
            f"n_items={len(self)})"
        )

    def __getstate__(self):
        # File handles cannot be pickled (DataLoader workers, spawn).
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self):
        for handle in getattr(self, "_handles", {}).values():
            handle.close()

    def _make_items(self) -> list[dict[str, tuple[int, ...]]]:
        """Regroup entries into items: {modality: candidate entry indices}."""
        if self.group_by is None:
            return [
                {self.entries[i]["meta"]["modality"]: (i,)}
                for i in self._selected
            ]

        groups: dict[Any, dict[str, list[int]]] = {}
        for i in self._selected:
            meta = self.entries[i]["meta"]
            group = groups.setdefault(self.group_by(meta), {})
            group.setdefault(meta["modality"], []).append(i)

        items = []
        for group in groups.values():
            if self.require_all and any(
                m not in group for m in self.modalities
            ):
                continue
            # Missing modalities get an empty candidate list -> None
            group = {
                m: tuple(
                    sorted(
                        group.get(m, ()), key=lambda i: self.entries[i]["key"]
                    )
                )
                for m in self.modalities
            }
            if self.instance == "first":
                items.append({m: c[:1] for m, c in group.items()})
            elif self.instance == "random":
                items.append(group)
            else:  # "all"
                pools = [c or (None,) for c in group.values()]
                items.extend(
                    {
                        m: (() if c is None else (c,))
                        for m, c in zip(group, combo)
                    }
                    for combo in itertools.product(*pools)
                )
        return items

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def _read(self, shard: int, offset: int, size: int) -> bytes:
        handle = self._handles.get(shard)
        if handle is None:
            handle = self._handles[shard] = open(  # ruff: ignore[open-file-with-context-handler]
                self.shards[shard], "rb"
            )
        handle.seek(offset)
        return handle.read(size)

    def _load_image(self, entry: dict) -> np.ndarray:
        image = self._decode(self._read(entry["shard"], *entry["image"]))
        if self.apply_mask and entry["mask"] is not None:
            mask = self._decode(self._read(entry["shard"], *entry["mask"]))
            image = image * (mask > 0)
        return image

    @staticmethod
    def _decode(buf: bytes) -> np.ndarray:
        """Decode the bytes of a .nii or .nii.gz file to `(1, X, Y, Z)`."""
        # Some shards were written with a double gzip layer.
        while buf[:2] == b"\x1f\x8b":
            buf = gzip.decompress(buf)
        # sizeof_hdr is 348 for NIfTI-1 and 540 for NIfTI-2, in either
        # endianness.
        sizes = {struct.unpack(f"{e}i", buf[:4])[0] for e in "<>"}
        cls = nibabel.Nifti2Image if 540 in sizes else nibabel.Nifti1Image
        data = cls.from_bytes(buf).get_fdata(dtype=np.float32)
        return data[None]

    def _load_index(self, index_path: str) -> list[dict]:
        """Load the cached shard index, rebuilding it if stale."""
        signature = [
            (os.path.basename(p), os.stat(p).st_size, os.stat(p).st_mtime_ns)
            for p in self.shards
        ]
        try:
            with open(index_path, "rb") as f:
                cached = pickle.load(f)
            if (
                cached["version"] == _INDEX_VERSION
                and cached["signature"] == signature
            ):
                return cached["entries"]
        except (
            OSError,
            KeyError,
            TypeError,
            pickle.UnpicklingError,
            EOFError,
        ):
            pass  # missing, corrupted or outdated cache

        entries = []
        for shard_idx, path in enumerate(self.shards):
            entries.extend(self._index_shard(path, shard_idx))
        try:
            tmp = f"{index_path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                pickle.dump(
                    {
                        "version": _INDEX_VERSION,
                        "signature": signature,
                        "entries": entries,
                    },
                    f,
                )
            os.replace(tmp, index_path)  # atomic: safe with several ranks
        except OSError:  # read-only root
            pass
        return entries

    @staticmethod
    def _index_shard(path: str, shard_idx: int) -> list[dict]:
        """Record the byte offsets of every image of a shard.

        Two layouts are supported: one modality per sample (``modality`` in
        ``meta.json``, files ``<modality>.nii.gz`` and
        ``<modality>_mask.nii.gz``), or several modalities bundled in one
        sample (``t1.nii.gz``, ``t2.nii.gz``... next to a single
        ``meta.json``). Either way, one entry is created per image.
        """
        found = _DATASET_RE.search(os.path.basename(path))
        dataset = found.group(0) if found else None
        samples: dict[str, dict] = {}
        with tarfile.open(path, "r:") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                # Match known suffixes rather than splitting at the first dot
                # like webdataset: keys may contain dots (e.g. "ses-1.5T").
                found = _MEMBER_RE.match(os.path.basename(member.name))
                if found is None:
                    continue
                key, label = found.group("key"), found.group("label")
                sample = samples.setdefault(key, {})
                if label == "meta":
                    sample["meta"] = json.load(tar.extractfile(member))
                else:
                    sample[label] = (member.offset_data, member.size)

        entries = []
        for key, sample in samples.items():
            meta = sample.get("meta")
            if meta is None:
                continue
            modalities = (
                [meta["modality"]]
                if "modality" in meta
                else [m for m in MODALITIES if m in sample]
            )
            for modality in modalities:
                if modality not in sample:
                    continue  # mask without its image
                entries.append(
                    {
                        "key": key,
                        "shard": shard_idx,
                        "meta": {
                            "run": None,
                            "entities": None,
                            **meta,
                            "modality": modality,
                            "dataset": dataset,
                        },
                        "image": sample[modality],
                        "mask": sample.get(f"{modality}_mask"),
                    }
                )
        return entries

    def _warn_duplicates(self) -> None:
        """Warn if the same image is indexed twice (e.g. a stale shard)."""
        seen: dict[tuple, int] = {}
        for e in self.entries:
            m = e["meta"]
            ident = (
                m["dataset"],
                m["subject"],
                m["session"],
                m["modality"],
                tuple(sorted((m["entities"] or {}).items())),
            )
            seen[ident] = seen.get(ident, 0) + 1
        dups = [i for i, n in seen.items() if n > 1]
        if dups:
            warnings.warn(
                f"{len(dups)} images are present in several shards, e.g. "
                f"{dups[0]}. Restrict `shards` to avoid duplicates.",
                stacklevel=3,
            )

    @staticmethod
    def _is_kept(meta: dict, keep: set) -> bool:
        return (
            meta["subject"] in keep
            or (meta["dataset"], meta["subject"]) in keep
        )

    @staticmethod
    def _parse_modality(modality) -> tuple[str, ...]:
        if modality is None:
            return MODALITIES
        modalities = (modality,) if isinstance(modality, str) else modality
        modalities = tuple(m.lower() for m in modalities)
        unknown = set(modalities) - set(MODALITIES)
        if unknown or not modalities:
            raise ValueError(
                f"Unknown modality {sorted(unknown)}, expected among "
                f"{MODALITIES}"
            )
        return modalities

    @staticmethod
    def _parse_group_by(group_by) -> Optional[Callable[[dict], Any]]:
        if group_by is None or callable(group_by):
            return group_by
        if group_by == "session":
            return itemgetter("dataset", "subject", "session")
        if group_by == "subject":
            return itemgetter("dataset", "subject")
        raise ValueError(
            f"group_by must be None, 'session', 'subject' or a callable, "
            f"got {group_by!r}"
        )
