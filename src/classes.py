"""
Lesiv 10-class harmonisation used throughout the experiment (Section 2.2).

The 13 categories of the original Lesiv et al. (2025) reference dataset are
reduced to 10 by excluding three minor or non-target classes (burnt land,
fallow, and 'not sure'), aligned with the WorldCover main classes.
"""

# Original Lesiv class IDs -> 10-class label name.
LESIV_CLASSES = {
    3024: "tree_cover",
    3025: "shrubland",
    3026: "grassland",
    3027: "cropland",
    3028: "built_up",
    3029: "bare_sparse_vegetation",
    3031: "permanent_water",
    3032: "snow_and_ice",
    3471: "herbaceous_wetland",
    4074: "moss_lichen",
}

# Excluded original Lesiv class IDs.
LESIV_EXCLUDE_IDS = {
    3030,   # burnt land
    3033,   # fallow
    3034,   # not sure
}

# Canonical class ordering (used for arrays, confusion matrices, plots).
CLASS_ORDER = [
    "tree_cover",
    "shrubland",
    "grassland",
    "cropland",
    "built_up",
    "bare_sparse_vegetation",
    "permanent_water",
    "snow_and_ice",
    "herbaceous_wetland",
    "moss_lichen",
]

NAME_TO_LABEL = {name: i for i, name in enumerate(CLASS_ORDER)}
LABEL_TO_NAME = {i: name for i, name in enumerate(CLASS_ORDER)}
CLASS_ID_TO_LABEL = {cid: NAME_TO_LABEL[name]
                     for cid, name in LESIV_CLASSES.items()}

VALID_CLASS_IDS = sorted(LESIV_CLASSES.keys())

assert len(CLASS_ORDER) == 10
assert set(LESIV_CLASSES.values()) == set(CLASS_ORDER)
