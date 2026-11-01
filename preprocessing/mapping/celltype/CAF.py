from mapping.common import normalize_label


PROVENANCE = "GSE210347:CAF"
FINE_COLUMNS = ["cluster", "CAF_subtype", "subtype"]

MAPPING = {
    "c1": "CAF_myo",
    "c2": "CAF_infla",
    "c3": "Fibroblasts",
    "c4": "CAF_adi",
    "c5": "Fibroblasts",
    "c6": "CAF_EndoMT",
    "c7": "CAF_PN",
    "c8": "CAF_api",
    "cafmyo": "CAF_myo",
    "cafinfla": "CAF_infla",
    "cafadi": "CAF_adi",
    "cafendmt": "CAF_EndoMT",
    "cafendomt": "CAF_EndoMT",
    "cafpn": "CAF_PN",
    "cafap": "CAF_api",
    "cafapi": "CAF_api",
}


def map_celltype(value):
    return MAPPING.get(normalize_label(value))
