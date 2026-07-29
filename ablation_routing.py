"""
Torch-free ablation variant routing (importable without PyTorch).
"""

ABLATION_VARIANTS = [
    "Full", "w_o_TS", "w_o_Gate", "w_o_Contra",
    "w_o_UGG", "w_o_KD", "w_o_UGG_KD", "w_o_TCAM",
]

_UNDERSCORE_ALIAS = {
    "w_o_TS": "w/o_TS",
    "w_o_Gate": "w/o_Gate",
    "w_o_Contra": "w/o_Contra",
    "w_o_UGG": "w/o_UGG",
    "w_o_KD": "w/o_KD",
    "w_o_UGG_KD": "w/o_UGG_KD",
    "w_o_TCAM": "w/o_TCAM",
}

STRUCTURAL_VARIANTS = {"w/o_TS", "w/o_Gate", "w_o_TS", "w_o_Gate"}

_NO_KD_VARIANTS = {"w/o_KD", "w/o_UGG_KD", "w_o_KD", "w_o_UGG_KD"}


def resolve_variant_name(name: str) -> str:
    return _UNDERSCORE_ALIAS.get(name, name)


def is_structural_variant(name: str) -> bool:
    return resolve_variant_name(name) in STRUCTURAL_VARIANTS


def get_use_kd(name: str) -> bool:
    return resolve_variant_name(name) not in _NO_KD_VARIANTS
