"""Project-wide constants for synthetic alarm graph generation and modeling."""

from __future__ import annotations

from typing import Final

NODE_TYPE_IDS: Final[dict[str, int]] = {
    "wl_station": 0,
    "phy_site": 1,
    "router": 2,
    "alarm_entity": 3,
    "alarm": 4,
    "padding": 5,
}

ALARM_IDS: Final[dict[str, int]] = {
    "ne_is_disconnected": 0,
    "mains_failure": 1,
    "device_powered_off": 2,
    "link_down": 3,
}

ALARM_DEFINITIONS: Final[dict[str, dict[str, int | str | bool]]] = {
    "ne_is_disconnected": {
        "alarm_id": ALARM_IDS["ne_is_disconnected"],
        "node_type": "wl_station",
        "domain": 1002,
        "trainable": True,
    },
    "mains_failure": {
        "alarm_id": ALARM_IDS["mains_failure"],
        "node_type": "phy_site",
        "domain": 500,
        "trainable": False,
    },
    "device_powered_off": {
        "alarm_id": ALARM_IDS["device_powered_off"],
        "node_type": "router",
        "domain": 600,
        "trainable": False,
    },
    "link_down": {
        "alarm_id": ALARM_IDS["link_down"],
        "node_type": "router",
        "domain": 400,
        "trainable": False,
    },
}

RELATION_TYPE_IDS: Final[dict[str, int]] = {
    "ne_alarm_entity": 0,
    "rev_ne_alarm_entity": 1,
    "alarm_entity_alarm": 2,
    "rev_alarm_entity_alarm": 3,
    "co_site_ne_ne": 4,
    "rev_co_site_ne_ne": 5,
    "cross_site_ne_ne": 6,
    "rev_cross_site_ne_ne": 7,
    "self": 8,
}

HGT_NODE_TYPE_IDS: Final[dict[str, int]] = {
    "ne": 0,
    "alarm_entity": 1,
    "alarm": 2,
}

NE_SUBTYPE_IDS: Final[dict[str, int]] = {
    "wl_station": 0,
    "phy_site": 1,
    "router": 2,
}

FORWARD_TO_REVERSE_RELATION: Final[dict[str, str]] = {
    "ne_alarm_entity": "rev_ne_alarm_entity",
    "alarm_entity_alarm": "rev_alarm_entity_alarm",
    "co_site_ne_ne": "rev_co_site_ne_ne",
    "cross_site_ne_ne": "rev_cross_site_ne_ne",
}

NODE_TYPES_WITH_FEATURES: Final[tuple[str, ...]] = (
    "wl_station",
    "phy_site",
    "router",
    "alarm_entity",
    "alarm",
    "padding",
)

NE_NODE_TYPES: Final[tuple[str, ...]] = ("wl_station", "phy_site", "router")

SITE_LEVEL_FLAGS: Final[tuple[str, ...]] = (
    "is_an",
    "is_fault_or_risk_anchor",
    "is_padding",
)

ROLE_IDS: Final[dict[str, int]] = {"ne": 0, "alarm_entity": 1, "alarm": 2}
