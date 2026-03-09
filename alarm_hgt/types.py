"""Typed records used across synthetic generation and dataset loading."""

from __future__ import annotations

from typing import Literal, TypedDict


NodeType = Literal["wl_station", "phy_site", "router", "alarm_entity", "alarm", "padding"]
RelationType = Literal[
    "ne_alarm_entity",
    "alarm_entity_alarm",
    "co_site_ne_ne",
    "cross_site_ne_ne",
    "self",
]
AlarmName = Literal[
    "ne_is_disconnected",
    "mains_failure",
    "device_powered_off",
    "link_down",
]


class NodeRecord(TypedDict, total=False):
    id: str
    type: NodeType
    site_id: str
    is_an: bool
    is_outage: bool
    is_fault_or_risk_anchor: bool
    is_padding: bool
    is_router: bool
    an_site_id: str | None


class EdgeRecord(TypedDict):
    source: str
    target: str
    relation: RelationType


class AlarmEntityRecord(TypedDict, total=False):
    id: str
    ne_id: str
    site_id: str
    alarm_name: AlarmName
    alarm_id: int
    domain: int
    label: int
    is_trainable_alarm: bool
    owner_is_an: bool
    owner_is_fault_or_risk_anchor: bool
    owner_is_padding: bool
