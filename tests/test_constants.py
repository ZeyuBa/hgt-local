from src.graph.feature_extraction import (
    ALARM_DEFINITIONS,
    ALARM_IDS,
    NODE_TYPE_IDS,
    RELATION_TYPE_IDS,
)


def test_node_type_ids_are_stable():
    assert NODE_TYPE_IDS == {
        "wl_station": 0,
        "phy_site": 1,
        "router": 2,
        "alarm_entity": 3,
        "alarm": 4,
        "padding": 5,
    }


def test_alarm_ids_and_domains_are_stable():
    assert ALARM_IDS == {
        "ne_is_disconnected": 0,
        "mains_failure": 1,
        "device_powered_off": 2,
        "link_down": 3,
    }
    assert ALARM_DEFINITIONS["ne_is_disconnected"]["domain"] == 1002
    assert ALARM_DEFINITIONS["mains_failure"]["node_type"] == "phy_site"
    assert ALARM_DEFINITIONS["device_powered_off"]["node_type"] == "router"
    assert ALARM_DEFINITIONS["link_down"]["node_type"] == "router"


def test_relation_types_cover_forward_reverse_and_self():
    expected = {
        "ne_alarm_entity",
        "rev_ne_alarm_entity",
        "alarm_entity_alarm",
        "rev_alarm_entity_alarm",
        "co_site_ne_ne",
        "rev_co_site_ne_ne",
        "cross_site_ne_ne",
        "rev_cross_site_ne_ne",
        "self",
    }
    assert set(RELATION_TYPE_IDS) == expected
    assert len(RELATION_TYPE_IDS) == 9
