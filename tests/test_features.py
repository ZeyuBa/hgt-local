from src.graph.feature_extraction import FeatureBundle, build_feature_bundle
from training_data.topo_complete import generate_complete_sample


def test_build_feature_bundle_returns_typed_structure_with_direct_lookup_maps():
    sample = generate_complete_sample(seed=70)

    bundle = build_feature_bundle(sample)

    assert isinstance(bundle, FeatureBundle)
    assert bundle.node_features.shape[0] == len(bundle.node_ids)
    assert bundle.node_type.shape[0] == len(bundle.node_ids)

    first_entity_id = sample["alarm_entities"][0]["id"]
    entity_position = bundle.alarm_entity_id_to_position[first_entity_id]
    assert bundle.alarm_entity_ids[entity_position] == first_entity_id

    ae_node_index = bundle.ae_node_index_by_id[first_entity_id]
    assert bundle.node_ids[ae_node_index] == first_entity_id


def test_feature_bundle_exposes_owner_and_alarm_node_indices_for_edge_building():
    sample = generate_complete_sample(seed=71)

    bundle = build_feature_bundle(sample)
    first_entity = sample["alarm_entities"][0]

    owner_index = bundle.ne_id_to_index[first_entity["ne_id"]]
    ae_node_index = bundle.ae_node_index_by_id[first_entity["id"]]
    alarm_node_index = bundle.alarm_name_to_index[first_entity["alarm_name"]]

    assert bundle.ae_owner_ne_indices[bundle.alarm_entity_id_to_position[first_entity["id"]]] == owner_index
    assert bundle.node_ids[ae_node_index] == first_entity["id"]
    assert bundle.node_ids[alarm_node_index] == f"alarm:{first_entity['alarm_name']}"
