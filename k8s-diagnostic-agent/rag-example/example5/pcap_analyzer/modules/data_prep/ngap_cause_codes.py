from typing import Dict, Tuple, Optional


def get_ngap_cause_text(category: Optional[str], value: Optional[str]) -> str:
    """Return human-readable NGAP cause text given category and value per 38.413.

    Category is one of: radioNetwork, transport, nas, protocol, misc
    Value is a snake_case identifier aligned with spec names.
    """
    if not category or not value:
        return "Unknown Cause"

    maps: Dict[str, Dict[str, str]] = {
        'misc': {
            'control_processing_overload': 'Misc - Control Processing Overload',
            'not_enough_user_plane_processing_resources': 'Misc - Not Enough User Plane Processing Resources',
            'hardware_failure': 'Misc - Hardware Failure',
            'om_intervention': 'Misc - OM Intervention',
            'unknown_plmn': 'Misc - Unknown PLMN',
            'misc_unspecified': 'Misc - Unspecified',
        },
        'nas': {
            'normal_release': 'NAS - Normal Release',
            'authentication_failure': 'NAS - Authentication Failure',
            'deregister': 'NAS - Deregister',
            'nas_unspecified': 'NAS - Unspecified',
        },
        'protocol': {
            'transfer_syntax_error': 'Protocol - Transfer Syntax Error',
            'abstract_syntax_error_reject': 'Protocol - Abstract Syntax Error Reject',
            'abstract_syntax_error_ignore_and_notify': 'Protocol - Abstract Syntax Error Ignore and Notify',
            'message_not_compatible_with_receiver_state': 'Protocol - Message not Compatible With Receiver State',
            'semantic_error': 'Protocol - Semantic Error',
            'abstract_syntax_error_falsely_constructed_message': 'Protocol - Abstract Syntax Error Falsely Constructed Message',
            'protocol_unspecified': 'Protocol - Unspecified',
        },
        'radioNetwork': {
            'radio_network_unspecified': 'Radio Network - Unspecified',
            'txnrelocoverall_expiry': 'Radio Network - txnrelocoverall Expiry',
            'successful_handover': 'Radio Network - Successful Handover',
            'release_due_to_ngran_generated_reason': 'Radio Network - Release Due to NGRAN Generated Reason',
            'release_due_to_5gc_generated_reason': 'Radio Network - Release Due to 5GC Generated Reason',
            'handover_cancelled': 'Radio Network - Handover Cancelled',
            'partial_handover': 'Radio Network - Partial Handover',
            'ho_failure_in_target_5gc_ngran_node_or_target_system': 'Radio Network - HO Failure in Target 5GC NGRAN Node or Target System',
            'ho_target_not_allowed': 'Radio Network - HO Target Not Allowed',
            'tngrelocoverall_expiry': 'Radio Network - tngrelocoverall Expiry',
            'tngrelocprep_expiry': 'Radio Network - tngrelocprep Expiry',
            'cell_not_available': 'Radio Network - Cell Not Available',
            'unknown_targetID': 'Radio Network - Unknown TargetID',
            'no_radio_resources_available_in_target_cell': 'Radio Network - No Radio Resources Available in Target Cell',
            'unknown_local_ue_ngap_id': 'Radio Network - Unknown Local UE NGAP ID',
            'inconsistent_remote_ue_ngap_id': 'Radio Network - Inconsistent Remote UE NGAP ID',
            'handover_desirable_for_radio_reason': 'Radio Network - Handover Desirable for Radio Reason',
            'time_critical_handover': 'Radio Network - Time Critical Handover',
            'resource_optimisation_handover': 'Radio Network - Resource Optimisation Handover',
            'reduce_load_in_serving_cell': 'Radio Network - Reduce Load in Serving Cell',
            'user_inactivity': 'Radio Network - User Inactivity',
            'radio_connection_with_ue_lost': 'Radio Network - Radio Connection With UE Lost',
            'load_balancing_tau_required': 'Radio Network - Load Balancing TAU Required',
            'radio_resources_not_available': 'Radio Network - Radio Resources Not Available',
            'invalid_qos_combination': 'Radio Network - Invalid QoS Combination',
            'failure_in_radio_interface_procedure': 'Radio Network - Failure in Radio Interface Procedure',
            'interaction_with_other_procedure': 'Radio Network - Interaction With Other Procedure',
            'radio_network_unknown_pdu_session_id': 'Radio Network - Unknown PDU Session ID',
            'unkown_qos_flow_id': 'Radio Network - Unknown QoS Flow Id',
            'radio_network_multiple_pdu_session_id_instances': 'Radio Network - Multiple PDU Session Id Instances',
            'multiple_qos_flow_id_instances': 'Radio Network - Multiple QoS Flow Id Instances',
            'encryption_and_or_integrity_protection_algorithms_not_supported': 'Radio Network - Encryption and/or Integrity Protection Algorithms Not Supported',
            'ng_intra_system_handover_triggered': 'Radio Network - NG Intra System Handover Triggered',
            'ng_inter_system_handover_triggered': 'Radio Network - NG Inter System Handover Triggered',
            'xn_handover_triggered': 'Radio Network - XN Handover Triggered',
            'not_supported_5qi_value': 'Radio Network - Not Supported 5QI Value',
            'ue_context_transfer': 'Radio Network - UE Context Transfer',
            'ims_voice_eps_fallback_or_rat_fallback_triggered': 'Radio Network - IMS Voice EPS Fallback or RAT Fallback Triggered',
            'up_integrity_protection_not_possible': 'Radio Network - UP Integrity Protection Not Possible',
            'up_confidentiality_protection_not_possible': 'Radio Network - UP Confidentiality Protection Not Possible',
            'slice_not_supported': 'Radio Network - Slice Not Supported',
            'ue_in_rrc_inactive_state_not_reachable': 'Radio Network - UE In RRC Inactive State Not Reached',
            'redirection': 'Radio Network - Redirection',
            'resources_not_available_for_the_slice': 'Radio Network - Resources Not Available for the Slice',
            'ue_max_integrity_protected_data_rate_reason': 'Radio Network - UE Max Integrity Protected Data Rate Reason',
            'release_due_to_cn_detected_mobility': 'Radio Network - Release Due to CN Detected Mobility',
            'n26_interface_not_available': 'Radio Network - N26 Interface Not Available',
            'release_due_to_pre_emption': 'Radio Network - Release Due to Pre-Emption',
            'multiple_location_reporting_reference_id_instances': 'Radio Network - Multiple Location Reporting Reference Id Instances',
            'rsn_not_available_for_the_up': 'Radio Network - RSN Not Available for the UP',
            'npn_access_denied': 'Radio Network - NPN Access Denied',
            'cag_only_access_denied': 'Radio Network - CAG Only Access Denied',
        },
        'transport': {
            'transport_resource_unavailable': 'Transport - Transport Resource Unavailable',
            'transport_network_unspecified': 'Transport - Network Unspecified',
        },
    }

    category_map = maps.get(category, {})
    return category_map.get(value, f"{category} - {value}")


