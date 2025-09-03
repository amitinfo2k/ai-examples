import logging
from typing import Optional, Dict, Any


class NGAPDecoder:
    """Thin wrapper around an NGAP ASN.1 PER decoder.

    This implementation attempts to use an external ASN.1 toolkit if available
    (e.g., asn1tools) and a provided NGAP ASN.1 schema path. If not available,
    it gracefully disables decoding and callers should fall back to heuristics.
    """

    def __init__(self, schema_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.decoder = None
        self.schema_loaded = False

        if not schema_path:
            self.logger.info("NGAP ASN.1 schema path not provided; decoder disabled.")
            return
        try:
            import asn1tools  # type: ignore
        except Exception as ex:
            self.logger.warning(f"asn1tools not available, NGAP decoding disabled: {ex}")
            return

        try:
            # NGAP uses aligned PER
            self.decoder = asn1tools.compile_files(schema_path, codec='per')
            self.schema_loaded = True
            self.logger.info(f"Loaded NGAP ASN.1 schema from {schema_path}")
        except Exception as ex:
            self.logger.warning(f"Failed to compile NGAP ASN.1 schema: {ex}")
            self.decoder = None
            self.schema_loaded = False

    def is_available(self) -> bool:
        return self.decoder is not None and self.schema_loaded

    def decode_pdu(self, payload_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Decode an NGAP PDU, returning a structured dict or None on failure."""
        if not self.is_available():
            return None
        try:
            # The NGAP top-level type name is commonly 'NGAP-PDU'
            decoded = self.decoder.decode('NGAP-PDU', payload_bytes)
            return decoded  # asn1tools returns nested dicts/lists
        except Exception:
            return None

    def extract_basic_fields(self, decoded: Dict[str, Any]) -> Dict[str, Any]:
        """Extract commonly used fields from a decoded NGAP PDU.

        Returns keys: procedure_code, message_type (0: initiating, 1: successful, 2: unsuccessful),
        amf_ue_ngap_id, ran_ue_ngap_id, cause (category, value).
        """
        result: Dict[str, Any] = {
            'procedure_code': None,
            'message_type': None,
            'amf_ue_ngap_id': None,
            'ran_ue_ngap_id': None,
            'cause': None
        }

        try:
            # NGAP-PDU ::= CHOICE { initiatingMessage, successfulOutcome, unsuccessfulOutcome }
            if 'initiatingMessage' in decoded:
                msg = decoded['initiatingMessage']
                result['message_type'] = 0
            elif 'successfulOutcome' in decoded:
                msg = decoded['successfulOutcome']
                result['message_type'] = 1
            elif 'unsuccessfulOutcome' in decoded:
                msg = decoded['unsuccessfulOutcome']
                result['message_type'] = 2
            else:
                return result

            # Common fields
            result['procedure_code'] = msg.get('procedureCode')

            # Information Elements are under 'value' (ProtocolIE-Container)
            ie_container = msg.get('value', {}).get('protocolIEs', [])
            for ie in ie_container:
                ie_id = ie.get('id')
                ie_val = ie.get('value', {})
                # AMF-UE-NGAP-ID
                if ie_id in (10, 'id-AMF-UE-NGAP-ID') and 'AMF-UE-NGAP-ID' in ie_val:
                    result['amf_ue_ngap_id'] = ie_val['AMF-UE-NGAP-ID']
                # RAN-UE-NGAP-ID
                if ie_id in (85, 'id-RAN-UE-NGAP-ID') and 'RAN-UE-NGAP-ID' in ie_val:
                    result['ran_ue_ngap_id'] = ie_val['RAN-UE-NGAP-ID']
                # Cause IE
                if ie_id in (15, 'id-Cause') and 'Cause' in ie_val:
                    cause = ie_val['Cause']
                    # Cause is CHOICE { radioNetwork, transport, nas, protocol, misc }
                    for cat in ('radioNetwork', 'transport', 'nas', 'protocol', 'misc'):
                        if cat in cause:
                            result['cause'] = {
                                'category': cat,
                                'value': cause[cat]
                            }
                            break

        except Exception:
            # Be forgiving; return what we managed to extract
            return result

        return result


