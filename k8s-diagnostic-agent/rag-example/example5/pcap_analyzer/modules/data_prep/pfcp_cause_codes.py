"""
PFCP Cause Code Definitions and Analysis

This module provides comprehensive support for all PFCP (Packet Forwarding Control Protocol)
cause codes as defined in 3GPP specifications. It includes cause code meanings,
categorization, and analysis functions for PCAP parsing and prediction.
"""

from typing import Dict, List, Tuple, Optional
from enum import IntEnum


class PFCPCauseCode(IntEnum):
    """PFCP Cause Code enumeration with all supported values."""
    
    # Reserved - shall not be sent
    RESERVED = 0
    
    # Acceptance causes (1-63)
    REQUEST_ACCEPTED = 1
    REQUEST_ACCEPTED_PARTIALLY = 2
    
    # Rejection causes (64-77)
    REQUEST_REJECTED_REASON_NOT_SPECIFIED = 64
    SESSION_CONTEXT_NOT_FOUND = 65
    MANDATORY_IE_MISSING = 66
    CONDITIONAL_IE_MISSING = 67
    INVALID_LENGTH = 68
    MANDATORY_IE_INCORRECT = 69
    INVALID_FORWARDING_POLICY = 70
    INVALID_F_TEID_ALLOCATION_OPTION = 71
    NO_ESTABLISHED_PFCP_ASSOCIATION = 72
    RULE_CREATION_MODIFICATION_FAILURE = 73
    PFCP_ENTITY_IN_CONGESTION = 74
    NO_RESOURCES_AVAILABLE = 75
    SERVICE_NOT_SUPPORTED = 76
    SYSTEM_FAILURE = 77


class PFCPCauseCategory:
    """Categories for PFCP cause codes."""
    
    ACCEPTANCE = "acceptance"
    REJECTION = "rejection"
    RESERVED = "reserved"
    SPARE = "spare"


class PFCPCauseAnalyzer:
    """Analyzer for PFCP cause codes with comprehensive support."""
    
    def __init__(self):
        """Initialize the PFCP cause code analyzer."""
        self._cause_definitions = self._initialize_cause_definitions()
        self._cause_categories = self._initialize_cause_categories()
    
    def _initialize_cause_definitions(self) -> Dict[int, Dict[str, str]]:
        """Initialize comprehensive cause code definitions."""
        return {
            0: {
                "name": "Reserved",
                "description": "Shall not be sent and if received the Cause shall be treated as an invalid IE",
                "category": PFCPCauseCategory.RESERVED,
                "acceptance_in_response": False
            },
            1: {
                "name": "Request accepted (success)",
                "description": "Request accepted (success) is returned when the PFCP entity has accepted a request.",
                "category": PFCPCauseCategory.ACCEPTANCE,
                "acceptance_in_response": True
            },
            2: {
                "name": "Request accepted (success) - partial",
                "description": "Request accepted with partial success.",
                "category": PFCPCauseCategory.ACCEPTANCE,
                "acceptance_in_response": True
            },
            # Spare values 3-63
            3: {"name": "Spare", "description": "Spare value for future use in acceptance responses", "category": PFCPCauseCategory.SPARE, "acceptance_in_response": True},
            4: {"name": "Spare", "description": "Spare value for future use in acceptance responses", "category": PFCPCauseCategory.SPARE, "acceptance_in_response": True},
            5: {"name": "Spare", "description": "Spare value for future use in acceptance responses", "category": PFCPCauseCategory.SPARE, "acceptance_in_response": True},
            6: {"name": "Spare", "description": "Spare value for future use in acceptance responses", "category": PFCPCauseCategory.SPARE, "acceptance_in_response": True},
            7: {"name": "Spare", "description": "Spare value for future use in acceptance responses", "category": PFCPCauseCategory.SPARE, "acceptance_in_response": True},
            8: {"name": "Spare", "description": "Spare value for future use in acceptance responses", "category": PFCPCauseCategory.SPARE, "acceptance_in_response": True},
            9: {"name": "Spare", "description": "Spare value for future use in acceptance responses", "category": PFCPCauseCategory.SPARE, "acceptance_in_response": True},
            10: {"name": "Spare", "description": "Spare value for future use in acceptance responses", "category": PFCPCauseCategory.SPARE, "acceptance_in_response": True},
            # Add more spare values as needed...
            63: {"name": "Spare", "description": "Spare value for future use in acceptance responses", "category": PFCPCauseCategory.SPARE, "acceptance_in_response": True},
            
            # Rejection causes
            64: {
                "name": "Request rejected (reason not specified)",
                "description": "This cause shall be returned to report an unspecified rejection cause",
                "category": PFCPCauseCategory.REJECTION,
                "acceptance_in_response": False
            },
            65: {
                "name": "Session context not found",
                "description": "This cause shall be returned, if the F-SEID included in a PFCP Session Modification/Deletion Request message is unknown.",
                "category": PFCPCauseCategory.REJECTION,
                "acceptance_in_response": False
            },
            66: {
                "name": "Mandatory IE missing",
                "description": "This cause shall be returned when the PFCP entity detects that a mandatory IE is missing in a request message",
                "category": PFCPCauseCategory.REJECTION,
                "acceptance_in_response": False
            },
            67: {
                "name": "Conditional IE missing",
                "description": "This cause shall be returned when the PFCP entity detects that a Conditional IE is missing in a request message.",
                "category": PFCPCauseCategory.REJECTION,
                "acceptance_in_response": False
            },
            68: {
                "name": "Invalid length",
                "description": "This cause shall be returned when the PFCP entity detects that an IE with an invalid length in a request message",
                "category": PFCPCauseCategory.REJECTION,
                "acceptance_in_response": False
            },
            69: {
                "name": "Mandatory IE incorrect",
                "description": "This cause shall be returned when the PFCP entity detects that a Mandatory IE is incorrect in a request message, e.g. the Mandatory IE is malformated or it carries an invalid or unexpected value.",
                "category": PFCPCauseCategory.REJECTION,
                "acceptance_in_response": False
            },
            70: {
                "name": "Invalid Forwarding Policy",
                "description": "This cause shall be used by the UP function in the PFCP Session Establishment Response or PFCP Session Modification Response message if the CP function attempted to provision a FAR with a Forwarding Policy Identifier for which no Forwarding Policy is locally configured in the UP function.",
                "category": PFCPCauseCategory.REJECTION,
                "acceptance_in_response": False
            },
            71: {
                "name": "Invalid F-TEID allocation option",
                "description": "This cause shall be used by the UP function in the PFCP Session Establishment Response or PFCP Session Modification Response message if the CP function attempted to provision a PDR with a F-TEID allocation option which is incompatible with the F-TEID allocation option used for already created PDRs (by the same or a different CP function).",
                "category": PFCPCauseCategory.REJECTION,
                "acceptance_in_response": False
            },
            72: {
                "name": "No established PFCP Association",
                "description": "This cause shall be used by the CP function or the UP function if they receive an PFCP Session related message from a peer with which there is no established PFCP Association.",
                "category": PFCPCauseCategory.REJECTION,
                "acceptance_in_response": False
            },
            73: {
                "name": "Rule creation/modification Failure",
                "description": "This cause shall be used by the UP function if a received Rule failed to be stored and be applied in the UP function.",
                "category": PFCPCauseCategory.REJECTION,
                "acceptance_in_response": False
            },
            74: {
                "name": "PFCP entity in congestion",
                "description": "This cause shall be returned when a PFPC entity has detected node level congestion and performs overload control, which does not allow the request to be processed.",
                "category": PFCPCauseCategory.REJECTION,
                "acceptance_in_response": False
            },
            75: {
                "name": "No resources available",
                "description": "This cause shall be returned to indicate a temporary unavailability of resources to process the received request.",
                "category": PFCPCauseCategory.REJECTION,
                "acceptance_in_response": False
            },
            76: {
                "name": "Service not supported",
                "description": "This cause shall be returned when a PFCP entity receives a message requesting a feature or service that is not supported.",
                "category": PFCPCauseCategory.REJECTION,
                "acceptance_in_response": False
            },
            77: {
                "name": "System failure",
                "description": "This cause shall be returned to indicate a system error condition.",
                "category": PFCPCauseCategory.REJECTION,
                "acceptance_in_response": False
            }
        }
    
    def _initialize_cause_categories(self) -> Dict[str, List[int]]:
        """Initialize cause code categories."""
        categories = {
            PFCPCauseCategory.ACCEPTANCE: [],
            PFCPCauseCategory.REJECTION: [],
            PFCPCauseCategory.RESERVED: [],
            PFCPCauseCategory.SPARE: []
        }
        
        for cause_code, definition in self._cause_definitions.items():
            category = definition["category"]
            categories[category].append(cause_code)
        
        return categories
    
    def get_cause_definition(self, cause_code: int) -> Optional[Dict[str, str]]:
        """Get the definition for a specific cause code.
        
        Args:
            cause_code: The PFCP cause code value
            
        Returns:
            Dictionary containing cause code definition or None if not found
        """
        return self._cause_definitions.get(cause_code)
    
    def get_cause_name(self, cause_code: int) -> str:
        """Get the human-readable name for a cause code.
        
        Args:
            cause_code: The PFCP cause code value
            
        Returns:
            Human-readable name or "Unknown cause code {code}"
        """
        definition = self.get_cause_definition(cause_code)
        if definition:
            return definition["name"]
        return f"Unknown cause code {cause_code}"
    
    def get_cause_description(self, cause_code: int) -> str:
        """Get the detailed description for a cause code.
        
        Args:
            cause_code: The PFCP cause code value
            
        Returns:
            Detailed description or "Unknown cause code {code}"
        """
        definition = self.get_cause_definition(cause_code)
        if definition:
            return definition["description"]
        return f"Unknown cause code {cause_code}"
    
    def get_cause_category(self, cause_code: int) -> str:
        """Get the category for a cause code.
        
        Args:
            cause_code: The PFCP cause code value
            
        Returns:
            Category string or "unknown"
        """
        definition = self.get_cause_definition(cause_code)
        if definition:
            return definition["category"]
        return "unknown"
    
    def is_acceptance_cause(self, cause_code: int) -> bool:
        """Check if a cause code indicates acceptance.
        
        Args:
            cause_code: The PFCP cause code value
            
        Returns:
            True if acceptance cause, False otherwise
        """
        definition = self.get_cause_definition(cause_code)
        if definition:
            return definition["acceptance_in_response"]
        return False
    
    def is_rejection_cause(self, cause_code: int) -> bool:
        """Check if a cause code indicates rejection.
        
        Args:
            cause_code: The PFCP cause code value
            
        Returns:
            True if rejection cause, False otherwise
        """
        return self.get_cause_category(cause_code) == PFCPCauseCategory.REJECTION
    
    def get_causes_by_category(self, category: str) -> List[int]:
        """Get all cause codes in a specific category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of cause codes in the category
        """
        return self._cause_categories.get(category, [])
    
    def get_all_acceptance_causes(self) -> List[int]:
        """Get all acceptance cause codes.
        
        Returns:
            List of acceptance cause codes
        """
        return self.get_causes_by_category(PFCPCauseCategory.ACCEPTANCE)
    
    def get_all_rejection_causes(self) -> List[int]:
        """Get all rejection cause codes.
        
        Returns:
            List of rejection cause codes
        """
        return self.get_causes_by_category(PFCPCauseCategory.REJECTION)
    
    def analyze_cause_codes(self, cause_codes: List[int]) -> Dict[str, any]:
        """Analyze a list of cause codes and provide insights.
        
        Args:
            cause_codes: List of PFCP cause codes
            
        Returns:
            Dictionary with analysis results
        """
        if not cause_codes:
            return {"total": 0, "analysis": "No cause codes found"}
        
        analysis = {
            "total": len(cause_codes),
            "acceptance_count": 0,
            "rejection_count": 0,
            "reserved_count": 0,
            "spare_count": 0,
            "unknown_count": 0,
            "categories": {},
            "severity": "info",
            "insights": []
        }
        
        for cause_code in cause_codes:
            category = self.get_cause_category(cause_code)
            analysis["categories"][category] = analysis["categories"].get(category, 0) + 1
            
            if category == PFCPCauseCategory.ACCEPTANCE:
                analysis["acceptance_count"] += 1
            elif category == PFCPCauseCategory.REJECTION:
                analysis["rejection_count"] += 1
                analysis["severity"] = "warning"
            elif category == PFCPCauseCategory.RESERVED:
                analysis["reserved_count"] += 1
                analysis["severity"] = "error"
            elif category == PFCPCauseCategory.SPARE:
                analysis["spare_count"] += 1
            else:
                analysis["unknown_count"] += 1
        
        # Generate insights
        if analysis["rejection_count"] > 0:
            analysis["insights"].append(f"Found {analysis['rejection_count']} rejection cause(s) indicating potential issues")
        
        if analysis["reserved_count"] > 0:
            analysis["insights"].append(f"Found {analysis['reserved_count']} reserved cause code(s) - this indicates protocol violations")
        
        if analysis["acceptance_count"] > 0 and analysis["rejection_count"] == 0:
            analysis["insights"].append("All operations completed successfully")
        
        return analysis
    
    def get_cause_summary_text(self, cause_codes: List[int]) -> str:
        """Generate a human-readable summary of cause codes.
        
        Args:
            cause_codes: List of PFCP cause codes
            
        Returns:
            Formatted summary string
        """
        if not cause_codes:
            return "No cause codes found"
        
        # Group by category
        categories = {}
        for cause_code in cause_codes:
            category = self.get_cause_category(cause_code)
            if category not in categories:
                categories[category] = []
            categories[category].append(cause_code)
        
        # Build summary
        parts = []
        for category, codes in categories.items():
            if category == PFCPCauseCategory.ACCEPTANCE:
                parts.append(f"Acceptance: {', '.join([f'{c}:{self.get_cause_name(c)}' for c in sorted(codes)])}")
            elif category == PFCPCauseCategory.REJECTION:
                parts.append(f"Rejection: {', '.join([f'{c}:{self.get_cause_name(c)}' for c in sorted(codes)])}")
            elif category == PFCPCauseCategory.RESERVED:
                parts.append(f"Reserved: {', '.join([f'{c}:{self.get_cause_name(c)}' for c in sorted(codes)])}")
            else:
                parts.append(f"Other: {', '.join([f'{c}:{self.get_cause_name(c)}' for c in sorted(codes)])}")
        
        return "; ".join(parts)
    
    def get_all_cause_codes(self) -> List[int]:
        """Get all supported cause codes.
        
        Returns:
            List of all cause code values
        """
        return list(self._cause_definitions.keys())
    
    def get_cause_code_mapping(self) -> Dict[int, str]:
        """Get a simple mapping of cause codes to names.
        
        Returns:
            Dictionary mapping cause codes to names
        """
        return {code: definition["name"] for code, definition in self._cause_definitions.items()}


# Global instance for easy access
pfcp_cause_analyzer = PFCPCauseAnalyzer()


def get_pfcp_cause_analyzer() -> PFCPCauseAnalyzer:
    """Get the global PFCP cause analyzer instance.
    
    Returns:
        PFCPCauseAnalyzer instance
    """
    return pfcp_cause_analyzer
