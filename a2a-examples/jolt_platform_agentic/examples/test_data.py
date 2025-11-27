"""
Example test data for JOLT platform
Contains sample input/output pairs for testing
"""

# Example 1: Simple field mapping and restructuring
EXAMPLE_1 = {
    "name": "Simple Field Mapping",
    "description": "Basic user profile transformation",
    "input": {
        "user": {
            "firstName": "John",
            "lastName": "Doe",
            "email": "john.doe@example.com"
        },
        "timestamp": "2024-01-01T12:00:00Z"
    },
    "expected_output": {
        "fullName": "John Doe",
        "contact": {
            "email": "john.doe@example.com"
        },
        "eventtime": "2024-01-01T12:00:00Z"
    }
}

# Example 2: Nested structure transformation
EXAMPLE_2 = {
    "name": "Nested Structure",
    "description": "Complex order transformation with nested items",
    "input": {
        "order": {
            "id": "ORD-123",
            "customer": {
                "name": "Alice Smith",
                "email": "alice@example.com",
                "phone": "+1-555-0123"
            },
            "items": [
                {"product": "Laptop", "price": 999.99, "quantity": 1},
                {"product": "Mouse", "price": 29.99, "quantity": 2}
            ],
            "total": 1059.97
        }
    },
    "expected_output": {
        "orderId": "ORD-123",
        "customerInfo": {
            "customerName": "Alice Smith",
            "contactEmail": "alice@example.com",
            "contactPhone": "+1-555-0123"
        },
        "orderItems": [
            {"productName": "Laptop", "amount": 999.99, "qty": 1},
            {"productName": "Mouse", "amount": 29.99, "qty": 2}
        ],
        "orderTotal": 1059.97
    }
}

# Example 3: Array processing
EXAMPLE_3 = {
    "name": "Array Processing",
    "description": "Transform array of events",
    "input": {
        "events": [
            {
                "type": "login",
                "user": "john@example.com",
                "time": "2024-01-01T10:00:00Z"
            },
            {
                "type": "purchase",
                "user": "john@example.com",
                "time": "2024-01-01T10:30:00Z",
                "amount": 99.99
            }
        ]
    },
    "expected_output": {
        "userEvents": [
            {
                "eventType": "login",
                "userId": "john@example.com",
                "timestamp": "2024-01-01T10:00:00Z"
            },
            {
                "eventType": "purchase",
                "userId": "john@example.com",
                "timestamp": "2024-01-01T10:30:00Z",
                "purchaseAmount": 99.99
            }
        ]
    }
}

# Example 4: Flattening nested structure
EXAMPLE_4 = {
    "name": "Flatten Structure",
    "description": "Flatten deeply nested JSON",
    "input": {
        "company": {
            "name": "TechCorp",
            "location": {
                "address": {
                    "street": "123 Main St",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip": "94105"
                }
            }
        }
    },
    "expected_output": {
        "companyName": "TechCorp",
        "street": "123 Main St",
        "city": "San Francisco",
        "state": "CA",
        "zipCode": "94105"
    }
}

# Example 5: Combining fields
EXAMPLE_5 = {
    "name": "Combine Fields",
    "description": "Combine multiple fields into one",
    "input": {
        "person": {
            "title": "Dr.",
            "firstName": "Jane",
            "middleName": "Marie",
            "lastName": "Johnson",
            "suffix": "PhD"
        }
    },
    "expected_output": {
        "fullName": "Dr. Jane Marie Johnson PhD"
    }
}

ALL_EXAMPLES = [
    EXAMPLE_1,
    EXAMPLE_2,
    EXAMPLE_3,
    EXAMPLE_4,
    EXAMPLE_5
]


def get_example(index: int = 0):
    """Get an example by index (0-4)."""
    if 0 <= index < len(ALL_EXAMPLES):
        return ALL_EXAMPLES[index]
    return EXAMPLE_1


def list_examples():
    """List all available examples."""
    print("\n📚 Available Examples:\n")
    for i, example in enumerate(ALL_EXAMPLES):
        print(f"{i + 1}. {example['name']}")
        print(f"   {example['description']}\n")
