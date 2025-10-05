"""StructuredOpenAI: LlamaIndex OpenAI LLM with enforced JSON schema.

This module provides a subclass of LlamaIndex's OpenAI LLM that enforces
structured JSON responses using OpenAI's response_format with json_schema.
"""

from typing import Any, Dict
from llama_index.llms.openai import OpenAI


# JSON Schema for pain agent responses
# Using non-strict mode to allow conditional fields based on response_type
PAIN_AGENT_JSON_SCHEMA = {
    "name": "pain_assessment_response",
    "description": "Structured response format for pain assessment - either a question with options OR a final conclusion text",
    "strict": False,  # Must be False to support conditional required fields
    "schema": {
        "type": "object",
        "properties": {
            "response_type": {
                "type": "string",
                "enum": ["question", "conclusion"],
                "description": "Whether this is a question during assessment or the final conclusion"
            },
            "question": {
                "type": "string",
                "description": "The pain assessment question to ask (required when response_type=question)"
            },
            "type": {
                "type": "string",
                "enum": ["single_choice", "multiple_choice"],
                "description": "Selection type (required when response_type=question)"
            },
            "options": {
                "type": "array",
                "description": "List of answer options (required when response_type=question)",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique identifier for this option (snake_case, no spaces)"
                        },
                        "label": {
                            "type": "string",
                            "description": "The text label shown to the user in French"
                        }
                    },
                    "required": ["id", "label"],
                    "additionalProperties": False
                },
                "minItems": 2
            },
            "conclusion": {
                "type": "string",
                "description": "Final conclusion text with recommendations (required when response_type=conclusion)"
            }
        },
        "required": ["response_type"],
        "additionalProperties": False
    }
}


class StructuredOpenAI(OpenAI):
    """OpenAI LLM that enforces JSON schema responses using response_format.

    This class extends LlamaIndex's OpenAI to add support for OpenAI's
    response_format={"type": "json_schema"} feature, which enforces that
    the model's output strictly adheres to a predefined JSON schema.

    The schema ensures pain agent responses always contain:
    - question: The assessment question
    - type: "single_choice" or "multiple_choice"
    - options: Array of {id, label} objects

    Streaming behavior is preserved.
    """

    def __init__(self, *args, json_schema: Dict[str, Any] = None, **kwargs):
        """Initialize StructuredOpenAI with optional JSON schema.

        Args:
            json_schema: OpenAI JSON schema dict. If not provided, uses PAIN_AGENT_JSON_SCHEMA.
            *args, **kwargs: Passed to parent OpenAI class
        """
        super().__init__(*args, **kwargs)
        self._json_schema = json_schema or PAIN_AGENT_JSON_SCHEMA

    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """Override to inject response_format with JSON schema.

        This method is called by LlamaIndex before every OpenAI API call.
        We inject the response_format parameter to enforce JSON schema.
        """
        # Get base kwargs from parent
        model_kwargs = super()._get_model_kwargs(**kwargs)

        # Inject response_format with json_schema
        # This tells OpenAI to enforce the schema and return valid JSON
        model_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": self._json_schema
        }

        return model_kwargs
