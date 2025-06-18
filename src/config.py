def get_raw_configs() -> dict:
    '''
    Remember to update genie_space_id!!!

    This was the easiest & robust way to keep config maintenance simple using MLflow agent deployment, for now. 
    '''
    return {
        "genie_space_id": 'UPDATE GENIE SPACE ID HERE',
        "model_name": "databricks-claude-3-7-sonnet",
        "system_prompt": """
                        Your job is the help to demonstrate how easy agent development is on Databricks. And trigger Genie space when customer requires so.
                        """,
        "tools": {
            "get_weather": {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a chosen city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City to be used"
                            }
                        },
                        "required": ["city"],
                        "returns": {
                            "type": "string",
                            "description": "Weather information for the specified city"
                        }
                    }
                }
            },
            "run_genie": {
                "type": "function",
                "function": {
                    "name": "run_genie",
                    "description": "If this isn't updated, this tool triggers Genie space",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Write an optimized prompt to fetch the requested information from Genie Space"
                            }
                        },
                        "required": ["prompt"],
                        "returns": {
                            "type": "string",
                            "description": "Genie space output"
                        }
                    }
                }
            }
        }
    }