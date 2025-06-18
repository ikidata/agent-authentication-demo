from pydantic import BaseModel, field_validator
from mlflow.entities import SpanType
from typing import List, Generator, Any, Optional, Dict
import mlflow
import re 
from requests.exceptions import RequestException
import requests
from uuid import uuid4
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import ModelServingUserCredentials


def get_workspace_client(auth_type: str) -> object:
    """
    Returns an instance of WorkspaceClient based on the authentication type.

    Args:
        auth_type (str): The type of authentication to use. Must be 'system' or 'user'.

    Returns:
        WorkspaceClient: An instance of WorkspaceClient configured with the specified authentication type.
    """
    if auth_type == 'system':
        w = WorkspaceClient()
    elif auth_type == 'user':
        w = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
    else:
        raise ValueError("auth_type must be 'system' or 'user'")
    return w


class ToolException(Exception):
    """Custom exception for tool-related errors."""
    pass

# --- Pydantic models to validate the API response ---
class WeatherDesc(BaseModel):
    value: str

class ToolFunctionCall(BaseModel):
    name: str
    arguments: str 

    def get_parsed_arguments(self) -> Dict[str, Any]:
        return json.loads(self.arguments)
    
class CurrentCondition(BaseModel):
    temp_C: str
    weatherDesc: List[WeatherDesc]

class WeatherResponse(BaseModel):
    current_condition: List[CurrentCondition]

class ToolCall(BaseModel):
    id: str
    type: str
    function: ToolFunctionCall

class ToolCallsOutput(BaseModel):
    tool_calls: List[ToolCall]

def create_tool_calls_output(results: object) -> dict:
    tool_calls = []

    for tool_call in results.tool_calls:
        function_arguments = tool_call.function.arguments
        if isinstance(function_arguments, dict):
            function_arguments = json.dumps(function_arguments)

        tool_call_model = ToolCall(
            id=tool_call.id,
            type=tool_call.type,
            function=ToolFunctionCall(
                name=tool_call.function.name,
                arguments=function_arguments 
            )
        )
        tool_calls.append(tool_call_model)

    # Convert to native dict so it can be injected into ChatAgentMessage
    return ToolCallsOutput(tool_calls=tool_calls).model_dump()["tool_calls"]



@mlflow.trace(name="get_weather", span_type=SpanType.TOOL)
def get_weather(city: str) -> str:
    """
    Fetches the current weather for a given city using the wttr.in API.

    Args:
        city (str): Name of the city.

    Returns:
        str: Weather description and temperature.
    """
    url = f"https://wttr.in/{city}?format=j1"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        raw_data = response.json()

        try:
            weather_data = WeatherResponse.model_validate(raw_data)
        except ValidationError as e:
            raise ToolException(f"Invalid weather data format: {e}")

        current = weather_data.current_condition[0]
        description = current.weatherDesc[0].value
        temperature = current.temp_C

        return f"Weather in {city} is {description}, {temperature}°C."

    except RequestException as e:
        raise ToolException(
            f"Error retrieving weather for '{city}': {str(e)}"
        )


def prepare_messages_for_llm(messages: list[ChatAgentMessage]) -> list[dict[str, Any]]:
    """Filter out ChatAgentMessage fields that are not compatible with LLM message formats"""
    compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
    return [
        {k: v for k, v in m.model_dump_compat(exclude_none=True).items() if k in compatible_keys} for m in messages
    ]


@mlflow.trace(name="Call LLM model", span_type='LLM')
def call_chat_model(openai_client: any, model_name: str, messages: list, temperature: float = 0.3, max_tokens: int = 1000, **kwargs):
    """
    Calls the chat model and returns the response text or tool calls.

    Parameters:
        message (list): Message to send to the chat model.
        temperature (float, optional): Controls response randomness. Defaults to 0.3.
        max_tokens (int, optional): Maximum tokens for the response. Defaults to 750.
        **kwargs: Additional parameters for the chat model.

    Returns:
        str: Response from the chat model.
    """
    # Prepare arguments for the chat model
    chat_args = {
        "model": model_name,  
        "messages": prepare_messages_for_llm(messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs,  # Update with additional arguments
    }
    try:
        chat_completion = openai_client.chat.completions.create(**chat_args)
        return chat_completion  
    except Exception as e:
        error_message = f"Model endpoint calling error: {e}"
        print(error_message)
        raise RuntimeError(error_message)