from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from utils.utility import function_to_json_schema
import os
import json
import requests

from agents import (
    Agent,
    set_default_openai_api,
    function_tool,
    Runner,
    SQLiteSession,
)

load_dotenv()

set_default_openai_api("chat_completions")
set_default_openai_api(os.getenv("OPENAI_API_KEY"))


# Initialize OpenAI client
openai_client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url=os.environ.get("DEEPSEEK_BASE_URL"),
)

# 1. Create the FastAPI app
# This creates an API server object.
# Think of it as an empty container that will hold all your endpoints.
app = FastAPI()


# 2. Create a URL + HTTP method together
@app.get("/hello")
def say_hello():
    return {"Hello": "World"}


# 3. Example with Parameters (More Realistic)
@app.get("/customers/{customer_id}")
def get_customer(customer_id: int):
    return {"customer_id": customer_id}


class Item(BaseModel):
    name: str
    price: float


@app.post("/items")
def create_item(item: Item):
    return {"message": "Item created", "item": item}


@app.post("/sum")
def sum_numbers(numbers: list[float]):
    try:
        result = sum(numbers)
        return {"sum": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ======= For agent call ======
# Define input schema
class AgentRequest(BaseModel):
    question: str


# Define output schema
class AgentResponse(BaseModel):
    answer: str


# Agent endpoint
# adding response_model to validate the response data against the AgentResponse schema
@app.post("/agent", response_model=AgentResponse)
def agent(request: AgentRequest):
    """
    This function IS the agent.
    FastAPI exposes it.
    OpenAI SDK provides reasoning.
    """
    try:
        response = openai_client.chat.completions.create(
            model="deepseek-chat",  # non-thinking mode of deepseek
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": request.question},
            ],
            max_tokens=50,
            stream=False,
        )

        answer = response.choices[0].message.content
        return AgentResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ======= For agent with tool call ======
def calculate_loan_payment(principal: float, rate: float, years: int) -> float:
    """
    Simple mortgage-style payment calculation.
    """
    monthly_rate = rate / 12
    months = years * 12
    payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** -months)
    return round(payment, 2)


@app.post("/agent_tools", response_model=AgentResponse)
def agent_tools(request: AgentRequest):
    """
    This function IS the agent.
    FastAPI exposes it.
    OpenAI SDK provides reasoning.
    """

    tool_registry = {"calculate_loan_payment": calculate_loan_payment}

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": request.question},
    ]

    try:
        # 1. Send user question + tool definitions to the model
        response = openai_client.chat.completions.create(
            model="deepseek-chat",  # non-thinking mode of deepseek
            messages=messages,
            tools=[function_to_json_schema(calculate_loan_payment)],
            stream=False,
        )

        #  2. Check if the model wants to call a tool
        ## If not, return the model's response
        if not response.choices[0].message.tool_calls:
            return AgentResponse(answer=response.choices[0].message.content)

        ## If so, append the message, and then extract the tool name and inputs
        messages.append(response.choices[0].message)
        tool_name = response.choices[0].message.tool_calls[0].function.name
        tool_inputs = response.choices[0].message.tool_calls[0].function.arguments

        # Call the tool with the extracted arguments
        function_call = tool_registry.get(tool_name)
        if function_call:
            tool_outputs = function_call(**json.loads(tool_inputs))
        else:
            tool_outputs = f"Tool {tool_name} not found."

        # Append the tool output to the message history
        messages.append(
            {
                "role": "tool",
                "tool_call_id": response.choices[0].message.tool_calls[0].id,
                "content": str(tool_outputs),
            }
        )

        # 3. Send the updated message history back to the model
        response = openai_client.chat.completions.create(
            model="deepseek-chat",  # non-thinking mode of deepseek
            messages=messages,
            tools=[function_to_json_schema(calculate_loan_payment)],
            stream=False,
        )

        answer = response.choices[0].message.content
        return AgentResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# Agent use tool
# ========================================


# ======= Create a FastAPI endpoint ======
class PDResponse(BaseModel):
    customer_id: int
    pd: float


## Define the function tool
@function_tool
def get_probability_of_default(customer_id: int) -> float:
    """Get the probability of default for a customer given their ID."""
    response = requests.get(
        "http://127.0.0.1:8000/pd", params={"customer_id": customer_id}, timeout=5
    )
    data = response.json()
    return data["pd"]


## Define the FastAPI endpoint
@app.get("/pd", response_model=PDResponse)
def get_pd(customer_id: int):
    """
    This endpoint returns the probability of default (PD) for a given customer ID.
    """
    # For demonstration purposes, we return a dummy PD value.
    # --- Mock risk logic (replace with real model later) ---
    pd_value = 0.02 + (customer_id % 10) * 0.001
    return {"customer_id": customer_id, "pd": round(pd_value, 4)}


# ======= Create an Agent That Can Use This Tool =======
# Define the agent
credit_agent = Agent(
    name="Credit Risk Agent",
    instructions="An agent that assesses credit risk using external API calls.",
    model="gpt-4o-mini",
    tools=[get_probability_of_default],  # add the function tool
)


# Expose the Agent via FastAPI
@app.post("/credit_risk_assessment", response_model=AgentResponse)
def credit_risk_assessment(request: AgentRequest):
    """
    This endpoint uses the Credit Risk Agent to assess credit risk.
    """
    try:
        response = Runner.run_sync(credit_agent, request.question)
        answer = response.final_output
        return AgentResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/credit_risk_assessment_async", response_model=AgentResponse)
async def credit_risk_assessment_async(request: AgentRequest):
    """
    This endpoint uses the Credit Risk Agent to assess credit risk.
    """
    try:
        response = await Runner.run(credit_agent, request.question)
        answer = response.final_output
        return AgentResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/credit_risk_assessment_raw", response_model=AgentResponse)
def credit_risk_assessment_raw(request: str):
    """
    This endpoint uses the Credit Risk Agent to assess credit risk.
    """
    try:
        response = Runner.run_sync(credit_agent, request)
        answer = response.final_output
        return AgentResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
