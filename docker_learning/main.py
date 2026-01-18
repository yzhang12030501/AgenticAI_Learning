from agents import Agent, Runner

financial_agent = Agent(
    name="Financial Planning Agent",
    instructions=(
        "You are a professional financial planner. "
        "Ask clear questions and give responsible advice."
    ),
    model="gpt-4.1-mini",
)

customer_agent = Agent(
    name="Customer Agent",
    instructions=(
        "You are a customer ONLY. "
        "Answer questions asked by the planner. "
        "Do not give advice."
    ),
    model="gpt-4.1-mini",
)


def run_conversation(turns: int = 2):
    conversation_history = []

    # Initial customer message
    conversation_history.append(
        {
            "role": "user",
            "content": (
                "I earn $80,000 per year, have $20,000 in savings, "
                "and want to buy a house in five years."
            ),
        }
    )

    for _ in range(turns):
        # Financial planner responds using full history
        planner_result = Runner.run_sync(financial_agent, conversation_history)
        planner_reply = planner_result.final_output

        print("\n[Financial Planner]")
        print(planner_reply)

        conversation_history.append({"role": "assistant", "content": planner_reply})

        # Customer answers planner questions
        customer_prompt = (
            "The financial planner asked you the following:\n\n"
            f"{planner_reply}\n\n"
            "Answer ONLY the questions asked."
        )

        customer_result = Runner.run_sync(customer_agent, customer_prompt)
        customer_reply = customer_result.final_output

        print("\n[Customer]")
        print(customer_reply)

        conversation_history.append({"role": "user", "content": customer_reply})

    return conversation_history


if __name__ == "__main__":
    run_conversation()
