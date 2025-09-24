import os
import threading
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistoryTruncationReducer
from semantic_kernel.functions import KernelFunctionFromPrompt

REVIEWER_NAME = "Reviewer"
WRITER_NAME = "Writer"

# Background event loop setup (single loop reused)
_loop = asyncio.new_event_loop()
_thread = threading.Thread(target=_loop.run_forever, daemon=True)
_thread.start()

def run_async(coro):
    """Run an async coroutine on the background loop synchronously."""
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()

def create_kernel() -> Kernel:
    load_dotenv()
    

    client = AsyncOpenAI(
        api_key=os.environ.get("GITHUB_TOKEN"),
        base_url="https://models.inference.ai.azure.com/",
    )
    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            ai_model_id="gpt-4.1",
            async_client=client,
        )
    )
    return kernel

def build_chat():
    kernel = create_kernel()

    agent_reviewer = ChatCompletionAgent(
        kernel=kernel,
        name=REVIEWER_NAME,
        instructions="""
Your responsibility is to review and identify how to improve user provided content.
If the user has provided input or direction for content already provided, specify how to address this input.
Never directly perform the correction or provide an example.
Once the content has been updated in a subsequent response, review it again until it is satisfactory.

RULES:
- Only identify suggestions that are specific and actionable.
- Verify previous suggestions have been addressed.
- Never repeat previous suggestions.
""",
    )

    agent_writer = ChatCompletionAgent(
        kernel=kernel,
        name=WRITER_NAME,
        instructions="""
Your sole responsibility is to rewrite content according to review suggestions.
- Always apply all review directions.
- Always revise the content in its entirety without explanation.
- Never address the user.
""",
    )

    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
Examine the provided RESPONSE and choose the next participant.
State only the name of the chosen participant without explanation.
Never choose the participant named in the RESPONSE.

Choose only from these participants:
- {REVIEWER_NAME}
- {WRITER_NAME}

Rules:
- If RESPONSE is user input, it is {REVIEWER_NAME}'s turn.
- If RESPONSE is by {REVIEWER_NAME}, it is {WRITER_NAME}'s turn.
- If RESPONSE is by {WRITER_NAME}, it is {REVIEWER_NAME}'s turn.

RESPONSE:
{{{{$lastmessage}}}}
""",
    )

    termination_keyword = "yes"
    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt=f"""
Examine the RESPONSE and determine whether the content has been deemed satisfactory.
If the content is satisfactory, respond with a single word without explanation: {termination_keyword}.
If specific suggestions are being provided, it is not satisfactory.
If no correction is suggested, it is satisfactory.

RESPONSE:
{{{{$lastmessage}}}}
""",
    )

    history_reducer = ChatHistoryTruncationReducer(target_count=5)

    chat = AgentGroupChat(
        agents=[agent_reviewer, agent_writer],
        selection_strategy=KernelFunctionSelectionStrategy(
            initial_agent=agent_reviewer,
            function=selection_function,
            kernel=kernel,
            result_parser=lambda r: str(r.value[0]).strip() if r.value and r.value[0] else WRITER_NAME,
            history_variable_name="lastmessage",
            history_reducer=history_reducer,
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[agent_reviewer],
            function=termination_function,
            kernel=kernel,
            result_parser=lambda r: termination_keyword in str(r.value[0]).lower() if r.value and r.value[0] else False,
            history_variable_name="lastmessage",
            maximum_iterations=10,
            history_reducer=history_reducer,
        ),
    )
    return chat

def add_user_message(chat, text: str):
    return run_async(chat.add_chat_message(message=text))

def invoke_chat(chat):
    async def _collect():
        outputs = []
        async for msg in chat.invoke():
            outputs.append(msg)
        return outputs
    return run_async(_collect())

def main():
    chat = build_chat()
    print("Ready! Type your input, or 'exit' to quit, 'reset' to restart the conversation. You may use @<file>.")

    while True:
        print()
        user_input = input("User > ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "reset":
            run_async(chat.reset())
            print("[Conversation has been reset]")
            continue
        if user_input.startswith("@") and len(user_input) > 1:
            file_name = user_input[1:]
            file_path = os.path.join(os.getcwd(), file_name)
            if not os.path.exists(file_path):
                print(f"Unable to access file: {file_path}")
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                user_input = f.read()

        add_user_message(chat, user_input)

        try:
            responses = invoke_chat(chat)
            for r in responses:
                if r and r.name:
                    print()
                    print(f"# {r.name.upper()}:\n{r.content}")
        except Exception as e:
            print(f"Error: {e}")

        chat.is_complete = False  # allow continued turns

if __name__ == "__main__":
    main()
