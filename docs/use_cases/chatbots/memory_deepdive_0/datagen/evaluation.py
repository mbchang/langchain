"""
1. import questions
2. import dialogues
3. for each memory model:
    - apply the memory model to the dialogues
    - for each question:
        - initialize LLMChain with the memory model
        - ask the LLMChain to respond to the question
        - save the response
4. generate a plot of the results

Check out the log at log.md

TODO:
- add argument parser
- save results to json file
- be able to load results from json file and "use cached" results
    - have an argument to "use cached" results
-
"""
import argparse
from datetime import datetime
import json
import os
from tqdm import tqdm

from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import (
    ChatMessageHistory,
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory,
)
from langchain.schema import AIMessage, HumanMessage

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt-3.5-turbo", help="name of the model to use")
parser.add_argument("--memory", help="name of the memory model to use")
parser.add_argument("--question", help="specific question to run")
parser.add_argument(
    "--use_cache", action="store_true", help="use cached responses if available"
)
parser.add_argument("--debug", action="store_true", help="turn on debugging")

args = parser.parse_args()

#############
# CONSTANTS #
#############


MAX_TOKENS = {"gpt-3.5-turbo": 4096, "gpt-4": 8192}
LLM = ChatOpenAI(model_name=args.model, temperature=0)

RESULTS_DIR = "results"
QUESTIONS_FILE = "questions.txt"
QUESTIONS_DEBUGFILE = "questions_debug.txt"
DIALOGUE_DIR = "dialogues/5-15-23"
MEMORY_MODELS = {
    "memoryless": None,
    "full_list": lambda: ConversationBufferMemory(),
    "last_n_tokens": lambda: ConversationTokenBufferMemory(
        llm=LLM,
        max_token_limit=MAX_TOKENS[args.model],
    ),
    "full_summary": lambda: ConversationSummaryMemory(llm=LLM),
    "last_n_tokens_with_summary": lambda: ConversationSummaryBufferMemory(
        llm=LLM, max_token_limit=MAX_TOKENS[args.model]
    ),
}


###############
# DATALOADING #
###############


def load_data(data_dir):
    dialogues = {}
    for filename in os.listdir(data_dir):
        with open(f"{data_dir}/{filename}", "r") as f:
            dialogues[filename] = f.read()

    # turn each dialogue into a chat history object
    dialogue_chats = {}
    for filename, dialogue in dialogues.items():
        dialogue_chats[filename] = convert_to_chat_history(dialogue)
    return dialogue_chats


def convert_to_chat_history(dialogue):
    chat_history = ChatMessageHistory()
    narrator_delim = "Narrator: "
    ai_delim = "AI: "
    for message in dialogue.split("\n\n"):
        if message.startswith(narrator_delim):
            chat_history.add_user_message(message[len(narrator_delim) :])
        elif message.startswith(ai_delim):
            chat_history.add_ai_message(message[len(ai_delim) :])
        else:
            raise ValueError(
                f"Message does not start with {narrator_delim} or {ai_delim}"
            )
    return chat_history


##########
# MEMORY #
##########


def extend_memory(memory, dialogue_chat):
    """
    TODO: add this capability to ChatMessageHistory
    """
    for i in range(0, len(dialogue_chat.messages) - 1, 2):
        human_message = dialogue_chat.messages[i]
        ai_message = dialogue_chat.messages[i + 1]
        assert isinstance(human_message, HumanMessage)
        assert isinstance(ai_message, AIMessage)
        memory.save_context(
            inputs={"input": human_message.content},
            outputs={"output": ai_message.content},
        )
    return memory


def ingest_dialogues(memory, dialogue_chats):
    print(f"Ingesting {len(dialogue_chats)} dialogues...")
    if memory is not None:
        for dialogue_name, dialogue_chat in tqdm(dialogue_chats.items()):
            memory = extend_memory(memory, dialogue_chat)
    return memory


###########
# LOGGING #
###########


def load_json(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = {}
    return data


def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def create_results_dir():
    # Get the current date and hour
    current_date_hour = datetime.now().strftime("%Y-%m-%d_%H")

    # Get the current minute and round to the nearest half hour
    current_minute = datetime.now().minute
    rounded_minute = "00" if current_minute < 30 else "30"

    # Create a new directory for the current date and hour if it doesn't already exist
    results_dir = f"{RESULTS_DIR}/{current_date_hour}-{rounded_minute}"
    os.makedirs(results_dir, exist_ok=True)

    return results_dir


# can cache and load responses
def main():
    dialogue_chats = load_data(DIALOGUE_DIR)
    if args.debug:
        questions = open(QUESTIONS_DEBUGFILE, "r").read().split("\n")
    else:
        questions = open(QUESTIONS_FILE, "r").read().split("\n")

    for memory_name, memory in MEMORY_MODELS.items():
        if args.memory and memory_name != args.memory:
            continue
        print(f"Memory model: {memory_name}")
        print("-" * 20)

        results_dir = create_results_dir()

        responses_filename = f"responses_{args.model}_{memory_name}.json"
        if args.debug:
            responses_filename = "debug_" + responses_filename
        responses_path = os.path.join(results_dir, responses_filename)

        responses = load_json(responses_path)

        if memory is not None:
            memory = ingest_dialogues(memory(), dialogue_chats)
            chain = ConversationChain(llm=LLM, memory=memory)
        else:
            chain = LLMChain(llm=LLM, prompt=PromptTemplate.from_template("{input}"))

        for question in questions:
            if args.question and question != args.question:
                continue
            print(f"Question: {question}")

            if question in responses and args.use_cache:
                print(f"Response: {responses[question]} (cached)")
            else:
                try:
                    response = chain.predict(input=question)
                except Exception as e:
                    print(f"Exception: {e}")
                    response = str(e)
                print(f"Response: {response}")
                responses[question] = response
            print("=" * 20)
        save_json(responses_path, responses)
        print("#" * 20)


if __name__ == "__main__":
    main()
