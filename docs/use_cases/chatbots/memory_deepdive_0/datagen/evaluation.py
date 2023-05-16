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


MODEL_NAME = "gpt-3.5-turbo"
# MODEL_NAME = "gpt-4"
MAX_TOKENS = {"gpt-3.5-turbo": 4096, "gpt-4": 8192}
LLM = ChatOpenAI(model_name=MODEL_NAME, temperature=0)

QUESTIONS_FILE = "questions.txt"
DIALOGUE_DIR = "dialogues/5-15-23"
MEMORY_MODELS = {
    # "memoryless": None,
    # "full_list": lambda: ConversationBufferMemory(),
    # "last_n_tokens": lambda: ConversationTokenBufferMemory(
    #     llm=LLM,
    #     max_token_limit=MAX_TOKENS[MODEL_NAME],
    #     # max_token_limit=4096,
    # ),
    # "full_summary": lambda: ConversationSummaryMemory(llm=LLM),
    "last_n_tokens_with_summary": lambda: ConversationSummaryBufferMemory(
        llm=LLM, max_token_limit=MAX_TOKENS[MODEL_NAME]
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
# RESULTS #
###########


# can cache and load responses
def main():
    dialogue_chats = load_data(DIALOGUE_DIR)
    questions = open(QUESTIONS_FILE, "r").read().split("\n")

    for memory_name, memory in MEMORY_MODELS.items():
        print(f"Memory model: {memory_name}")
        if memory is not None:
            memory = ingest_dialogues(memory(), dialogue_chats)
            chain = ConversationChain(llm=LLM, memory=memory)
        else:
            chain = LLMChain(llm=LLM, prompt=PromptTemplate.from_template("{input}"))
            # chain = lambda x: llm([HumanMessage(content=x)])

        for question in questions:
            print(f"Question: {question}")
            try:
                response = chain.predict(input=question)
            except Exception as e:
                print(f"Exception: {e}")
                response = e
            print(f"Response: {response}")
            print("=====================================")


if __name__ == "__main__":
    main()
