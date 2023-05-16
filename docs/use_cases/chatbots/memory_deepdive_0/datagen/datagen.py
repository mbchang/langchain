"""
Chapters: Prologue, RULE NUMBER 1, RULE NUMBER 2, ..., RULE NUMBER 22, Epilogue, The Bro Code, ACKNOWLEDGMENTS

Written in first person.
Imagine a dialogue that the protagonist is talking with their AI companion about the events of each chapter.

Notes:
- "The AI companion should address the narrator by their name, which you need to infer from context." GPT-4 and GPT-3.5 fail at this.
If I directly ask the question: "what is the name of the narrator", GPT4 answers corectly. But if I just give it the above prompt, then GPT-4, GPT-3.5 fails

"""
import asyncio
from collections import OrderedDict
import re

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import Docx2txtLoader

from langchain.schema import HumanMessage, SystemMessage


# MODEL_NAME = "gpt-3.5-turbo"  # does not infer the narrator's name from the text to generate the dialogue, even when explictly asked
MODEL_NAME = "gpt-4"  # by simply asking it to address the narrator by name, it also does not automatically infer the narrator's name
# what if we explicitly ask it to infer the narrator's name first?

MAX_TOKENS = {"gpt-3.5-turbo": 4096, "gpt-4": 8192}


def load_data():
    loader = Docx2txtLoader("data/bro_code.docx")
    data = loader.load()[0]
    return data.page_content


def load_chapter_delimiters():
    CHAPTERS = OrderedDict(
        {
            "PROLOGUE": r"PROLOGUE\n\nDear Nick, \n\nWhat keeps sticking out is the first thing you said to me: Tell me I’m wrong.",
            "RULE NUMBER 1": r"RULE NUMBER 1\n\nBros",
            "RULE NUMBER 2": r"RULE NUMBER 2\n\nA bro shalt not get",
            "RULE NUMBER 3": r"RULE NUMBER 3\n\nA bro shalt always finish telling his joke.",
            "RULE NUMBER 4": r"RULE NUMBER 4\nThe Dating Clause: If a girl matches any of the following criteria,",
            "RULE NUMBER 5": r"RULE NUMBER 5\n\nA bro shalt not wake up before 11:00 a.m. on a Saturday.\n\n\n\nThe clock read 10:07 a.m.",
            "RULE NUMBER 6": r"RULE NUMBER 6\n\nA bro shalt not swim in",
            "RULE NUMBER 7": r"RULE NUMBER 7\n\nA bro shalt set up another bro only if he has asked to be set up.",
            "RULE NUMBER 8": r"RULE NUMBER 8\n\nA bro shalt not back down from spicy foods.",
            "RULE NUMBER 9": r"RULE NUMBER 9\n\nA bro shalt not shop.\n\n\n\nEliza",
            "RULE NUMBER 10": r"RULE NUMBER 10\n\nA bro shalt make the first move. \n\n\n\nFinally, it was Friday",
            "RULE NUMBER 11": r"RULE NUMBER 11\n\nThe forty-eight-hour rule. \n\n\n\nEliza.",
            "RULE NUMBER 12": r"RULE NUMBER 12\n\nA bro shalt not talk to a chick on the phone unless she is his relative or girlfriend.",
            "RULE NUMBER 13": r"RULE NUMBER 13\n\nA bro shalt treat his mother like a queen.",
            "RULE NUMBER 14": r"RULE NUMBER 14\n\nA bro shalt never half-ass a first date.",
            "RULE NUMBER 15": r"RULE NUMBER 15\n\nA bro shalt not tell non-bros the rules of the Bro Code.",
            "RULE NUMBER 16": r"RULE NUMBER 16\n\nA bro shalt always check his phone.\n\n\n\nWe played in the pool for a long time.",
            "RULE NUMBER 17": r"RULE NUMBER 17\n\nA bro shalt always keep his cool. \n\n\n\nI",
            "RULE NUMBER 18": r"RULE NUMBER 18\n\nA bro shalt not use the",
            "RULE NUMBER 19": r"RULE NUMBER 19\n\nA bro shalt not punch another bro in his face.",
            "RULE NUMBER 20": r"RULE NUMBER 20\n\nA bro shalt not complain about working out.",
            "RULE NUMBER 21": r"RULE NUMBER 21 \n\nA bro shalt not make his girl cry. \n\n",
            "RULE NUMBER 22": r"RULE NUMBER 22\n\nA bro shalt cheat only on his homework.",
            "Epilogue": r"Epilogue \n\n",
            "The Bro Code": r"The Bro Code\n\nBy Austin Banks, Nick Maguire, and Carter O’Connor—the OBs\n\n\n\n",
            "ACKNOWLEDGEMENTS": r"ACKNOWLEDGMENTS \n\n\n\nFirst, I thank you, cool reader,",
        }
    )
    return CHAPTERS


def find_split_intervals(data, chapter_delimiters):
    """
    Find the split points of the data based on the chapter delimiter.

    To generate the split intervals, we assume we do not skip any chapters.
    """
    split_points = []
    for chapter_name, chapter_delimiter in chapter_delimiters.items():
        # if there is only one match, then add the index of the match
        # otherwise, raise an error
        matches = re.findall(chapter_delimiter, data)

        if len(matches) == 1:
            split_points.append((chapter_name, data.index(matches[0])))
        else:
            import ipdb

            ipdb.set_trace(context=20)
            raise ValueError("Chapter delimiter is not unique")

    split_intervals = []
    for i in range(len(split_points)):
        chapter_name, start_index = split_points[i]
        if i == len(split_points) - 1:
            end_index = len(data)
        else:
            end_index = split_points[i + 1][1]
        split_intervals.append((chapter_name, start_index, end_index))

    # TESTING
    # print(split_intervals)
    # for split_point in split_intervals:
    #     print(split_point[0], "\t", split_point[1], "\t", split_point[2])

    return split_intervals


def create_data_chunks(data, split_intervals):
    # split data by split points
    data_chunks = {}
    for i in range(len(split_intervals)):
        chapter_name, start_index, end_index = split_intervals[i]
        data_chunks[chapter_name] = data[start_index:end_index]

    # TESTING
    # for name, data_chunk in data_chunks.items():
    #     print(data_chunk[:200])
    #     print("=====================================")
    return data_chunks


def get_max_tokens(input_messages):
    llm = ChatOpenAI(model_name=MODEL_NAME)
    print(f"Number of tokens: {llm.get_num_tokens_from_messages(input_messages)}")
    return MAX_TOKENS[MODEL_NAME] - llm.get_num_tokens_from_messages(input_messages) - 1


async def generate_dialogue(chapter):
    """
    Generate dialogue between the narrator and their AI companion

    Use langchain gpt-4-32k-0314
    """
    system_message = f"""
You are a playwright that specializes in translating novels into dialogues.
You are exceptional at translating novels into dialogues that faithfully reflect the original novel.
You will be given a chapter of a novel written in first-person to translate into a dialogue.
This dialogue should reflect the events that occur in the chapter.

Your task is to write a dialogue between the narrator in the chapter and their AI companion.
The dialogue should simulate the narrator's conversation with their AI companion, as if the narrator in the chapter was relaying the events that happened to them to their AI companion after the events had finished.
The dialogue should focus on the narrator's perspectives, beliefs, thoughts, feelings, and actions.
The AI companion dialogue's should be friendly, supportive, and helpful.
If the AI companion addresses the narrator, the AI companion should address the narrator by their name, which you need to infer from context.

The narrator's name is Nick Maguire.

Formatting instructions:
    Narrator: <narrator dialogue>
    AI: <AI dialogue>
"""

    input_messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=chapter),
    ]
    max_tokens = get_max_tokens(input_messages)

    print(
        f"\nGenerating dialogue with {max_tokens} tokens for chapter {chapter[:100]}..."
    )

    llm = ChatOpenAI(model_name=MODEL_NAME, max_tokens=max_tokens)

    # dialogue = llm.generate([input_messages]).content
    dialogue = await llm.agenerate([input_messages])

    # TESTING
    # print(dialogue)
    # print("=====================================")

    return dialogue


def save_dialogues(chapter_dialogues):
    """
    Save the generated dialogues to a file
    """
    for chapter_name, chapter_dialogue in chapter_dialogues.items():
        chapter_name = chapter_name.replace(" ", "-")
        with open(f"dialogues/{MODEL_NAME}_dialogue_{chapter_name}.txt", "w") as f:
            f.write(chapter_dialogue)


def generate_dialogues(chapters, concurrent=True):
    """
    Generate dialogues for each chapter
    """
    if concurrent:
        loop = asyncio.get_event_loop()
        chapter_dialogues = loop.run_until_complete(
            asyncio.gather(
                *[generate_dialogue(chapter) for chapter in chapters.values()]
            )
        )
    else:
        chapter_dialogues = [
            generate_dialogue(chapter) for chapter in chapters.values()
        ]

    chapter_dialogues = dict(
        zip(
            chapters.keys(),
            [
                chapter_dialogue.generations[0][0].text
                for chapter_dialogue in chapter_dialogues
            ],
        )
    )

    # TESTING
    for chapter_name, chapter_dialogue in chapter_dialogues.items():
        print(chapter_dialogue)
        print("=====================================")

    return chapter_dialogues


def generate_dialogues_in_chunks(chapters, chunk_size):
    if chunk_size == -1:
        return generate_dialogues(chapters)
    else:
        chapter_dialogues = {}
        for i in range(0, len(chapters), 4):
            chapter_group = dict(list(chapters.items())[i : i + 4])
            chapter_dialogues.update(generate_dialogues(chapter_group, concurrent=True))
    return chapter_dialogues


def main():
    # prepare data
    data = load_data()
    chapter_delimiters = load_chapter_delimiters()
    split_intervals = find_split_intervals(data, chapter_delimiters)
    chapters = create_data_chunks(data, split_intervals)

    # remove prologue and acknowledgements and chapter 18
    del chapters["PROLOGUE"]
    del chapters["ACKNOWLEDGEMENTS"]
    del chapters["RULE NUMBER 18"]  # this chapter was too long

    # generate dialogues in chunks: 4 chapters at a time
    chapter_dialogues = generate_dialogues_in_chunks(chapters, chunk_size=4)
    save_dialogues(chapter_dialogues)


if __name__ == "__main__":
    main()
