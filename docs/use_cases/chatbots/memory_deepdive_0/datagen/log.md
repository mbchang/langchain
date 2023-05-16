# Questions:
- can I instantiate a ConversationBufferMemory (and similar) from a ChatMessageHistory?
    - Answer: yes, you should be able to just assign memory.chat_memory to the ChatMessageHistory
- can I extend the ChatMessageHistory with another ChatMessageHistory?
    - from looking at the code it doesn't look like it.

# Issues:
- I cannot extend the ChatMessageHistory with another ChatMessageHistory
- load_memory_variables in ConversationBufferMemory does not use the inputs argument
- right now the memory objects force you to add an (Human, AI) message pair together, but I want to be able to add them separately
- Can I just have an LLM chain that does not take a prompt template? I just want it to wrap a normal LLM without a predefined prompt template.
- Probably need to also make the "time-weighted" feature compatible with all other memory systems too

# Observations:
- `last_n_tokens`:
    - When I ask "who are the narrator's friends?" the context window matters.
        - With gpt-4, if the context window is 8192, then it answers "The narrator's friends include Carter, Austin, Eliza, and Josh."
        - With gpt-4, if the context window is 4096, then it answers "The narrator's friends include Carter and Austin."
    - GPT-4 gets the answer right, but this is how gpt-3.5-turbo (with last n tokens) answers:
        Question: What is the narrator’s name?
        Response: The narrator's name is Nick.
        =====================================
        Question: Who are the narrator’s friends?
        Response: The narrator's friends include Carter, Austin, Eliza, and Josh.
    - running GPT-4 a bit later, and I get the following:
        ```
        Question: What is the narrator's name?
        Response: I'm sorry, but I don't have any information about the narrator's name in this conversation.
        ================================================================================
        Question: Who are the narrator's friends?
        Response: The narrator's friends include Carter, Eliza, Austin, and Josh.
        ================================================================================
        Question: List all the characters.
        Response: Here's a list of the characters mentioned in our conversation:

        1. Nick (you)
        2. Eliza
        3. Carter
        4. Austin
        5. Mr. Hoover
        6. Madison Hayes
        7. Josh Daley
        8. Robert
        9. Jamal
        10. Ms. O'Connor
        11. Hannah
        12. Nick's mom
        13. Nick's dad
        ```
        so this seems to suggest that the answer is there, but it just does not do proper memory retrieval.
        or perhaps this is because it doesn't know who the narrator is?

- `full_summary`:
    - takes a very long time to ingest the entire history
- `last_n_tokens_with_summary`:
    - also takes a long time because you recompute the summary after each additional memory that gets added into the buffer once the buffer exceeds its limit
    - it took 50 minutes to load and summarize 17 chapters.


- overall
    - there may be a potential tradeoff between doing a lot of upfront computation of summarizing the history vs doing retrieval at inference time
        - probably need caching algorithms


# Future Work
- compare with retrieval mechanisms too?