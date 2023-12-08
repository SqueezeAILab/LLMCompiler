from src.llm_compiler.constants import END_OF_PLAN, JOINNER_FINISH

PLANNER_PROMPT = (
    "Question: Which magazine was started first Arthur's Magazine or First for Women?\n"
    '1. search("Arthur\'s Magazine")\n'
    '2. search("First for Women (magazine)")\n'
    "Thought: I can answer the question now.\n"
    f"3. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?\n"
    '1. search("Pavel Urysohn")\n'
    '2. search("Leonid Levin")\n'
    "Thought: I can answer the question now.\n"
    f"3. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
)

OUTPUT_PROMPT = (
    "Solve a question answering task with interleaving Observation, Thought, and Action steps. Here are some guidelines:\n"
    "  - You will be given a Question and some Wikipedia passages, which are the Observations.\n"
    "  - Thought needs to reason about the question based on the Observations in 1-2 sentences.\n"
    "  - There are cases where the Observations are unclear or irrelevant (in the case wikipedia search was not successful). In such a case where the Observations are unclear, you must make a best guess based on your own knowledge if you don't know the answer. You MUST NEVER say in your thought that you don't know the answer.\n\n"
    "Action can be only one type:\n"
    f" (1) {JOINNER_FINISH}(answer): returns the answer and finishes the task. "
    "Answer should be short and a single item and MUST not be multiple choices. Answer MUST NEVER be 'unclear', 'unknown', 'neither', 'unrelated' or 'undetermined', and otherwise you will be PENALIZED.\n"
    "\n"
    "Here are some examples:\n"
    "\n"
    "Question: Which magazine was started first Arthur's Magazine or First for Women?\n"
    "\n"
    "search(Arthur's Magazine)\n"
    "Observation: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.\n"
    "search(First for Women (magazine))\n"
    "Observation: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.\n"
    "Thought: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.\n"
    f"Action: {JOINNER_FINISH}(Arthur's Magazine)\n"
    "###\n"
    "\n"
    "Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?\n"
    "search(Pavel Urysohn)\n"
    "Observation: Pavel Samuilovich Urysohn (February 3, 1898 - August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.\n"
    "search(Leonid Levin)\n"
    "Observation: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.\n"
    "Thought: Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.\n"
    f"Action: {JOINNER_FINISH}(yes)\n"
    "###\n"
    "\n"
)
