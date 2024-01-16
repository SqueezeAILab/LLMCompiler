from langchain.prompts.prompt import PromptTemplate

_PREFIX = (
    "Solve a question answering task with interleaving Thought, Action, Observation steps.\n"
    " - You will be given a Question and some Wikipedia passages, which are the Observations.\n"
    " - Thought needs to reason about the question based on the Observations in 1-2 sentences.\n"
    " - There are cases where the Observations are unclear or irrelevant (in the case wikipedia search was not successful). "
    "In such a case where the Observations are unclear, you must make a best guess based on your own knowledge if you don't know the answer. "
    "You MUST NEVER say in your thought that you don't know the answer.\n"
    # comment this for unoptimized react
    " - NEVER search the same entities twice. Otherwise, you will be PANALIZED. "
    "If the information is not available, use your own knowledge.\n"
    "\n"
    " - After Thought, you MUST always take an Action. Action can be two types:\n"
    "(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. "
    # comment this for unoptimized react
    "You MUST find all entities to make the most informed decision. Never search the same entity more than once.\n"
    " - Answer should be short and a single item and MUST not be multiple choices. "
    "Answer MUST NEVER be 'unclear', 'unknown', 'neither', 'unrelated' or 'undetermined', and otherwise you will be PENALIZED.\n"
    # comment this for unoptimized react
    " - You MUST NEVER search two entities more than once. Even if the search result is irrelavant or unavilable, you MUST NOT search it again or you will be PANALIZED. "
    # comment this for unoptimized react
    "Just say you will use your own knowledge and move on to produce the final answer.\n"
    "(2) Finish[answer], which returns the answer and finishes the task. After this action, you MUST output <END_OF_RESPONSE> to finish the task.\n"
    "Here are some examples."
)

EXAMPLES = [
    """Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action: Search[Arthur's Magazine]
Observation: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
Thought: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action: Search[First for Women (magazine)]
Observation: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.
Thought: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action: Finish[Arthur's Magazine]<END_OF_RESPONSE>""",
    """Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought: I need to search Pavel Urysohn first.
Action: Search[Pavel Urysohn]
Observation: Pavel Samuilovich Urysohn (February 3, 1898 - August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought: Then I need to search Leonid Levin.
Action: Search[Leonid Levin]
Observation: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.
Thought: Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.
Action: Finish[yes]<END_OF_RESPONSE>""",
    """Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action: Search[Pavel Urysohn]
Observation: Could not find [Pavel Urysohn].
Thought: I couldn't find Pavel Urysohn, so I will use my own knowledge. I need to search Leonid Levin next and the type of work.
Action: Search[Leonid Levin]
Observation: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.
Thought: Leonid Levin is a mathematician, and base on my knowledge Pavel Urysohn is also a mathematician. So they have the same type of work.
Action: Finish[yes]<END_OF_RESPONSE>""",
]


SUFFIX = """\nQuestion: {input}
{agent_scratchpad}"""

PROMPT = PromptTemplate.from_examples(
    EXAMPLES, SUFFIX, ["input", "agent_scratchpad"], prefix=_PREFIX
)
