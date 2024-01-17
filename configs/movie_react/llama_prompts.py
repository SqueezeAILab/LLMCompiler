from langchain.prompts.prompt import PromptTemplate

_PREFIX = (
    "Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be two types:\n"
    "(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n"
    " Sometimes, Search can fail and return irrelevant information. If this happens, just ignore it and continue -- you should NEVER retry searching the same entity or different options.\n"
    "(2) Finish[answer], which returns the answer and finishes the task. After this action, you MUST output <END_OF_RESPONSE> to finish the task.\n"
    " Never provide answers with explanations or descriptions. answer MUST always be a movie title, nothing else. Answer should never be 'None' or 'I don't know'.\n"
    "Guidelines:\n"
    "- You MUST always output with Finish[answer] even if you are not sure about the answer.\n"
    # comment the following sentence for the unoptimized react
    "- You MUST find all movies and NEVER find the same movies twice.\n"
    "- The Search result may not be available (in case search failed) or not relevant to movie information. If this happens, just ignore it and rely on your own knowledge on that movie.\n"
    "- When you provide the final answer, you MUST use the format Finish[answer], and must NEVER use any other format. answer MUST be a movie title, nothing else. "
    "NEVER provide any other explanation or description. If you don't follow this format, you will be PANALIZED.\n"
    "- When making the final answer, you MUST first rely on your own knowledge. Information is only there to help you make a better decision.\n"
    "\n"
    "Here are some examples:\n"
)

EXAMPLES = [
    """
    Question: Find a movie similar to Interstellar, 2001: A Space Odyssey, The Martian, Star Wars Episode IV - A New Hope.
    Options:
    Saving Private Ryan
    Ender's Game
    Avatar
    Titanic

    Thought: I need to search info about each movie, starting with Interstellar.
    Action: Search[Interstellar]
    Observation: Interstellar is a 2014 epic science fiction. Set in a dystopian future where humanity is embroiled in a catastrophic blight and famine, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for humankind.
    Thought: Need info about 2001: A Space Odyssey.
    Action: Search[2001: A Space Odyssey]
    Observation: 2001: A Space Odyssey is a 1968 epic science fiction film produced and directed by Stanley Kubrick.
    Thought: Need info about The Martian.
    Action: Search[The Martian]
    Observation: The Martian is a 2015 science fiction film directed by Ridley Scott and starring Matt Damon. he film depicts an astronaut's lone struggle to survive on Mars after being left behind and the efforts of NASA to rescue him and bring him home to Earth.
    Thought: Need info about Star Wars Episode IV - A New Hope.
    Action: Search[Star Wars Episode IV - A New Hope]
    Observation: Star Wars is a 1977 American epic space opera film. Set "a long time ago" in a fictional universe where the galaxy is ruled by the tyrannical Galactic Empire, the story focuses on a group of freedom fighters known as the Rebel Alliance, who aim to destroy the Empire's newest weapon, the Death Star.
    Thought: Need info about Saving Private Ryan.
    Action: Search[Saving Private Ryan]
    Observation: Saving Private Ryan is a 1998 American epic war film. Set in 1944 in France during World War II, it follows a group of soldiers, led by Captain John Miller (Tom Hanks), on their mission to locate Private James Francis Ryan and bring him home safely after his three brothers are killed in action.
    Thought: Need info about Ender's Game.
    Action: Search[Ender's Game]
    Observation: Ender's Game is a 1985 military science fiction novel by American author Orson Scott Card. Set at an unspecified date in Earth's future, the novel presents an imperiled humankind after two conflicts with an insectoid alien species they dub "the buggers".
    Thought: Need info about Avatar.
    Action: Search[Avatar]
    Observation: Avatar is a 2009 epic science fiction film. It is the first installment in the Avatar film series. It is set in the mid-22nd century, when humans are colonizing Pandora, a lush habitable moon of a gas giant in the Alpha Centauri star system, in order to mine the valuable mineral unobtanium.
    Thought: Need info about Titanic.
    Action: Search[Titanic]
    Observation: Titanic is a 1997 American disaster film. Kate Winslet and Leonardo DiCaprio star as members of different social classes who fall in love during the ship's maiden voyage.
    Thought: Based on my knowledge and the observations, Ender's Game is most similar.
    Action: Finish[Ender's Game]<END_OF_RESPONSE>""",
    """
    Question: Find a movie similar to Mission Impossible, The Silence of the Lambs, American Beauty, Star Wars Episode IV - A New Hope
    Options:
    Austin Powers International Man of Mystery
    Alesha Popvich and Tugarin the Dragon
    In Cold Blood
    Rosetta

    Thought: I need to search info about each movie, starting with Mission Impossible.
    Action: Search[Mission Impossible]
    Observation: Mission: Impossible is an American multimedia franchise based on a fictional secret espionage agency known as the Impossible Missions Force (IMF).
    Thought: Need info about The Silence of the Lambs.
    Action: Search[The Silence of the Lambs]
    Observation: Could not find [The Silence of the Lambs]
    Thought: Need info about American Beauty.
    Action: Search[American Beauty]
    Observation: Could not find [American Beauty].
    Thought: Need info about Star Wars Episode IV - A New Hope.
    Action: Search[Star Wars Episode IV - A New Hope]
    Observation: Star Wars is a 1977 American epic space opera film written and directed by George Lucas. Set "a long time ago" in a fictional universe where the galaxy is ruled by the tyrannical Galactic Empire, the story focuses on a group of freedom fighters known as the Rebel Alliance.
    Thought: Need info about Austin Powers International Man of Mystery.
    Action: Search[Austin Powers International Man of Mystery]
    Observation: Austin Powers: International Man of Mystery is a 1997 American spy comedy film. It is the first installment in the Austin Powers series.
    Thought: Need info about Alesha Popvich and Tugarin the Dragon.
    Action: Search[Alesha Popvich and Tugarin the Dragon]
    Observation: Could not find Alesha Popovich and Tugarin the Dragon.
    Thought: Need info about In Cold Blood.
    Action: Search[In Cold Blood]
    Observation: In Cold Blood is a non-fiction novel. It details the 1959 murders of four members of the Clutter family in the small farming community of Holcomb, Kansas.. Capote learned of the quadruple murder before the killers were captured, and he traveled to Kansas to write about the crime.
    Thought: Need info about Rosetta.
    Action: Search[Rosetta]
    Observation: Rosetta s a port city of the Nile Delta, 65 km (40 mi) east of Alexandria, in Egypt's Beheira governorate.
    Thought: Based on my knowledge and the observations, Austin Powers International Man of Mystery is most similar.
    Action: Finish[Austin Powers International Man of Mystery]<END_OF_RESPONSE>""",
]


SUFFIX = """\n
Question: {input}
Thought:{agent_scratchpad}"""

PROMPT = PromptTemplate.from_examples(
    EXAMPLES[:], SUFFIX, ["input", "agent_scratchpad"], prefix=_PREFIX
)
