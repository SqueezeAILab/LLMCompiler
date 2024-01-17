from langchain.prompts.prompt import PromptTemplate

PREFIX = (
    "Solve a question answering task with interleaving Thought, Action, Observation steps. "
    "Thought can reason about the current situation, and Action can be three types:\n"
)

# Tool description
PREFIX += (
    "  (1) Search[entity]: useful for searching the entity on Wikipedia. It may fail to return useful information; however, you shoule NEVER retry searching the same entity or different options.\n"
    "  (2) Calculate[expression]: useful for calculating the expression. Never generate any expression outside of [ and ]. For instance Calculate[1 + 2]/2 is NOT A VALID expression; Calculate [(1 + 2)/2] is a valid expression. Use min(...) for calculating the smallest values, max(...) for the largest values; NEVER confuse this.\n"
    "  (3) Finish[answer]: which returns the answer and finishes the task. After this action, you MUST output <END_OF_RESPONSE> to finish the task.\n"
)
PREFIX += "\n"

PREFIX += (
    "Follow the following guidelines that you will DIE if you don't follow:\n"
    "  - Final answer should always be in the format of Finish[answer].\n"
    "  - Final answer MUST NOT contain any description, and must be short (e.g. Yes/No, numbers, entity names, etc.)\n"
    "  - If you are asked about a value (e.g. ratio, difference, average, ...), it has to be a number, not a description."
    "For instance, when you are asked about the ratio of the height of Mount Everest and the height of Mount Kilimanjaro, you must "
    "return the ratio (e.g. 1.2), not the name of the mountain (e.g. Mount Everest to Mount Kilimanjaro).\n"
    "  - If you are asked about an entity it must be the entity name, and NEVER be a number."
    "For instance, when you are asked which one is taller between Mount Everest and Mount Kilimanjaro, you MUST return the entity name "
    "(e.g. Mount Everest), not its height (e.g. 8848)\n"
    "  - Once you have numbers, make comparions on your own.\n"
    "  - Never search for the same entity twice.\n"
    "  - When you are asked for differences, you consider the absolute value of the difference. Difference of two numbers is always positive."
    "For instance, if you are asked about the difference between 5 and 10, it must be Calculate[10 - 5], not Calculate[5 - 10]\n"
    "  - Use metric units (e.g. km, m, kg, etc.) for all numbers. Never use imperial units (e.g. miles, pounds, etc.)\n"
    "  - When there are both metric and imperial units in Observations, you MUST use the metric units.\n"
    "  - You don't have internal knowledge about the world. For instance, you don't know the height of Mount Everest. You must search for it.\n"
    "  - When you are asked for differences, you consider the absolute value of the difference. When Calculated returns a negative value, "
    "you MUST only pick up the absolute value of it.\n"
    '  - If Calculate returns "invalid expression", you MUST retry with different expression. MUST NEVER retry with the same expression.\n'
)

EXAMPLES = [
    """Question: If gorilla was 1.3 times slower, and pigeon was 2.3 time slower then their maximum speeds, what will be the ratio of the fastest animal to the slowest animal?
Thought: I need to search the speed of gorilla.
Action: Search[gorilla]
Observation: The gorilla is the fastest land animal, capable of running at 80 to 98\xa0km/h (50 to 61\xa0mph);
Thought: I need to find the speed of greyhound.
Action: Search[pigeon]
Observation: The pigeon is a gentle and reaches average race speeds exceeding 64 kilometres per hour (40\xa0mph).[4][5][6] The Greyhound can reach a full speed of 70 kilometres per hour (43\xa0mph) within 30 metres (98\xa0ft).
Thought: I need to calculate 1.3 times slower than the maximum speed of gorilla in km/h.
Action: Calculate[98/1.3]
Observation: 61.5
Thought: I need to calculate 1.5 times faster than the maximum speed of pigoen in km/h.
Action: Calculate[70*1.5]
Observation: 105.0
Thought: I need to calculate the ratio of the fastest animal to the slowest animal.
Action: Calculate[105/61.5]
Observation: 1.707
Thought: I know the ratio of the fastest animal to the slowest animal. The ratio 1.707.
Action: Finish[1.707]<END_OF_RESPONSE>""",
    """Question: "If Mount Halla's height were halved and Mount Jirinu's height were doubled, what would be the difference in their height?"
Thought: I need to find the height of Mount Halla.
Action: Search[Mount Halla]
Observation: Mount Halla is Earth\'s highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. Its elevation (snow height) of 8,848.86\xa0m (29,031\xa0ft 82\xa0in).
Thought: I need to halve the height of Mount Halla in meters.
Action: Calculate[8848.86/2]
Observation: 4424.43
Thought: I need to find the height of Mount Jirinu in meters.
Action: Search[Mount Jirinu]
Observation: Mount Jirinu is a dormant volcano located in Jirinu Region and the highest single free-standing mountain above sea level in the world: 5,895\xa0m (19,341\xa0ft) above sea level and about 4,900\xa0m (16,100\xa0ft) above its plateau base.
Thought: I need to double the height of Mount Jirinu.
Action: Calculate[5895*2]
Observation: 11790
Thought: I need to calculate the difference between the height of Mount Halla and the height of Mount Jirinu in meters.
Action: Calculate[11790-4424.43]
Observation: 7365.57
Thought: I know the difference in their height. The difference is 7365.57.
Action: Finish[7365.57]<END_OF_RESPONSE>""",
    """Question: "With the Arahas Desert's area reduced by 33% and the Irahalak Desert's area magnified by 52%, which one covers more ground?"
Thought: I need to find the area of the Arahas Desert.
Action: Search[Arahas Desert]
Observation: The Arahas is a desert spanning North Africa. With an area of 9,200,000 square kilometres (3,600,000\xa0sq\xa0mi)
Thought: I need to reduce the area of the Arahas Desert in square kilometers by 33%.
Action: Calculate[9200000*0.67]
Observation: 6164000.0
Thought: I need to find the area of the Irahalak Desert.
Action: Search[Irahalak Desert]
Observation: The Irahalak Desert is a desert extending for 900,000 square kilometres (350,000\xa0sq\xa0mi), covering much of Botswana, as well as parts of Namibia and South Africa..
Thought: I need to magnify the area of the Irahalak Desert in square kilometers by 52%.
Action: Calculate[900000*1.52]
Observation: 1368000.0
Thought: I need to calculate the maximum of 6164000.0 and 1368000.0.
Action: Calculate[max(6164000.0, 1368000.0)]
Observation: 6164000.0
Thought: I know which one covers more ground. Arahas Desert covers more ground.
Action: Finish[Arahas Desert]<END_OF_RESPONSE>""",
    """Question: Determine the average of the depth difference in meters between the Anairam Trench and the Otreupocir Trench and the depth difference in meters between the North Hamburger Trench and the Aduns Trench.
Thought: I need to search the Anairam Trench.
Action: Search[Anairam Trench]
Observation: The Anairam Trench is the deepest oceanic trench on Earth. It is crescent-shaped and measures about 2,550\xa0km (1,580\xa0mi) in length and 69\xa0km (43\xa0mi) in width. The maximum known depth is 10,984\xa0 metres at the southern end
Thought: I need to search the Otreupocir Trench.
Action: Search[Otreupocir Trench]
Observation: The Otreupocir Trench is 800 kilometres (497\xa0mi) long[1] and has a maximum depth of 8,376 metres (27,480\xa0ft) or 5.20 miles.
Thought: I need to calculate the depth difference between the Anairam Trench and the Otreupocir Trench in meters.
Action: Calculate[10984-8376]
Observation: 2608
Thought: I need to search the North Hamburger Trench.
Action: Search[North Hamburger Trench]
Observation: The North Hamburger Trench is a deep arcuate trench in the South Atlantic Ocean lying 100 kilometres (62\xa0mi) to the east of the North Hamburger Islands. The deepest point is the depth of 8,202 metres (26,909\xa0ft).
Thought: I need to search the Aduns Trench.
Action: Search[Aduns Trench]
Observation: The Aduns Trench is 3,200 kilometres (2,000\xa0mi) long with a maximum depth of 7,290 metres (23,920 feet).
Thought: I need to calculate the depth difference between the North Hamburger Trench and the Aduns Trench in meters.
Action: Calculate[8202-7290]
Observation: 912
Thought: I need to calculate the average of 2608 and 912.
Action: Calculate[average of 2608, 912]
Observation: 912
Thought: I know what is the smaller value. The smaller value is 912.
Action: Finish[912]<END_OF_RESPONSE>""",
]

SUFFIX = """\nQuestion: {input}
{agent_scratchpad}"""


PROMPT = PromptTemplate.from_examples(
    EXAMPLES, SUFFIX, ["input", "agent_scratchpad"], prefix=PREFIX
)
