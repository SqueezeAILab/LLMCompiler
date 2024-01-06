from langchain.prompts.prompt import PromptTemplate

PREFIX = (
    "Solve a question answering task with interleaving Thought, Action, Observation steps. "
    "Thought can reason about the current situation. "
    "After Thought, you MUST always take an Action.and Action can be one of three types:\n"
)

# Tool description
PREFIX = (
    "  (1) Search[entity]: useful for searching the entity on Wikipedia.\n"
    "  (2) Calculate[expression]: useful for calculating the expression.\n"
    "  (3) Finish[answer]: which returns the answer and finishes the task. "
    "You MUST use Finish to finish the task.\n"
)
PREFIX += "\n"

PREFIX += (
    "Follow the following guidelines:\n"
    "  - Final answer should always be in the format of Finish[answer].\n"
    "  - Final answer MUST NOT contain any description, and must be short (e.g. Yes/No, numbers, entity names, etc.)\n"
    "  - Never search for the same entity twice.\n"
    "  - Never introduce new Action types.\n"
    " - When using Calculate, you cannot calculate multiple expressions in one call. For instance, `Calculate('1 + 3, 2 + 4')` does not work. "
    "If you need to calculate multiple expressions, you need to call them separately like Calculate('1 + 3') and then Calculate('2 + 4').\n"
    "  - Once you have numbers, make comparions on your own.\n"
    "  - Use metric units (e.g. km, m, kg, etc.) for all numbers. Never use imperial units (e.g. miles, pounds, etc.)\n"
    "  - When there are both metric and imperial units in Observations, you MUST use the metric units.\n"
    "  - You don't have internal knowledge about the world. For instance, you don't know the height of Mount Everest. You must search for it.\n"
    "  - When you are asked for differences, you consider the absolute value of the difference.\n"
    "  - When you are asked about a value (e.g. ratio, difference, etc.), you must return the value, not the associated entity. "
    "For instance, when you are asked about the ratio of the height of Mount Everest and the height of Mount Kilimanjaro, you must "
    "return the ratio (e.g. 1.2), not the name of the mountain (e.g. Mount Everest to Mount Kilimanjaro).\n"
    "  - You MUST use Finish to finish the task. Otherwise, you will be panalized.\n"
    # " - Use entity name directly as the input to search actions (e.g., 'Barack Obama') rather than their attributes (e.g., 'the age of Barack Obama').\n"
)

EXAMPLES = [
    """If cheetah was 1.3 times slower, greyhound was 1.5 times faster, and falcon was 2.3 time slower then their maximum speeds, what will be the ratio of the fastest animal to the slowest animal?
Thought: I need to search the speed of cheetah.
Action: Search[cheetah]
Observation: The cheetah (Acinonyx jubatus) is a large cat with a tawny to creamy white or pale buff fur that is marked with evenly spaced, solid black spots. Its head is small and rounded, with a short snout and black tear-like facial streaks. It reaches 67–94\xa0cm (26–37\xa0in) at the shoulder, and the head-and-body length is between 1.1 and 1.5\xa0m (3\xa0ft 7\xa0in and 4\xa0ft 11\xa0in). Adults weigh between 21 and 72\xa0kg (46 and 159\xa0lb). It is the fastest land animal, capable of running at 80 to 98\xa0km/h (50 to 61\xa0mph); it has evolved specialized adaptations for speed, including a light build, long thin legs and a long tail..
Thought: I need to find the speed of greyhound.
Action: Search[greyhound]
Observation: The English Greyhound, or simply the Greyhound, is a breed of dog, a sighthound which has been bred for coursing, greyhound racing and hunting. Since the rise in large-scale adoption of retired racing Greyhounds, the breed has seen a resurgence in popularity as a family pet.. Greyhounds are defined as a tall, muscular, smooth-coated, "S-shaped" type of sighthound with a long tail and tough feet. Greyhounds are a separate breed from other related sighthounds, such as the Italian greyhound.[2][3]. The Greyhound is a gentle and intelligent breed whose combination of long, powerful legs, deep chest, flexible spine, and slim build allows it to reach average race speeds exceeding 64 kilometres per hour (40\xa0mph).[4][5][6] The Greyhound can reach a full speed of 70 kilometres per hour (43\xa0mph) within 30 metres (98\xa0ft), or six strides from the boxes, traveling at almost 20 metres per second (66\xa0ft/s) for the first 250 metres (820\xa0ft) of a race.[7][8].
Thought: I need to calculate 1.3 times slower than the maximum speed of cheetah.
Action: Calculate[98/1.3]
Observation: 61.53846153846154
Thought: I need to calculate 1.5 times faster than the maximum speed of greyhound.
Action: Calculate[70*1.5]
Observation: 105.0
Thought: Greyhound was faster.
Action: Finish[greyhound]""",
    """Question: "If Mount Everest's height were halved and Mount Kilimanjaro's height were doubled, what would be the difference in their height?"
Thought: I need to find the height of Mount Everest.
Action: Search[Mount Everest]
Observation: Mount Everest (Nepali: सगरमाथा, romanized:\xa0Sagarmāthā; Tibetan: Chomolungma ཇོ་མོ་གླང་མ; Chinese: 珠穆朗玛峰; pinyin: Zhūmùlǎngmǎ Fēng) is Earth\'s highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The China–Nepal border runs across its summit point.[2] Its elevation (snow height) of 8,848.86\xa0m (29,031\xa0ft 82\xa0in) was most recently established in 2020 by the Chinese and Nepali authorities.[3][4]. Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the "standard route") and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall.
Thought: I need to halve the height of Mount Everest.
Action: Calculate[8848.86/2]
Observation: 4424.43
Thought: I need to find the height of Mount Kilimanjaro.
Action: Search[Mount Kilimanjaro]
Observation: Mount Kilimanjaro (/ˌkɪlɪmənˈdʒɑːroʊ/)[4] is a dormant volcano located in Kilimanjaro Region of Tanzania. It has three volcanic cones: Kibo, Mawenzi, and Shira. It is the highest mountain in Africa and the highest single free-standing mountain above sea level in the world: 5,895\xa0m (19,341\xa0ft) above sea level and about 4,900\xa0m (16,100\xa0ft) above its plateau base. It is the highest volcano in Africa and the Eastern Hemisphere.. Kilimanjaro is the fourth most topographically prominent peak on Earth.
Thought: I need to double the height of Mount Kilimanjaro.
Action: Calculate[5895*2]
Observation: 11790
Thought: I need to calculate the difference between the height of Mount Everest and the height of Mount Kilimanjaro.
Action: Calculate[11790-4424.43]
Observation: 7365.57
Thought: The difference in their height is 7365.57.
Action: Finish[7365.57]""",
    """Question: "With the Sahara Desert's area reduced by 33% and the Kalahari Desert's area magnified by 52%, which one covers more ground?"
Thought: I need to find the area of the Sahara Desert.
Action: Search[Sahara Desert]
Observation: The Sahara (/səˈhɑːrə/, /səˈhærə/) is a desert spanning North Africa. With an area of 9,200,000 square kilometres (3,600,000\xa0sq\xa0mi), it is the largest hot desert in the world and the third-largest desert overall, smaller only than the deserts of Antarctica and the northern Arctic.[1][2][3]. The name "Sahara" is derived from the Arabic word for "desert" in the feminine irregular form, the singular ṣaḥra\' (صحراء /ˈsˤaħra/), plural ṣaḥārā (صَحَارَى /ˈsˤaħaːraː/),[4][5][6][7] ṣaḥār (صَحَار), ṣaḥrāwāt (صَحْرَاوَات), ṣaḥār).. The desert covers much of North Africa, excluding the fertile region on the Mediterranean Sea coast, the Atlas Mountains of the Maghreb, and the Nile Valley in Egypt and the Sudan.[8]. It stretches from the Red Sea in the east and the Mediterranean in the north to the Atlantic Ocean in the west, where the landscape gradually changes from desert to coastal plains.
Thought: I need to reduce the area of the Sahara Desert by 33%.
Action: Calculate[9200000*0.67]
Observation: 6164000.0
Thought: I need to find the area of the Kalahari Desert.
Action: Search[Kalahari Desert]
Observation: The Kalahari Desert is a large semi-arid sandy savanna in Southern Africa extending for 900,000 square kilometres (350,000\xa0sq\xa0mi), covering much of Botswana, as well as parts of Namibia and South Africa.. It is not to be confused with the Angolan, Namibian, and South African Namib coastal desert, whose name is of Khoekhoegowab origin and means "vast place".. Kalahari is derived from the Tswana word Kgala, meaning "the great thirst", or Kgalagadi, meaning "a waterless place";[1] the Kalahari has vast areas covered by red sand without any permanent surface water.. The Kalahari Desert was not always a dry desert. The fossil flora and fauna from Gcwihaba Cave in Botswana indicates that the region was much wetter and cooler at least from 30 to 11 thousand BP (before present), especially after 17,500 BP.[2].
Thought: I need to magnify the area of the Kalahari Desert by 52%.
Action: Calculate[900000*1.52]
Observation: 1368000.0
Thought: Sahara Desert covers more ground.
Action: Finish[Sahara Desert]""",
    """Question: Determine the smaller value: the depth difference in meters between the Mariana Trench and the Puerto Rico Trench, or the depth difference in meters between the South Sandwich Trench and the Sunda Trench.
Thought: I need to search the Mariana Trench.
Action: Search[Mariana Trench]
Observation: The Mariana Trench or Marianas Trench is located in the western Pacific Ocean about 200 kilometres (124\xa0mi) east of the Mariana Islands; it is the deepest oceanic trench on Earth. It is crescent-shaped and measures about 2,550\xa0km (1,580\xa0mi) in length and 69\xa0km (43\xa0mi) in width. The maximum known depth is 10,984\xa0±\xa025 metres (36,037\xa0±\xa082\xa0ft; 6,006\xa0±\xa014 fathoms; 6.825\xa0±\xa00.016\xa0mi) at the southern end of a small slot-shaped valley in its floor known as the Challenger Deep.[1] The deepest point of the trench is more than 2\xa0km (1.2\xa0mi) farther from sea level than the peak of Mount Everest.[a]. At the bottom of the trench, the water column above exerts a pressure of 1,086\xa0bar (15,750\xa0psi), more than 1,071 times the standard atmospheric pressure at sea level. At this pressure, the density of water is increased by 4.96%.
Thought: I need to search the Puerto Rico Trench.
Action: Search[Puerto Rico Trench]
Observation: The Puerto Rico Trench is located on the boundary between the Caribbean Sea and the Atlantic Ocean. The oceanic trench, the deepest in the Atlantic, is associated with a complex transition between the Lesser Antilles subduction zone to the south and the major transform fault zone or plate boundary, which extends west between Cuba and Hispaniola through the Cayman Trough to the coast of Central America.. The trench is 800 kilometres (497\xa0mi) long[1] and has a maximum depth of 8,376 metres (27,480\xa0ft)[2] or 5.20 miles. This constitutes the single deepest point in the Atlantic Ocean. This point is commonly referred to as the Milwaukee Deep, with the Brownson Deep naming the seabed surrounding it.[3] However, more recently, the latter term has also been used interchangeably with the former to refer to this point.[4][5][6] The exact point was identified by the DSSV Pressure Drop using a state-of-the-art Kongsberg EM124 multibeam sonar in 2018, and then directly visited and its depth verified by the crewed submersible Deep-Submergence Vehicle DSV Limiting Factor (a Triton 36000/2 model submersible) piloted by Victor Vescovo.[7][8][9].
Thought: I need to calculate the depth difference between the Mariana Trench and the Puerto Rico Trench in meters.
Action: Calculate[10984-8376]
Observation: 2608
Thought: I need to search the South Sandwich Trench.
Action: Search[South Sandwich Trench]
Observation: 55°25′44″S 26°11′29″W\ufeff / \ufeff55.42889°S 26.19139°W\ufeff / -55.42889; -26.19139. The South Sandwich Trench is a deep arcuate trench in the South Atlantic Ocean lying 100 kilometres (62\xa0mi) to the east of the South Sandwich Islands. It is the deepest trench of the Southern Atlantic Ocean, and the second deepest of the Atlantic Ocean after the Puerto Rico Trench. Since the trench extends south of the 60th parallel south, it also contains the deepest point in the Southern Ocean.. The deepest point in the entire trench is the Meteor Deep, whose location prior to February 2019 was identified as 55°25.12′S 26°24.28′W\ufeff / \ufeff55.41867°S 26.40467°W\ufeff / -55.41867; -26.40467\ufeff (Meteor Deep) at a depth of 8,202 metres (26,909\xa0ft).
Thought: I need to search the Sunda Trench.
Action: Search[Sunda Trench]
Observation: The Sunda Trench, earlier known as and sometimes still indicated as the Java Trench,[1] is an oceanic trench located in the Indian Ocean near Sumatra, formed where the Australian-Capricorn plates subduct under a part of the Eurasian Plate. It is 3,200 kilometres (2,000\xa0mi) long with a maximum depth of 7,290 metres (23,920 feet).[2] Its maximum depth is the deepest point in the Indian Ocean. The trench stretches from the Lesser Sunda Islands past Java, around the southern coast of Sumatra on to the Andaman Islands, and forms the boundary between Indo-Australian Plate and Eurasian plate (more specifically, Sunda Plate). The trench is considered to be part of the Pacific Ring of Fire as well as one of a ring of oceanic trenches around the northern edges of the Australian Plate.. In 2005, scientists found evidence that the 2004 earthquake activity in the area of the Java Trench could lead to further catastrophic shifting within a relatively short period of time, perhaps less than a decade.[3] This threat has resulted in international agreements to establish a tsunami warning system in place along the Indian Ocean coast.[4].
Thought: I need to calculate the depth difference between the South Sandwich Trench and the Sunda Trench in meters.
Action: Calculate[8202-7290]
Observation: 912
Thought: I need to calculate the minimum of 2608 and 912.
Action: Calculate[min(2608, 912)]
Observation: 912
Thought: The smaller value is 912.
Action: Finish[912]""",
]

SUFFIX = """\nQuestion: {input}
{agent_scratchpad}"""


PROMPT = PromptTemplate.from_examples(
    EXAMPLES, SUFFIX, ["input", "agent_scratchpad"], prefix=PREFIX
)
