from langchain.prompts.prompt import PromptTemplate

PREFIX = (
    " Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation. After Thought, you MUST always take an Action. Action can be two types:\n"
    " (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n"
    " - Sometines, Search can fail and return irrelevant information. If this happens, just ignore it and continue.\n"
    " (2) Finish[answer], which returns the answer and finishes the task. You MUST use Finish to finish the task.\n"
    "Here are some examples\n"
    "Guidelines:\n"
    " - You MUST always output with Finish[answer] even if you are not sure about the answer.\n"
    # NOTE: uncomment these ReAct-specific prompts for better ReAct accuracy
    # " - You MUST search for information about all movies even if the first 4 are dissimilar.\n"
    # " - You MUST find all movies to make the most informed decision including the movies in the options. If you do not find all movies and make early decision, you will be PENALIZED.\n"
    # " - NEVER make a decision before finding all movies (both the ones in the Question and the Options). NEVER use the Finish operation before you search all of those movies. See the example below.\n"
    " - The Search result may not be available (in case search failed) or not relevant to movie information. If this happens, just ignore it and rely on your own knowledge on that movie.\n"
    " - When making the final answer, you MUST use both your own knowledge and the information you found. Information is only there to help you make a better decision.\n"
    "Here are some examples:\n"
)

EXAMPLES = [
    """
    Question: Find a movie similar to Interstellar, 2001: A Space Odyssey, The Martian, Star Wars Episode IV - A New Hope\n
    Options:\n
    Saving Private Ryan\n
    Ender's Game\n
    Avatar\n
    Titanic\n
    Thought: I need to search for information about each movie to see what they're about, starting with Interstellar.\n
    Action: Search[Interstellar]\n
    Observation: Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan. It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine. Set in a dystopian future where humanity is embroiled in a catastrophic blight and famine, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for humankind.\n
    Thought: I need to search for information about 2001: A Space Odyssey next.\n
    Action: Search[2001: A Space Odyssey]\n
    Observation: 2001: A Space Odyssey is a 1968 epic science fiction film produced and directed by Stanley Kubrick. The screenplay was written by Kubrick and science fiction author Arthur C. Clarke, and was inspired by Clarke's 1951 short story "The Sentinel" and other short stories by Clarke. Clarke also published a novelisation of the film, in part written concurrently with the screenplay, after the film's release. The film stars Keir Dullea, Gary Lockwood, William Sylvester, and Douglas Rain and follows a voyage by astronauts, scientists, and the sentient supercomputer HAL to Jupiter to investigate an alien monolith.\n
    Thought: I need to search for information about The Martian next.\n
    Action: Search[The Martian]\n
    Observation: The Martian is a 2015 science fiction film directed by Ridley Scott and starring Matt Damon. Drew Goddard adapted the screenplay from the 2011 novel The Martian by Andy Weir. The film depicts an astronaut's lone struggle to survive on Mars after being left behind and the efforts of NASA to rescue him and bring him home to Earth. It also stars Jessica Chastain, Jeff Daniels, Kristen Wiig, Chiwetel Ejiofor, Sean Bean, Michael Peña, Kate Mara, Sebastian Stan, Aksel Hennie, Mackenzie Davis, Donald Glover, and Benedict Wong.\n
    Thought: I need to search for information about Star Wars Episode IV - A New Hope next.\n
    Action: Search[Star Wars Episode IV - A New Hope]\n
    Observation: Star Wars (retroactively retitled Star Wars: Episode IV – A New Hope) is a 1977 American epic space opera film written and directed by George Lucas, produced by Lucasfilm and distributed by 20th Century-Fox. It was the first film released in the Star Wars film series and the fourth chronological chapter of the "Skywalker Saga". Set "a long time ago" in a fictional universe where the galaxy is ruled by the tyrannical Galactic Empire, the story focuses on a group of freedom fighters known as the Rebel Alliance, who aim to destroy the Empire's newest weapon, the Death Star. When Rebel leader Princess Leia is apprehended by the Empire, Luke Skywalker acquires stolen architectural plans of the Death Star and sets out to rescue her while learning the ways of a metaphysical power known as "the Force" from Jedi Master Obi-Wan Kenobi. The cast includes Mark Hamill, Harrison Ford, Carrie Fisher, Peter Cushing, Alec Guinness, David Prowse, James Earl Jones, Anthony Daniels, Kenny Baker, and Peter Mayhew.\n
    Thought: I need to search for information about Saving Private Ryan next.\n
    Action: Search[Saving Private Ryan]\n
    Observation: Saving Private Ryan is a 1998 American epic war film directed by Steven Spielberg and written by Robert Rodat. Set in 1944 in France during World War II, it follows a group of soldiers, led by Captain John Miller (Tom Hanks), on their mission to locate Private James Francis Ryan (Matt Damon) and bring him home safely after his three brothers are killed in action. The cast also includes Edward Burns, Tom Sizemore, Barry Pepper, Giovanni Ribisi, Vin Diesel, Adam Goldberg, and Jeremy Davies.\n
    Thought: I need to search for information about Ender's Game next.\n
    Action: Search[Ender's Game]\n
    Observation: Ender's Game is a 1985 military science fiction novel by American author Orson Scott Card. Set at an unspecified date in Earth's future, the novel presents an imperiled humankind after two conflicts with an insectoid alien species they dub "the buggers". In preparation for an anticipated third invasion, Earth's international military force recruits young children, including the novel's protagonist, Andrew "Ender" Wiggin, to be trained as elite officers. The children learn military strategy and leadership by playing increasingly difficult war games, including some in zero gravity, where Ender's tactical genius is revealed.\n
    Thought: I need to search for information about Avatar next.\n
    Action: Search[Avatar]\n
    Observation: Avatar (marketed as James Cameron's Avatar) is a 2009 epic science fiction film directed, written, co-produced, and co-edited by James Cameron and starring Sam Worthington, Zoe Saldana, Stephen Lang, Michelle Rodriguez,[6] and Sigourney Weaver. It is the first installment in the Avatar film series. It is set in the mid-22nd century, when humans are colonizing Pandora, a lush habitable moon of a gas giant in the Alpha Centauri star system, in order to mine the valuable mineral unobtanium.[a] The expansion of the mining colony threatens the continued existence of a local tribe of Na'vi, a humanoid species indigenous to Pandora. The title of the film refers to a genetically engineered Na'vi body operated from the brain of a remotely located human that is used to interact with the natives of Pandora.[10]\n
    Thought: I need to search for information about Titanic next.\n
    Action: Search[Titanic]\n
    Observation: Titanic is a 1997 American disaster film directed, written, produced, and co-edited by James Cameron. Incorporating both historical and fictionalized aspects, it is based on accounts of the sinking of RMS Titanic in 1912. Kate Winslet and Leonardo DiCaprio star as members of different social classes who fall in love during the ship's maiden voyage. The film also features Billy Zane, Kathy Bates, Frances Fisher, Gloria Stuart, Bernard Hill, Jonathan Hyde, Victor Garber, David Warner, Suzy Amis and Bill Paxton.\n
    Thought: Interstellar, 2001: A Space Odyssey, The Martian, and Star Wars Episode IV - A New Hope are all space movies. The only space movie option is Ender's Game.\n
    Action: Finish[Ender's Game]""",
    """
    Question: Find a movie similar to Mission Impossible, The Silence of the Lambs, American Beauty, Star Wars Episode IV - A New Hope\n
    Options:\n
    Austin Powers International Man of Mystery\n
    Alesha Popvich and Tugarin the Dragon\n
    In Cold Blood\n
    Rosetta\n
    Thought: I need to search all movies in the Question, starting with Mission Impossible.\n
    Action: Search[Mission Impossible]\n
    Observation: Mission: Impossible is an American multimedia franchise based on a fictional secret espionage agency known as the Impossible Missions Force (IMF). The 1966 TV series ran for seven seasons and was revived in 1988 for two seasons. It inspired a series of theatrical motion pictures starring Tom Cruise beginning in 1996. By 2011, the franchise generated over $4 billion in revenue making Mission: Impossible one of the highest-grossing media franchises of all time.[1] As of 2023, the latest media released in the franchise was the action-spy film Mission: Impossible – Dead Reckoning Part One, which premiered in Rome on July 12, 2023.. Main pageContentsCurrent eventsRandom articleAbout WikipediaContact usDonate.\n
    Thought: I need to search for information about The Silence of the Lambs next.\n
    Action: Search[The Silence of the Lambs]\n
    Observation: Could not find [The Silence of the Lambs]. Similar: ['The Silence of the Lambs (film)', 'The Silence of the Lambs (novel)', 'Buffalo Bill (character) (redirect from Buffalo Bill (The Silence of the Lambs))', 'The Silence of the Lambs', 'Hannibal (2001 film) (category Films about the Federal Bureau of Investigation)]'\n
    Thought: I need to search for information about American Beauty next.\n
    Action: Search[American Beauty]\n
    Observation: Could not find [American Beauty]. Similar: ['American Beauty (1999 film)', 'American Beauty', 'American Beauty (album)', 'American Beauty/American Psycho', 'American Beauty (soundtrack)'].\n
    Thought: I need to search for information about Star Wars Episode IV - A New Hope next.\n
    Action: Search[Star Wars Episode IV - A New Hope]\n
    Observation: Star Wars (retroactively retitled Star Wars: Episode IV – A New Hope) is a 1977 American epic space opera film written and directed by George Lucas, produced by Lucasfilm and distributed by 20th Century-Fox. It was the first film released in the Star Wars film series and the fourth chronological chapter of the "Skywalker Saga". Set "a long time ago" in a fictional universe where the galaxy is ruled by the tyrannical Galactic Empire, the story focuses on a group of freedom fighters known as the Rebel Alliance, who aim to destroy the Empire's newest weapon, the Death Star. When Rebel leader Princess Leia is apprehended by the Empire, Luke Skywalker acquires stolen architectural plans of the Death Star and sets out to rescue her while learning the ways of a metaphysical power known as "the Force" from Jedi Master Obi-Wan Kenobi. The cast includes Mark Hamill, Harrison Ford, Carrie Fisher, Peter Cushing, Alec Guinness, David Prowse, James Earl Jones, Anthony Daniels, Kenny Baker, and Peter Mayhew..\n
    Thought: I found all movies in the Question. Then I need to find all movies in the Options, starting from Austin Powers International Man of Mystery next.\n
    Action: Search[Austin Powers International Man of Mystery]\n
    Observation: Austin Powers: International Man of Mystery (or simply Austin Powers[2]) is a 1997 American spy comedy film directed by Jay Roach. It is the first installment in the Austin Powers series. It stars franchise co-producer and writer Mike Myers, playing the roles of Austin Powers and Dr. Evil,[5][6] Powers' arch-enemy. Supporting roles are played by Elizabeth Hurley, Robert Wagner, Seth Green, and Michael York.\n
    Thought: I need to search for information about Alesha Popvich and Tugarin the Dragon next.\n
    Action: Search[Alesha Popvich and Tugarin the Dragon]\n
    Observation: Could not find Alesha Popovich and Tugarin the Dragon. Similar: ['Alyosha Popovich', 'The Three Bogatyrs', 'List of fantasy films of the 2000s'].\n
    Thought: I need to search for information about In Cold Blood next.\n
    Action: Search[In Cold Blood]\n
    Observation: In Cold Blood is a non-fiction novel[1] by American author Truman Capote, first published in 1966. It details the 1959 murders of four members of the Clutter family in the small farming community of Holcomb, Kansas.. Capote learned of the quadruple murder before the killers were captured, and he traveled to Kansas to write about the crime. He was accompanied by his childhood friend and fellow author Harper Lee, and they interviewed residents and investigators assigned to the case and took thousands of pages of notes. Killers Richard Hickock and Perry Smith were arrested six weeks after the murders and later executed by the state of Kansas.\n
    Thought: I need to search for information about Rosetta next.\n
    Action: Search[Rosetta]
    Observation: Rosetta (/roʊˈzɛtə/ roh-ZET-ə)[a] or Rashid (Arabic: رشيد, romanized: Rašīd, IPA: [ɾɑˈʃiːd]; Coptic: ϯⲣⲁϣⲓⲧ, romanized: ti-Rashit)[b] is a port city of the Nile Delta, 65 km (40 mi) east of Alexandria, in Egypt's Beheira governorate. The Rosetta Stone was discovered there in 1799.. Founded around the 9th century on site of the ancient town Bolbitine, Rosetta boomed with the decline of Alexandria following the Ottoman conquest of Egypt in 1517, only to wane in importance after Alexandria's revival. During the 19th century, it was a popular British tourist destination, known for its Ottoman mansions, citrus groves and relative cleanliness.. The name of the town most likely comes from an Arabic name Rašīd (meaning "guide")[3] and was transcribed and corrupted in numerous ways – the name Rexi was used by the Crusaders in Middle Ages[4] and Rosetta or Rosette ("little rose" in Italian and French respectively) was used by the French at the time of Napoleon Bonaparte's campaign in Egypt.\n
    Thought: Based on my own knowledge and the provided observations, Austin Powers International Man of Mystery is most similar.\n
    Action: Finish[Austin Powers International Man of Mystery]\n""",
]

SUFFIX = """\n
Question: {input}
{agent_scratchpad}"""

# We use one example as 1-shot
PROMPT = PromptTemplate.from_examples(
    EXAMPLES[1:], SUFFIX, ["input", "agent_scratchpad"], prefix=PREFIX
)
