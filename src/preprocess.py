"""
Data loading, JSON parsing, text cleaning, feature combination,
and synthetic dataset fallback generation for Movie Genre Classification.
"""

import os
import re
import ast
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd


# Default stop words fallback in case NLTK data is not pre-downloaded
DEFAULT_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves"
}


def get_stop_words() -> set:
    """Retrieve NLTK English stopwords or fall back to DEFAULT_STOP_WORDS."""
    try:
        import nltk
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
    except Exception:
        return DEFAULT_STOP_WORDS


STOP_WORDS = get_stop_words()


def safe_parse_json(val) -> list:
    """
    Safely parse JSON or Python literal string into a Python list of dicts.
    Handles malformed strings, None, floats, and existing list objects.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, dict):
        return [val]
    if not isinstance(val, str):
        return []

    val = val.strip()
    if not val or val == "[]":
        return []

    # Attempt ast.literal_eval first as specified
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
    except (ValueError, SyntaxError):
        pass

    # Fallback to json.loads
    try:
        parsed = json.loads(val.replace("'", '"'))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    return []


def extract_primary_genre(genres_raw) -> Optional[str]:
    """
    Parse the JSON-formatted genres column safely using ast.literal_eval
    and extract the primary genre (first listed genre).
    """
    parsed = safe_parse_json(genres_raw)
    for item in parsed:
        if isinstance(item, dict) and "name" in item and item["name"]:
            return str(item["name"]).strip()
    return None


def extract_keywords_text(keywords_raw) -> str:
    """
    Parse the JSON-formatted keywords column safely and combine
    all keyword names into a space-separated string.
    """
    parsed = safe_parse_json(keywords_raw)
    names = []
    for item in parsed:
        if isinstance(item, dict) and "name" in item and item["name"]:
            names.append(str(item["name"]).strip())
    return " ".join(names)


def clean_text(text: str) -> str:
    """
    Clean text:
    1. Lowercase conversion
    2. Punctuation and number removal (letters and whitespace only)
    3. Stopword filtering
    4. Normalization of whitespace
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation, symbols, and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize and filter stop words + short tokens
    tokens = text.split()
    filtered = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]

    return " ".join(filtered)


def generate_synthetic_data(data_dir: Optional[str] = None, save_files: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a rich, realistic synthetic movie metadata and keywords dataset
    covering the top 10 genres when Kaggle CSV files are not found.
    This guarantees zero-configuration, immediate out-of-the-box execution.
    """
    records = [
        # Action
        {"id": 101, "title": "The Dark Knight", "genre": "Action",
         "overview": "Batman raises the stakes in his war on crime. With the help of allies Lt. Jim Gordon and DA Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets. The partnership proves to be effective, but they soon find themselves prey to a reign of chaos unleashed by a rising criminal mastermind known to the terrified citizens of Gotham as the Joker.",
         "keywords": ["superhero", "vigilante", "joker", "gotham city", "crime fighter", "explosions"]},
        {"id": 102, "title": "Mad Max: Fury Road", "genre": "Action",
         "overview": "An apocalyptic story set in the furthest reaches of our planet, in a stark desert landscape where humanity is broken, and most everyone is crazed fighting for the necessities of life. Within this world exist two rebels on the run who just might be able to restore order.",
         "keywords": ["post-apocalyptic", "car chase", "desert wasteland", "survival", "warlord", "action hero"]},
        {"id": 103, "title": "Die Hard", "genre": "Action",
         "overview": "NYPD cop John McClane's plan to reconcile with his estranged wife is thrown for a serious loop when, minutes after he arrives at her office, the whole building is overtaken by a group of terrorists. With little help from the LAPD, McClane takes matters into his own hands.",
         "keywords": ["hostage", "skyscraper", "terrorist", "one-man army", "nypd officer", "shootout"]},
        {"id": 104, "title": "John Wick", "genre": "Action",
         "overview": "An ex-hit-man comes out of retirement to track down the gangsters that took everything from him. With New York City as his bullet-riddled playground, Wick embarks on a merciless rampage of revenge.",
         "keywords": ["assassin", "revenge", "martial arts", "gun fu", "underworld", "bounty hunter"]},
        {"id": 105, "title": "Terminator 2: Judgment Day", "genre": "Action",
         "overview": "A cyborg, identical to the one who failed to kill Sarah Connor, must now protect her ten-year-old son John from an even more advanced and powerful cyborg made of liquid metal sent back through time.",
         "keywords": ["cyborg", "time travel", "future war", "protect child", "liquid metal", "apocalypse"]},
        {"id": 106, "title": "Gladiator", "genre": "Action",
         "overview": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery. In the arena of the Colosseum, he fights for honor and the soul of Rome.",
         "keywords": ["ancient rome", "gladiator", "revenge", "colosseum", "emperor", "sword battle"]},
        {"id": 107, "title": "Top Gun: Maverick", "genre": "Action",
         "overview": "After thirty years, Maverick is still pushing the envelope as a top naval aviator, but must confront ghosts of his past when he leads TOP GUN's elite graduates on a mission that demands the ultimate sacrifice from those chosen to fly it.",
         "keywords": ["fighter jet", "pilot", "navy", "aerial dogfight", "high speed", "military mission"]},
        {"id": 108, "title": "Raiders of the Lost Ark", "genre": "Action",
         "overview": "Archaeology professor Indiana Jones ventures across the globe to retrieve the legendary Ark of the Covenant before insidious enemy agents harness its supernatural power for world domination.",
         "keywords": ["archaeologist", "lost artifact", "booby traps", "adventure hero", "globe-trotting", "whip"]},
        {"id": 109, "title": "The Matrix", "genre": "Action",
         "overview": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers. He masters bullet time and martial arts to liberate humanity.",
         "keywords": ["cyberpunk", "virtual reality", "bullet time", "martial arts", "chosen one", "dystopia"]},
        {"id": 110, "title": "Avengers: Endgame", "genre": "Action",
         "overview": "After the devastating events of Infinity War, the universe is in ruins. With the help of remaining allies, the Avengers assemble once more in order to reverse Thanos' actions and restore balance to the universe.",
         "keywords": ["superheroes", "avengers", "marvel cinematic", "time heist", "epic showdown", "infinity stones"]},

        # Comedy
        {"id": 201, "title": "Superbad", "genre": "Comedy",
         "overview": "Two co-dependent high school seniors are forced to deal with separation anxiety after their plan to stage a booze-soaked party goes awry. Hilarious mishaps with fake IDs, eccentric cops, and teenage romance unfold.",
         "keywords": ["high school", "teen comedy", "fake id", "wild party", "friendship", "coming of age"]},
        {"id": 202, "title": "The Hangover", "genre": "Comedy",
         "overview": "Three buddies wake up from a bachelor party in Las Vegas with no memory of the previous night and the bachelor missing. They make their way around the city in order to find their friend before his wedding.",
         "keywords": ["bachelor party", "las vegas", "amnesia", "missing person", "tiger", "wedding morning"]},
        {"id": 203, "title": "Groundhog Day", "genre": "Comedy",
         "overview": "A cynical TV weatherman finds himself reliving the same day over and over again when he goes on location to the small town of Punxsutawney to film the annual Groundhog Day festival.",
         "keywords": ["time loop", "weatherman", "groundhog", "romantic comedy", "second chances", "sarcasm"]},
        {"id": 204, "title": "Step Brothers", "genre": "Comedy",
         "overview": "Two aimless middle-aged men are forced to live together as roommates after their single parents marry. Sibling rivalry escalates into chaotic pranks, treehouse construction, and musical hilarity.",
         "keywords": ["man-child", "step brothers", "slapstick", "family dysfunction", "living at home", "pranks"]},
        {"id": 205, "title": "Ferris Bueller's Day Off", "genre": "Comedy",
         "overview": "A high school slacker pretends to be sick in order to skip school and have an unforgettable day in Chicago with his best friend and girlfriend, all while evading his determined high school principal.",
         "keywords": ["skipping school", "chicago", "ferrari", "breaking the fourth wall", "teen rebellion", "parade"]},
        {"id": 206, "title": "Anchorman: The Legend of Ron Burgundy", "genre": "Comedy",
         "overview": "Ron Burgundy is San Diego's top-rated newsman in the male-dominated 1970s, but that's all about to change when a new ambitious female anchor arrives to disrupt his quirky broadcast news team.",
         "keywords": ["news anchor", "1970s", "broadcast television", "improv comedy", "mustache", "rivalry"]},
        {"id": 207, "title": "Shaun of the Dead", "genre": "Comedy",
         "overview": "An unmotivated London electronics salesman and his slacker roommate must rescue their friends and loved ones when a sudden zombie apocalypse sweeps through their sleepy suburban neighborhood.",
         "keywords": ["zombie comedy", "london pub", "slacker", "vinyl records", "apocalypse satire", "friendship"]},
        {"id": 208, "title": "Airplane!", "genre": "Comedy",
         "overview": "A former fighter pilot traumatized by the war must take the controls of an airliner after the entire flight crew falls violently ill from poisoned fish. Packed with rapid-fire visual gags and puns.",
         "keywords": ["parody", "airplane catastrophe", "slapstick", "absurdist humor", "deadpan", "cockpit"]},
        {"id": 209, "title": "Bridesmaids", "genre": "Comedy",
         "overview": "Competition between the maid of honor and a wealthy bridesmaid, over who is the bride's best friend, threatens to upend the life of an out-of-work pastry chef throughout wild bachelorette festivities.",
         "keywords": ["wedding party", "maid of honor", "female friendship", "bachelorette party", "jealousy", "food poisoning"]},
        {"id": 210, "title": "Monty Python and the Holy Grail", "genre": "Comedy",
         "overview": "King Arthur and his Knights of the Round Table embark on a surreal, low-budget search for the Holy Grail, encountering absurd obstacles including silly French soldiers, a killer rabbit, and the Black Knight.",
         "keywords": ["knights of the round table", "medieval parody", "holy grail", "satire", "surreal humour", "coconuts"]},

        # Drama
        {"id": 301, "title": "The Shawshank Redemption", "genre": "Drama",
         "overview": "A banker convicted of uxoricide forms a friendship over a quarter of a century with a hardened convict, while maintaining his innocence and trying to remain hopeful through quiet perseverance inside Shawshank prison.",
         "keywords": ["prison life", "wrongful conviction", "escape", "enduring friendship", "hope", "corrupt warden"]},
        {"id": 302, "title": "The Godfather", "genre": "Drama",
         "overview": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant youngest son, pulling him inexorably into family tragedy and moral compromise.",
         "keywords": ["mafia family", "patriarch", "organized crime", "succession", "loyalty", "tragedy"]},
        {"id": 303, "title": "Schindler's List", "genre": "Drama",
         "overview": "In German-occupied Poland during World War II, industrialist Oskar Schindler gradually becomes concerned for his Jewish workforce after witnessing their persecution by the Nazi regime, bankrupting himself to save hundreds of lives.",
         "keywords": ["holocaust", "world war ii", "humanitarian hero", "concentration camp", "moral awakening", "historical drama"]},
        {"id": 304, "title": "Forrest Gump", "genre": "Drama",
         "overview": "The history of the United States from the 1950s to the '70s unfolds from the perspective of an Alabama man with an IQ of 75, who yearns to be reunited with his childhood sweetheart.",
         "keywords": ["running", "vietnam war", "ping pong", "destiny", "innocence", "american history"]},
        {"id": 305, "title": "Fight Club", "genre": "Drama",
         "overview": "An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into something much more dangerous, exploring societal alienation, consumerism, and fractured identity.",
         "keywords": ["insomnia", "alter ego", "underground fight club", "anti-consumerism", "nihilism", "anarchy"]},
        {"id": 306, "title": "12 Angry Men", "genre": "Drama",
         "overview": "The jury in a New York City murder trial is frustrated by a single member whose skeptical caution forces them to more carefully consider the evidence before jumping to a hasty guilty verdict.",
         "keywords": ["jury room", "reasonable doubt", "justice system", "intense dialogue", "deliberation", "prejudice"]},
        {"id": 307, "title": "Good Will Hunting", "genre": "Drama",
         "overview": "Will Hunting, a janitor at MIT, has a gift for mathematics, but needs help from a compassionate psychologist to find direction in his life and confront deep-seated emotional trauma from his youth.",
         "keywords": ["math genius", "psychotherapy", "south boston", "emotional healing", "mentorship", "romance"]},
        {"id": 308, "title": "Whiplash", "genre": "Drama",
         "overview": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored and ruthlessly pushed to the brink by an abusive jazz instructor who will stop at nothing to realize a student's potential.",
         "keywords": ["jazz drumming", "music conservatory", "abusive teacher", "obsession", "perfectionism", "ambition"]},
        {"id": 309, "title": "The Social Network", "genre": "Drama",
         "overview": "As Harvard student Mark Zuckerberg creates the social networking website that would become known as Facebook, he is sued by the twins who claimed he stole their idea, and by the co-founder who was squeezed out.",
         "keywords": ["facebook origin", "harvard student", "intellectual property", "betrayal", "venture capital", "silicon valley"]},
        {"id": 310, "title": "A Beautiful Mind", "genre": "Drama",
         "overview": "After John Nash, a brilliant but asocial mathematician, accepts secret work in cryptography, his life takes a nightmarish turn as he battles paranoid schizophrenia with the unwavering support of his devoted wife.",
         "keywords": ["nobel prize", "mathematician", "paranoid schizophrenia", "cryptography", "mental illness", "devotion"]},

        # Horror
        {"id": 401, "title": "The Conjuring", "genre": "Horror",
         "overview": "Paranormal investigators Ed and Lorraine Warren work to help a family terrorized by a dark presence in their secluded farmhouse. Demonic occurrences and escalating hauntings push the couple to their absolute limits.",
         "keywords": ["demonology", "haunted house", "paranormal investigation", "possession", "exorcism", "evil spirits"]},
        {"id": 402, "title": "The Shining", "genre": "Horror",
         "overview": "A family heads to an isolated hotel for the winter where a sinister presence influences the father into violent madness, while his psychic son sees horrific forebodings from both past and future.",
         "keywords": ["isolated hotel", "cabin fever", "psychic powers", "haunted maze", "axe murderer", "evil spirits"]},
        {"id": 403, "title": "A Nightmare on Elm Street", "genre": "Horror",
         "overview": "Monstrous spirit Freddy Krueger invades the dreams of teenagers to murder them in reality. A group of suburban friends must find a way to stay awake to survive the nightmare entity seeking retribution.",
         "keywords": ["dream monster", "razor glove", "slasher", "sleep deprivation", "vengeful spirit", "nightmare"]},
        {"id": 404, "title": "Halloween", "genre": "Horror",
         "overview": "Fifteen years after murdering his sister on Halloween night 1963, Michael Myers escapes from a mental hospital and returns to the small town of Haddonfield, Illinois to stalk high school student Laurie Strode.",
         "keywords": ["slasher", "escaped lunatic", "halloween night", "masked killer", "final girl", "babysitter"]},
        {"id": 405, "title": "Hereditary", "genre": "Horror",
         "overview": "A grieving family is haunted by tragic and disturbing occurrences after the death of their secretive grandmother. An ancient occult demon and ancestral curses tear their home apart in psychological terror.",
         "keywords": ["occult cult", "grief", "ancestral curse", "possession", "macabre decapitation", "demonic king"]},
        {"id": 406, "title": "Get Out", "genre": "Horror",
         "overview": "A young African-American visits his white girlfriend's parents for the weekend, where his simmering uneasiness about their reception of him eventually reaches a boil of terrifying psychological abduction.",
         "keywords": ["sunken place", "hypnosis", "racial dread", "disturbing family", "abduction", "psychological horror"]},
        {"id": 407, "title": "The Exorcist", "genre": "Horror",
         "overview": "When a 12-year-old girl is possessed by a mysterious demonic entity, her desperate mother seeks the help of two Catholic priests to save her through an intense, grueling ritual of spiritual warfare.",
         "keywords": ["demonic possession", "exorcist priest", "levitation", "holy water", "ancient evil", "pazuzu"]},
        {"id": 408, "title": "It", "genre": "Horror",
         "overview": "In the summer of 1989, a group of bullied kids band together to destroy a shape-shifting monster that disguises itself as a clown and preys on the children of Derry, Maine by manifesting their deepest fears.",
         "keywords": ["evil clown", "shape shifter", "childhood fears", "sewers", "small town mystery", "balloon"]},
        {"id": 409, "title": "The Babadook", "genre": "Horror",
         "overview": "A widowed mother, plagued by the violent death of her husband, battles with her young son's fear of a monster lurking in the house, but soon discovers a sinister presence emerging from a pop-up children's storybook.",
         "keywords": ["storybook monster", "widowed mother", "haunted house", "grief trauma", "psychological breakdown", "dark closet"]},
        {"id": 410, "title": "A Quiet Place", "genre": "Horror",
         "overview": "In a post-apocalyptic world, a family is forced to live in absolute silence while hiding from blind monsters with ultra-sensitive hearing that hunt and eradicate any creature making the slightest sound.",
         "keywords": ["blind monsters", "silence", "survival", "sign language", "sound sensory", "farmhouse trap"]},

        # Science Fiction
        {"id": 501, "title": "Interstellar", "genre": "Science Fiction",
         "overview": "When Earth becomes uninhabitable in the future, a farmer and ex-NASA pilot, Joseph Cooper, is tasked to pilot a spacecraft, along with a team of researchers, to find a new planet for humans through a mysterious wormhole near Saturn.",
         "keywords": ["black hole", "wormhole", "space travel", "gravitational anomaly", "time dilation", "future earth", "tesseract"]},
        {"id": 502, "title": "Blade Runner 2049", "genre": "Science Fiction",
         "overview": "Officer K, a new blade runner for the Los Angeles Police Department, unearths a long-buried secret that has the potential to plunge what's left of society into chaos. His discovery leads him on a quest to find Rick Deckard.",
         "keywords": ["replicant", "cyberpunk", "artificial intelligence", "hologram companion", "dystopian futuristic los angeles", "existential dread"]},
        {"id": 503, "title": "Inception", "genre": "Science Fiction",
         "overview": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O., but his tragic past may doom the project and his team to disaster.",
         "keywords": ["dream within a dream", "subconscious heist", "totem", "limbo state", "zero gravity fight", "corporate espionage"]},
        {"id": 504, "title": "Arrival", "genre": "Science Fiction",
         "overview": "A linguist is recruited by the military to assist in translating extraterrestrial communications after twelve mysterious alien spacecraft touch down across the globe, revealing a non-linear perception of time.",
         "keywords": ["alien encounter", "linguistics", "heptapod", "non-linear time", "first contact", "spaceship"]},
        {"id": 505, "title": "Dune: Part One", "genre": "Science Fiction",
         "overview": "Paul Atreides, a brilliant and gifted young man born into a great destiny beyond his understanding, must travel to the most dangerous desert planet in the universe to ensure the future of his family and his people.",
         "keywords": ["desert planet", "spice melange", "giant sandworm", "intergalactic empire", "prophecy", "bene gesserit"]},
        {"id": 506, "title": "The Martian", "genre": "Science Fiction",
         "overview": "An astronaut becomes stranded on Mars after his team assume him dead, and must rely on his botanical ingenuity and engineering skills to find a way to signal to Earth that he is alive and survive until rescue.",
         "keywords": ["stranded on mars", "botanist", "nasa rescue mission", "red planet", "survival science", "space habitat"]},
        {"id": 507, "title": "Ex Machina", "genre": "Science Fiction",
         "overview": "A young programmer is selected to participate in a ground-breaking experiment in synthetic intelligence by evaluating the human qualities of a highly advanced female humanoid robot equipped with artificial consciousness.",
         "keywords": ["artificial intelligence", "turing test", "humanoid robot", "isolated laboratory", "manipulation", "tech CEO"]},
        {"id": 508, "title": "2001: A Space Odyssey", "genre": "Science Fiction",
         "overview": "After uncovering a mysterious artifact buried beneath the lunar surface, a spacecraft is sent to Jupiter to find its origins, manned by two astronauts and the supercomputer H.A.L. 9000.",
         "keywords": ["monolith", "space travel", "sentient computer hal 9000", "jupiter mission", "cosmic evolution", "stargate"]},
        {"id": 509, "title": "Alien", "genre": "Science Fiction",
         "overview": "The crew of a commercial spacecraft encounter a deadly extraterrestrial lifeform after investigating an unknown transmission from an uncharted planetoid. The creature stalks and picks off the crew one by one.",
         "keywords": ["xenomorph", "commercial spaceship", "facehugger", "outer space isolation", "survival horror", "android betrayal"]},
        {"id": 510, "title": "Back to the Future", "genre": "Science Fiction",
         "overview": "Marty McFly, a 17-year-old high school student, is accidentally sent thirty years into the past in a time-traveling DeLorean invented by his close friend, the eccentric scientist Doc Brown.",
         "keywords": ["time travel", "delorean", "flux capacitor", "1950s rock and roll", "mad scientist", "temporal paradox"]},

        # Romance
        {"id": 601, "title": "La La Land", "genre": "Romance",
         "overview": "While navigating their careers in Los Angeles, a pianist and an aspiring actress fall in love while attempting to reconcile their aspirations for the future with their passionate devotion to one another.",
         "keywords": ["jazz musician", "aspiring actress", "los angeles romance", "musical love story", "bittersweet dreams", "tap dance"]},
        {"id": 602, "title": "Titanic", "genre": "Romance",
         "overview": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic during its fateful maiden voyage across the North Atlantic Ocean.",
         "keywords": ["shipwreck", "forbidden love", "class difference", "ocean liner", "heart of the ocean", "tragic romance"]},
        {"id": 603, "title": "The Notebook", "genre": "Romance",
         "overview": "An elderly man reads to a fellow nursing home resident with dementia the story of two young lovers in 1940s South Carolina who are separated by social differences and the onset of war.",
         "keywords": ["summer romance", "class difference", "love letters", "nursing home", "dementia memory", "southern love"]},
        {"id": 604, "title": "Pride & Prejudice", "genre": "Romance",
         "overview": "Sparks fly when spirited Elizabeth Bennet meets single, rich, and proud Mr. Darcy. But Mr. Darcy reluctantly finds himself falling in love with a woman beneath his class.",
         "keywords": ["regency era", "mr darcy", "elizabeth bennet", "english countryside", "misunderstandings", "true love"]},
        {"id": 605, "title": "Before Sunrise", "genre": "Romance",
         "overview": "A young man and an adventurous French woman meet on a train in Europe and end up spending one unforgettable, conversation-filled evening together wandering through the charming streets of Vienna.",
         "keywords": ["vienna streets", "chance meeting", "deep conversation", "one night romance", "train traveler", "soulmates"]},
        {"id": 606, "title": "Eternal Sunshine of the Spotless Mind", "genre": "Romance",
         "overview": "When their relationship turns sour, a couple undergoes a high-tech medical procedure to have each other completely erased from their memories, only to rediscover their deep romantic connection.",
         "keywords": ["memory erasure", "heartbreak", "subconscious romance", "second chance", "eccentric love", "montauk beach"]},
        {"id": 607, "title": "Crazy Rich Asians", "genre": "Romance",
         "overview": "This contemporary romantic comedy follows native New Yorker Rachel Chu as she accompanies her longtime boyfriend to his best friend's lavish wedding in Singapore, discovering he is the heir of an ultra-wealthy dynasty.",
         "keywords": ["singapore elite", "high society wedding", "mother-in-law conflict", "wealthy boyfriend", "cultural clash", "rom-com"]},
        {"id": 608, "title": "About Time", "genre": "Romance",
         "overview": "At the age of 21, Tim discovers he can travel in time and change what happens in his own life. His decision to make his world a better place by getting a girlfriend turns out not to be as simple as you think.",
         "keywords": ["time travel romance", "london courtship", "family warmth", "soulmate devotion", "cherishing every day", "sweet comedy"]},
        {"id": 609, "title": "Me Before You", "genre": "Romance",
         "overview": "A cheerful young woman in a small English town forms an unexpected bond with the wealthy, paralyzed man she takes a job caring for, showing him that life is still worth living through laughter and affection.",
         "keywords": ["caregiver love", "paralyzed gentleman", "english castle town", "heartfelt devotion", "tearjerker", "romantic date"]},
        {"id": 610, "title": "500 Days of Summer", "genre": "Romance",
         "overview": "An offbeat romantic comedy about a greeting card writer who falls head over heels for a free-spirited woman who doesn't believe true love exists, tracing the 500 days of their relationship in non-linear order.",
         "keywords": ["greeting card writer", "unrequited love", "expectations vs reality", "indie romance", "breakup recovery", "summer finn"]},

        # Animation
        {"id": 701, "title": "Toy Story", "genre": "Animation",
         "overview": "A cowboy doll is profoundly threatened and jealous when a new spaceman action figure supplants him as top toy in a boy's bedroom, leading to an adventure across suburban perils to reunite with their owner.",
         "keywords": ["toys come to life", "cowboy woody", "space ranger buzz lightyear", "childhood bedroom", "pixar animation", "friendship"]},
        {"id": 702, "title": "Spirited Away", "genre": "Animation",
         "overview": "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits, and where humans are changed into beasts, working in a magical bathhouse to save her parents.",
         "keywords": ["studio ghibli", "spirit bathhouse", "witch yubaba", "haku dragon", "magical creatures", "japanese folklore"]},
        {"id": 703, "title": "Spider-Man: Into the Spider-Verse", "genre": "Animation",
         "overview": "Teen Miles Morales becomes the new Spider-Man of his universe, and must join with five spider-powered individuals from other dimensions to stop a threat to all reality orchestrated by the Kingpin.",
         "keywords": ["multiverse", "spider-man", "comic book art style", "brooklyn teenager", "alternate dimensions", "leap of faith"]},
        {"id": 704, "title": "Finding Nemo", "genre": "Animation",
         "overview": "After his son is captured in the Great Barrier Reef and taken to Sydney, a timid clownfish sets out on a journey across the open ocean to bring him home, aided by a forgetful blue tang fish.",
         "keywords": ["ocean journey", "clownfish father", "great barrier reef", "sydney harbor dentist", "shark meeting", "pixar classic"]},
        {"id": 705, "title": "The Lion King", "genre": "Animation",
         "overview": "Lion prince Simba and his father are targeted by his bitter uncle Scar, who wants to ascend the throne himself. Driven into exile, Simba grows up with eccentric friends before returning to reclaim his kingdom.",
         "keywords": ["african savanna", "lion prince", "circle of life", "evil uncle scar", "hakuna matata", "pride rock"]},
        {"id": 706, "title": "Shrek", "genre": "Animation",
         "overview": "A mean lord exiles fairytale creatures to the swamp of a grumpy ogre, who must go on a quest and rescue a feisty princess for the lord in order to get his land back, accompanied by an annoying talking donkey.",
         "keywords": ["green ogre", "talking donkey", "fairytale parody", "princess fiona", "dragon tower", "swamp home"]},
        {"id": 707, "title": "Up", "genre": "Animation",
         "overview": "78-year-old Carl Fredricksen travels to Paradise Falls in his house equipped with thousands of helium balloons, inadvertently taking a young Wilderness Explorer stowaway along on the South American adventure.",
         "keywords": ["balloon house", "wilderness explorer", "paradise falls", "talking golden retriever", "elderly adventurer", "pixar classic"]},
        {"id": 708, "title": "Coco", "genre": "Animation",
         "overview": "Aspiring musician Miguel, confronted with his family's ancestral ban on music, enters the vibrant Land of the Dead to find his great-great-grandfather, a legendary singer, during Dia de los Muertos.",
         "keywords": ["land of the dead", "day of the dead", "mexican culture", "acoustic guitar", "family ancestry", "alebreje spirits"]},
        {"id": 709, "title": "Wall-E", "genre": "Animation",
         "overview": "In the distant, abandoned future, a small waste-collecting robot inadvertently embarks on a space journey that will ultimately decide the fate of mankind after meeting a sleek search probe named EVE.",
         "keywords": ["waste robot", "abandoned earth", "space probe eve", "clean energy", "silent romance", "human spaceship"]},
        {"id": 710, "title": "Ratatouille", "genre": "Animation",
         "overview": "A rat who can cook makes an unusual alliance with a clumsy young kitchen worker at a famous Parisian gourmet restaurant, steering his culinary creations to the astonishment of France's top food critic.",
         "keywords": ["paris restaurant", "culinary chef", "talking rat remy", "gourmet soup", "food critic ego", "kitchen comedy"]},

        # Thriller
        {"id": 801, "title": "The Silence of the Lambs", "genre": "Thriller",
         "overview": "A young FBI cadet must receive the help of an incarcerated and manipulative cannibal killer, Dr. Hannibal Lecter, to help catch another serial killer who skins his victims across the Midwest.",
         "keywords": ["fbi profiler", "serial killer buffalo bill", "hannibal lecter", "psychological mind games", "moth cocoon", "investigation"]},
        {"id": 802, "title": "Se7en", "genre": "Thriller",
         "overview": "Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his terrifying motives in a perpetually rain-drenched, morally corrupt metropolis.",
         "keywords": ["seven deadly sins", "detective partners", "gruesome crime scene", "box ending", "rainy city", "serial murderer"]},
        {"id": 803, "title": "Shutter Island", "genre": "Thriller",
         "overview": "In 1954, a U.S. Marshal investigates the disappearance of a murderess who escaped from a hospital for the criminally insane on a desolate, storm-swept Boston Harbor island, uncovering dark conspiracies.",
         "keywords": ["asylum for criminally insane", "us marshal", "lighthouse secret", "hallucinations", "paranoia twist", "stormy island"]},
        {"id": 804, "title": "Gone Girl", "genre": "Thriller",
         "overview": "With his wife's disappearance having become the focus of an intense media circus, a man sees the spotlight turned on him when it's suspected that he may not be as innocent as he claims.",
         "keywords": ["missing wife", "media frenzy", "diabolical framing", "toxic marriage", "cool girl monologue", "deception"]},
        {"id": 805, "title": "Memento", "genre": "Thriller",
         "overview": "A man with short-term memory loss attempts to track down his wife's murderer using polaroid photos, tattoos etched onto his body, and handwritten notes to reconstruct what happened.",
         "keywords": ["anterograde amnesia", "polaroid photos", "tattoos memory", "reverse chronological order", "unreliable narrator", "revenge"]},
        {"id": 806, "title": "Prisoners", "genre": "Thriller",
         "overview": "When Keller Dover's daughter and her friend go missing, he takes matters into his own hands as the police pursue multiple leads and the pressure mounts, pushing him into ruthless interrogation.",
         "keywords": ["missing children", "vigilante father", "detective loki", "riddles and mazes", "morality descent", "abduction mystery"]},
        {"id": 807, "title": "Zodiac", "genre": "Thriller",
         "overview": "Between 1968 and 1983, a San Francisco cartoonist becomes an amateur detective obsessed with tracking down the infamous Zodiac Killer who terrorizes Northern California with cryptic coded ciphers.",
         "keywords": ["zodiac killer", "cryptic ciphers", "san francisco chronicle", "obsessive investigation", "true crime", "unsolved case"]},
        {"id": 808, "title": "Black Swan", "genre": "Thriller",
         "overview": "A committed dancer wins the lead role in a production of Tchaikovsky's Swan Lake only to find herself struggling to maintain her sanity as she faces fierce competition from a seductive new rival.",
         "keywords": ["ballet dancer", "swan lake", "psychological breakdown", "hallucinatory double", "perfectionism obsession", "rivalry"]},
        {"id": 809, "title": "Parasite", "genre": "Thriller",
         "overview": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan, culminating in a shocking, violent subterranean confrontation.",
         "keywords": ["class divide", "con artist family", "subterranean bunker", "rain storm flooding", "secret housekeeper", "black comedy thriller"]},
        {"id": 810, "title": "The Sixth Sense", "genre": "Thriller",
         "overview": "A child psychologist starts treating a troubled young boy who claims to see dead people walking among the living unaware of their demise, culminating in an extraordinary psychological revelation.",
         "keywords": ["i see dead people", "child psychologist", "ghost sightings", "supernatural mystery", "shocking twist ending", "unresolved spirits"]},

        # Crime
        {"id": 901, "title": "Pulp Fiction", "genre": "Crime",
         "overview": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption across the gritty underbelly of Los Angeles.",
         "keywords": ["mob hitmen", "briefcase mystery", "overdose adrenaline", "diner robbery", "nonlinear story", "quentin tarantino"]},
        {"id": 902, "title": "GoodFellas", "genre": "Crime",
         "overview": "The story of Henry Hill and his life in the mafia, covering his relationship with his wife Karen Hill and his mob partners Jimmy Conway and Tommy DeVito in the Italian-American crime syndicate.",
         "keywords": ["mafia wiseguy", "italian-american mob", "heist robbery", "fbi informant", "cocaine paranoia", "organized crime"]},
        {"id": 903, "title": "The Departed", "genre": "Crime",
         "overview": "An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in South Boston led by a ruthless crime boss, resulting in lethal cat-and-mouse tension.",
         "keywords": ["undercover cop", "boston mob mole", "irish mob boss", "wiretapping", "cat and mouse", "betrayal shootout"]},
        {"id": 904, "title": "Scarface", "genre": "Crime",
         "overview": "In 1980 Miami, a determined Cuban immigrant takes over a drug cartel and succumbs to greed, paranoia, and explosive violence as his empire expands beyond his control.",
         "keywords": ["tony montana", "cocaine kingpin", "miami 1980s", "cuban refugee", "say hello to my little friend", "mansion shootout"]},
        {"id": 905, "title": "Heat", "genre": "Crime",
         "overview": "A group of high-end professional bank robbers start to feel the heat from the LAPD when a loose-cannon partner leaves a clue at their latest heist, leading to a legendary standoff between cop and thief.",
         "keywords": ["bank heist", "lapd homicide detective", "professional thieves", "downtown shootout", "cat and mouse", "diner confrontation"]},
        {"id": 906, "title": "Casino", "genre": "Crime",
         "overview": "In early 1970s Las Vegas, two best friends - a mafia enforcer and a casino executive - compete and clash over a gambling empire and a fast-living, deceptive socialite.",
         "keywords": ["las vegas casino", "mob bosses", "gambling empire", "deception betrayal", "skimming cash", "desert burials"]},
        {"id": 907, "title": "Fargo", "genre": "Crime",
         "overview": "Minnesota car salesman Jerry Lundegaard's inept crime falls apart due to his and his henchmen's bungling and the persistent police work of the very pregnant police chief Marge Gunderson.",
         "keywords": ["kidnapping ransom", "wood chipper", "snowy minnesota", "pregnant police chief", "botched crime", "coen brothers"]},
        {"id": 908, "title": "The Usual Suspects", "genre": "Crime",
         "overview": "A sole survivor tells of the twisty events leading up to a horrific gun battle on a boat, which began when five criminals met at a seemingly random police lineup under the shadow of Keyser Soze.",
         "keywords": ["police lineup", "keyser soze", "customs interrogation", "boat massacre", "criminal syndicate", "shocking revelation"]},
        {"id": 909, "title": "Reservoir Dogs", "genre": "Crime",
         "overview": "When a simple jewelry heist goes horribly wrong, the surviving criminals begin to suspect that one of them is a police informant hiding among them in their abandoned warehouse rendezvous.",
         "keywords": ["botched diamond heist", "color codenames", "warehouse standoff", "undercover cop", "mexican standoff", "tarantino classic"]},
        {"id": 910, "title": "L.A. Confidential", "genre": "Crime",
         "overview": "As corruption grows in 1950s Los Angeles, three policemen - one strait-laced, one brutal, and one sleazy - investigate a series of murders with their own brand of justice in Hollywood's dark corners.",
         "keywords": ["1950s los angeles", "police corruption", "fleur-de-lis call girl", "hollywood scandals", "tabloid press", "hush hush"]},

        # Adventure
        {"id": 1001, "title": "The Lord of the Rings: The Fellowship of the Ring", "genre": "Adventure",
         "overview": "A meek Hobbit from the Shire and eight companions set out on a perilous journey to Mount Doom to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron.",
         "keywords": ["one ring", "middle-earth", "frodo hobbit", "gandalf wizard", "fellowship quest", "mordor darkness"]},
        {"id": 1002, "title": "The Lord of the Rings: The Return of the King", "genre": "Adventure",
         "overview": "Gandalf and Aragorn lead the World of Men against Sauron's army to draw his gaze from Frodo and Sam as they approach Mount Doom with the One Ring for the final reckoning.",
         "keywords": ["battle for minas tirith", "mount doom", "king aragorn", "epic fantasy quest", "middle-earth fate", "gollum betrayal"]},
        {"id": 1003, "title": "Pirates of the Caribbean: The Curse of the Black Pearl", "genre": "Adventure",
         "overview": "Blacksmith Will Turner teams up with eccentric pirate Captain Jack Sparrow to save his love, the governor's daughter, from Jack's former pirate allies who are cursed undead skeletons in moonlight.",
         "keywords": ["captain jack sparrow", "cursed aztec gold", "ghost pirate ship", "black pearl", "sword dueling", "caribbean island"]},
        {"id": 1004, "title": "Jurassic Park", "genre": "Adventure",
         "overview": "A pragmatic paleontologist touring an almost complete theme park on an island in Central America is tasked with protecting a couple of kids after a power failure causes the cloned dinosaurs to run loose.",
         "keywords": ["cloned dinosaurs", "tyrannosaurus rex", "velociraptor", "theme park disaster", "island survival", "amber dna"]},
        {"id": 1005, "title": "Life of Pi", "genre": "Adventure",
         "overview": "A young man who survives a disaster at sea is hurtled into an epic journey of adventure and discovery. While cast away, he forms an unexpected connection with another survivor: a fearsome Bengal tiger.",
         "keywords": ["shipwreck adrift", "bengal tiger richard parker", "lifeboat survival", "spiritual journey", "bioluminescent sea", "open ocean"]},
        {"id": 1006, "title": "Avatar", "genre": "Adventure",
         "overview": "A paraplegic Marine dispatched to the lush alien moon of Pandora on a unique mission becomes torn between following his orders and protecting the indigenous Na'vi tribe and their sacred homeland.",
         "keywords": ["pandora moon", "blue alien na'vi", "floating mountains", "bioluminescent jungle", "avatar body", "james cameron"]},
        {"id": 1007, "title": "The Revenant", "genre": "Adventure",
         "overview": "A frontiersman on a fur trading expedition in the 1820s fights for survival after being mauled by a bear and left for dead by members of his own hunting team, trekking hundreds of miles through brutal snow.",
         "keywords": ["grizzly bear attack", "wilderness survival", "winter wilderness", "fur trapper", "harsh nature", "revenge quest"]},
        {"id": 1008, "title": "Cast Away", "genre": "Adventure",
         "overview": "A FedEx systems analyst executive undergoes a physical and emotional transformation after surviving a plane crash and washing ashore on an uninhabited tropical island in the South Pacific.",
         "keywords": ["plane crash survivor", "deserted island", "wilson volleyball", "spear fishing", "lonely survival", "four years stranded"]},
        {"id": 1009, "title": "Indiana Jones and the Last Crusade", "genre": "Adventure",
         "overview": "When his estranged father Dr. Henry Jones Sr. suddenly disappears while pursuing the legendary Holy Grail, Indiana Jones must follow in his footsteps to stop ancient treasure from falling into evil hands.",
         "keywords": ["holy grail", "father and son", "temple booby traps", "archaeological quest", "venice catacombs", "brotherhood of the cruciform"]},
        {"id": 1010, "title": "King Kong", "genre": "Adventure",
         "overview": "A theatrical movie crew travels to a mysterious, uncharted Skull Island where they encounter prehistoric beasts and a giant ape who becomes infatuated with their glamorous leading lady.",
         "keywords": ["skull island", "giant gorilla kong", "prehistoric monsters", "uncharted ocean", "empire state building", "lost expedition"]}
    ]

    # Convert to DataFrames matching Kaggle's schema
    movies_meta = []
    keywords_list = []

    for item in records:
        mid = item["id"]
        genre_json = json.dumps([{"id": mid, "name": item["genre"]}])
        kw_json = json.dumps([{"id": i, "name": kw} for i, kw in enumerate(item["keywords"])])

        movies_meta.append({
            "id": str(mid),
            "title": item["title"],
            "overview": item["overview"],
            "genres": genre_json,
            "release_date": "2010-01-01",
            "vote_average": 8.0
        })

        keywords_list.append({
            "id": str(mid),
            "keywords": kw_json
        })

    df_meta = pd.DataFrame(movies_meta)
    df_kw = pd.DataFrame(keywords_list)

    if save_files and data_dir:
        os.makedirs(data_dir, exist_ok=True)
        meta_path = os.path.join(data_dir, "movies_metadata.csv")
        kw_path = os.path.join(data_dir, "keywords.csv")
        if not os.path.exists(meta_path):
            df_meta.to_csv(meta_path, index=False)
            print(f"[Preprocess] Created fallback dataset: {meta_path}")
        if not os.path.exists(kw_path):
            df_kw.to_csv(kw_path, index=False)
            print(f"[Preprocess] Created fallback dataset: {kw_path}")

    return df_meta, df_kw


def load_raw_datasets(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load movies_metadata.csv and keywords.csv from data_dir.
    If either file is missing or unreadable, trigger the synthetic fallback generator.
    """
    meta_path = os.path.join(data_dir, "movies_metadata.csv")
    kw_path = os.path.join(data_dir, "keywords.csv")

    has_meta = os.path.isfile(meta_path)
    has_kw = os.path.isfile(kw_path)

    if not has_meta or not has_kw:
        print(f"[Preprocess] Kaggle CSV files missing in '{data_dir}'. Generating fallback dataset...")
        return generate_synthetic_data(data_dir=data_dir, save_files=True)

    try:
        # Load Kaggle CSV with low_memory=False to handle mixed types
        print(f"[Preprocess] Loading dataset from '{meta_path}'...")
        df_meta = pd.read_csv(meta_path, low_memory=False)
        print(f"[Preprocess] Loading keywords from '{kw_path}'...")
        df_kw = pd.read_csv(kw_path, low_memory=False)
        return df_meta, df_kw
    except Exception as e:
        print(f"[Preprocess] Error loading files ({e}). Using synthetic generator...")
        return generate_synthetic_data(data_dir=data_dir, save_files=True)


def load_and_preprocess_data(data_dir: Optional[str] = None, top_n_genres: int = 10) -> pd.DataFrame:
    """
    Complete end-to-end data preprocessing pipeline:
    1. Loads raw datasets (or falls back to synthetic data).
    2. Merges on 'id' cleanly (sanitizing ids to avoid corruption).
    3. Safely parses JSON genres via ast.literal_eval and extracts primary genre.
    4. Safely parses JSON keywords into textual tokens.
    5. Combines overview + keywords into a unified corpus.
    6. Cleans the text (lowercasing, punctuation/number removal, stopword filtering).
    7. Filters to top N most frequent genres to resolve class imbalance.
    """
    if data_dir is None:
        # Default to ../data or ./data relative to this script
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = str(base_dir / "data")

    df_meta, df_kw = load_raw_datasets(data_dir)

    # Ensure required columns exist
    for col in ["id", "title", "overview", "genres"]:
        if col not in df_meta.columns:
            raise ValueError(f"Missing required column '{col}' in movies metadata.")
    if "id" not in df_kw.columns or "keywords" not in df_kw.columns:
        raise ValueError("Missing required columns ('id', 'keywords') in keywords dataset.")

    # Sanitize 'id' column: drop invalid string dates like '1997-08-20' present in Kaggle dataset
    df_meta = df_meta.copy()
    df_kw = df_kw.copy()

    df_meta["id_clean"] = pd.to_numeric(df_meta["id"], errors="coerce")
    df_kw["id_clean"] = pd.to_numeric(df_kw["id"], errors="coerce")

    df_meta = df_meta.dropna(subset=["id_clean"])
    df_kw = df_kw.dropna(subset=["id_clean"])

    df_meta["id_clean"] = df_meta["id_clean"].astype(int)
    df_kw["id_clean"] = df_kw["id_clean"].astype(int)

    # Drop duplicate IDs
    df_meta = df_meta.drop_duplicates(subset=["id_clean"])
    df_kw = df_kw.drop_duplicates(subset=["id_clean"])

    # Merge on clean integer ID
    print("[Preprocess] Merging movie metadata and keywords...")
    merged = pd.merge(df_meta, df_kw[["id_clean", "keywords"]], on="id_clean", how="left")

    # Drop rows missing overview or genres
    merged = merged.dropna(subset=["overview", "genres"])
    merged["overview"] = merged["overview"].astype(str)
    merged["keywords"] = merged["keywords"].fillna("[]").astype(str)

    # Parse primary genre
    print("[Preprocess] Parsing primary genres with ast.literal_eval...")
    merged["primary_genre"] = merged["genres"].apply(extract_primary_genre)
    merged = merged.dropna(subset=["primary_genre"])
    merged = merged[merged["primary_genre"].str.len() > 0]

    # Parse keywords to text
    print("[Preprocess] Extracting keywords text...")
    merged["keywords_text"] = merged["keywords"].apply(extract_keywords_text)

    # Filter to top N most frequent genres to avoid class imbalance
    top_genres = merged["primary_genre"].value_counts().nlargest(top_n_genres).index.tolist()
    print(f"[Preprocess] Top {top_n_genres} Genres: {top_genres}")
    filtered = merged[merged["primary_genre"].isin(top_genres)].copy()

    # Combine input features: plot summary + keywords
    print("[Preprocess] Combining overview and keywords corpus...")
    filtered["combined_text"] = filtered["overview"] + " " + filtered["keywords_text"]

    # Clean the combined text
    print("[Preprocess] Applying NLP cleaning (lowercasing, punctuation/number removal, stop words)...")
    filtered["cleaned_text"] = filtered["combined_text"].apply(clean_text)

    # Filter out empty or trivially short cleaned texts
    filtered = filtered[filtered["cleaned_text"].str.strip().str.len() > 10].copy()

    filtered = filtered.reset_index(drop=True)
    print(f"[Preprocess] Preprocessing complete. Final dataset shape: {filtered.shape}")

    return filtered


if __name__ == "__main__":
    df = load_and_preprocess_data()
    print(f"Sample Processed Data:\n{df[['title', 'primary_genre', 'cleaned_text']].head()}")
