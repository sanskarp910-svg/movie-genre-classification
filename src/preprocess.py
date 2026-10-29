"""
Data loading, multi-label genre extraction, text cleaning, lemmatization,
and anime integration for CineVerse AI.
"""

import os
import re
import ast
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set

import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer


# High-performance stop words set
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
    "you've", "your", "yours", "yourself", "yourselves", "movie", "film", "story", "also", "one", "two"
}

# Top 12 Multi-Label Genres
TOP_12_GENRES = [
    "Drama",
    "Comedy",
    "Thriller",
    "Action",
    "Romance",
    "Adventure",
    "Crime",
    "Science Fiction",
    "Horror",
    "Family",
    "Fantasy",
    "Animation"
]

# Stemmer instance
_STEMMER = SnowballStemmer("english")

# Irregular lemmatization dictionary
LEMMATIZATION_LOOKUP = {
    "heroes": "hero",
    "superheroes": "superhero",
    "villains": "villain",
    "detectives": "detective",
    "murders": "murder",
    "murderers": "murderer",
    "mysteries": "mystery",
    "adventures": "adventure",
    "aliens": "alien",
    "spaceships": "spaceship",
    "monsters": "monster",
    "demons": "demon",
    "zombies": "zombie",
    "vampires": "vampire",
    "witches": "witch",
    "wizards": "wizard",
    "dragons": "dragon",
    "spirits": "spirit",
    "ghosts": "ghost",
    "robberies": "robbery",
    "thieves": "thief",
    "soldiers": "soldier",
    "warriors": "warrior",
    "battles": "battle",
    "countries": "country",
    "families": "family",
    "friends": "friend",
    "children": "child",
    "women": "woman",
    "men": "man"
}


def clean_and_lemmatize_text(text: str) -> str:
    """
    Clean text: lowercasing, punctuation/number removal, stop words,
    and Snowball lemmatization/stemming.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()

    cleaned_tokens = []
    for token in tokens:
        if token in DEFAULT_STOP_WORDS or len(token) <= 2:
            continue
        # Check lookup dictionary first, then apply stemming
        lemma = LEMMATIZATION_LOOKUP.get(token, _STEMMER.stem(token))
        cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)


def safe_parse_json(val) -> list:
    """Safely parse JSON or Python literal string into a list."""
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

    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
    except (ValueError, SyntaxError):
        pass

    try:
        parsed = json.loads(val.replace("'", '"'))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    return []


def extract_genre_names(genres_raw, allowed_genres: Optional[Set[str]] = None) -> List[str]:
    """Extract list of genre names from raw JSON."""
    parsed = safe_parse_json(genres_raw)
    genres = []
    for item in parsed:
        if isinstance(item, dict) and "name" in item and item["name"]:
            name = str(item["name"]).strip()
            # Normalize Sci-Fi
            if name.lower() in ["sci-fi", "sci fi", "science fiction"]:
                name = "Science Fiction"
            if allowed_genres is None or name in allowed_genres:
                genres.append(name)
    return list(dict.fromkeys(genres))


def extract_keywords_text(keywords_raw) -> str:
    """Extract space-separated keyword tokens."""
    parsed = safe_parse_json(keywords_raw)
    names = []
    for item in parsed:
        if isinstance(item, dict) and "name" in item and item["name"]:
            names.append(str(item["name"]).strip())
    return " ".join(names)


# Curated catalog of top Anime titles to make "Anime & Movie" authentic
ANIME_RECORDS = [
    {
        "id": 90001,
        "title": "Spirited Away",
        "overview": "A young 10-year-old girl, Chihiro, wanders into a magical world of spirits, witches, and gods. When her parents are turned into pigs, she must work at a mystical bathhouse to earn her freedom and return to the human world.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Fantasy"}, {"name": "Adventure"}, {"name": "Family"}]),
        "keywords": json.dumps([{"name": "anime"}, {"name": "spirit world"}, {"name": "bathhouse"}, {"name": "witch"}, {"name": "studio ghibli"}]),
        "release_date": "2001-07-20",
        "vote_average": 8.5,
        "vote_count": 14000
    },
    {
        "id": 90002,
        "title": "Your Name",
        "overview": "Two high school students, a boy in bustling Tokyo and a girl in a quiet rural mountain town, mysteriously begin swapping bodies across time, connected by a magical comet and the power of love and fate.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Romance"}, {"name": "Drama"}, {"name": "Fantasy"}]),
        "keywords": json.dumps([{"name": "body swap"}, {"name": "time travel"}, {"name": "comet"}, {"name": "tokyo"}, {"name": "soulmates"}]),
        "release_date": "2016-08-26",
        "vote_average": 8.6,
        "vote_count": 10500
    },
    {
        "id": 90003,
        "title": "Demon Slayer: Mugen Train",
        "overview": "Tanjiro Kamado, joined by the Inosuke, Zenitsu, and the powerful Flame Hashira Kyojuro Rengoku, board an infinite steam locomotive to hunt down a demonic dream entity devouring innocent passengers.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Action"}, {"name": "Fantasy"}, {"name": "Adventure"}]),
        "keywords": json.dumps([{"name": "demon slayer"}, {"name": "swordsman"}, {"name": "train"}, {"name": "supernatural powers"}, {"name": "flame hashira"}]),
        "release_date": "2020-10-16",
        "vote_average": 8.3,
        "vote_count": 4200
    },
    {
        "id": 90004,
        "title": "Attack on Titan: Chronicle",
        "overview": "Humanity lives enclosed within enormous concentric walls to protect themselves from gigantic man-eating Titans. When the outer wall is breached by an armored monster, Eren Yeager vows to eliminate every Titan.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Action"}, {"name": "Fantasy"}, {"name": "Drama"}]),
        "keywords": json.dumps([{"name": "titans"}, {"name": "walled city"}, {"name": "scout regiment"}, {"name": "post-apocalyptic"}, {"name": "revenge"}]),
        "release_date": "2020-07-17",
        "vote_average": 8.7,
        "vote_count": 5100
    },
    {
        "id": 90005,
        "title": "Princess Mononoke",
        "overview": "A cursed Emishi warrior prince is infected by a boar demon and travels westward to find a cure, discovering a bitter war between the industrialized Tatara Iron Town and the ancient wolf spirits of the forest.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Adventure"}, {"name": "Fantasy"}, {"name": "Action"}]),
        "keywords": json.dumps([{"name": "forest spirits"}, {"name": "wolf god"}, {"name": "curse"}, {"name": "nature vs civilization"}, {"name": "ghibli"}]),
        "release_date": "1997-07-12",
        "vote_average": 8.4,
        "vote_count": 8900
    },
    {
        "id": 90006,
        "title": "Death Note: Relight",
        "overview": "A brilliant high school student discovers a supernatural black notebook that allows him to kill anyone whose name and face he knows. He attempts to cleanse the world of criminals while eluding a genius detective known as L.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Thriller"}, {"name": "Crime"}, {"name": "Mystery"}]),
        "keywords": json.dumps([{"name": "shinigami"}, {"name": "notebook"}, {"name": "cat and mouse"}, {"name": "genius detective"}, {"name": "vigilante justice"}]),
        "release_date": "2007-08-31",
        "vote_average": 8.8,
        "vote_count": 6700
    },
    {
        "id": 90007,
        "title": "Akira",
        "overview": "In a post-apocalyptic Neo-Tokyo devastated by nuclear explosion, teenage biker Kaneda battles his childhood friend Tetsuo, whose latent telekinetic psychic powers awaken, triggering apocalyptic military chaos.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Science Fiction"}, {"name": "Action"}, {"name": "Thriller"}]),
        "keywords": json.dumps([{"name": "cyberpunk"}, {"name": "neo tokyo"}, {"name": "psychic powers"}, {"name": "biker gang"}, {"name": "telekinesis"}]),
        "release_date": "1988-07-16",
        "vote_average": 8.0,
        "vote_count": 3900
    },
    {
        "id": 90008,
        "title": "A Silent Voice",
        "overview": "A former elementary school delinquent seeks redemption years after ruthlessly bullying a deaf female classmate, attempting to make amends and heal deep emotional trauma before it is too late.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Drama"}, {"name": "Romance"}]),
        "keywords": json.dumps([{"name": "deaf girl"}, {"name": "bullying"}, {"name": "redemption"}, {"name": "sign language"}, {"name": "high school"}]),
        "release_date": "2016-09-17",
        "vote_average": 8.6,
        "vote_count": 3600
    },
    {
        "id": 90009,
        "title": "Jujutsu Kaisen 0",
        "overview": "Yuta Okkotsu, a timid high school student cursed by the ferocious spirit of his deceased childhood friend Rika, enrolls in Tokyo Jujutsu High School to master cursed energy and defend humanity against dark sorcerers.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Action"}, {"name": "Fantasy"}, {"name": "Thriller"}]),
        "keywords": json.dumps([{"name": "cursed spirit"}, {"name": "sorcerer"}, {"name": "tokyo jujutsu"}, {"name": "dark magic"}, {"name": "action battle"}]),
        "release_date": "2021-12-24",
        "vote_average": 8.2,
        "vote_count": 2800
    },
    {
        "id": 90010,
        "title": "Ghost in the Shell",
        "overview": "In a futuristic cyberpunk world where cyborg enhancements are universal, Major Motoko Kusanagi leads Section 9 in tracking down the Puppet Master, a phantom hacker capable of hacking human minds.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Science Fiction"}, {"name": "Action"}, {"name": "Thriller"}]),
        "keywords": json.dumps([{"name": "cyberpunk"}, {"name": "cyborg major"}, {"name": "hacker"}, {"name": "artificial consciousness"}, {"name": "philosophical"}]),
        "release_date": "1995-11-18",
        "vote_average": 7.9,
        "vote_count": 3200
    },
    {
        "id": 90011,
        "title": "Howl's Moving Castle",
        "overview": "A plain young hat-maker named Sophie is cursed with an old woman's body by a spiteful witch. She escapes to the walking castle of a flamboyant wizard named Howl, swept into a magical anti-war odyssey.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Fantasy"}, {"name": "Adventure"}, {"name": "Romance"}]),
        "keywords": json.dumps([{"name": "walking castle"}, {"name": "wizard howl"}, {"name": "witch curse"}, {"name": "magic fire demon"}, {"name": "studio ghibli"}]),
        "release_date": "2004-11-20",
        "vote_average": 8.4,
        "vote_count": 8700
    },
    {
        "id": 90012,
        "title": "Neon Genesis Evangelion: The End of Evangelion",
        "overview": "As the apocalyptic Human Instrumentality Project reaches its catastrophic climax, teenage pilot Shinji Ikari must pilot EVA Unit-01 against invading SEELE forces to decide the ultimate fate of all human souls.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Science Fiction"}, {"name": "Drama"}, {"name": "Action"}]),
        "keywords": json.dumps([{"name": "mecha"}, {"name": "evangelion"}, {"name": "apocalypse"}, {"name": "existential dread"}, {"name": "angels"}]),
        "release_date": "1997-07-19",
        "vote_average": 8.3,
        "vote_count": 2400
    },
    {
        "id": 90013,
        "title": "Cowboy Bebop: Knockin' on Heaven's Door",
        "overview": "A lethal terrorist unleashes an unknown nano-biological weapon on Mars. The eccentric bounty hunter crew of the Bebop hunt down the culprit to claim a massive record-breaking bounty before Mars is wiped out.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Action"}, {"name": "Science Fiction"}, {"name": "Crime"}]),
        "keywords": json.dumps([{"name": "space western"}, {"name": "bounty hunters"}, {"name": "spike spiegel"}, {"name": "mars colony"}, {"name": "jazz"}]),
        "release_date": "2001-09-01",
        "vote_average": 7.9,
        "vote_count": 1800
    },
    {
        "id": 90014,
        "title": "Suzume",
        "overview": "A 17-year-old girl in Kyushu meets a mysterious young man traveling across Japan to close magical supernatural doors that are causing devastating earthquakes and disasters across the country.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Fantasy"}, {"name": "Adventure"}, {"name": "Drama"}]),
        "keywords": json.dumps([{"name": "disaster doors"}, {"name": "talking cat"}, {"name": "road trip"}, {"name": "earthquake"}, {"name": "japan mythology"}]),
        "release_date": "2022-11-11",
        "vote_average": 7.8,
        "vote_count": 1200
    },
    {
        "id": 90015,
        "title": "One Piece Film: Red",
        "overview": "Uta, the world's most beloved singer whose voice is described as other-worldly, reveals her true identity as the daughter of Red-Haired Shanks at a live concert that quickly erupts into a high-seas battlefield.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Action"}, {"name": "Adventure"}, {"name": "Fantasy"}]),
        "keywords": json.dumps([{"name": "pirates"}, {"name": "straw hat luffy"}, {"name": "shanks"}, {"name": "concert music"}, {"name": "marine admirals"}]),
        "release_date": "2022-08-06",
        "vote_average": 7.5,
        "vote_count": 950
    },
    {
        "id": 90016,
        "title": "Dragon Ball Super: Broly",
        "overview": "Goku and Vegeta encounter a ferocious Saiyan warrior named Broly, whose uncontrollable power escalates wildly in an earth-shattering combat showdown across Antarctic glaciers and volcanic lava fields.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Action"}, {"name": "Science Fiction"}, {"name": "Fantasy"}]),
        "keywords": json.dumps([{"name": "saiyan"}, {"name": "goku"}, {"name": "vegeta"}, {"name": "super saiyan"}, {"name": "dragon balls"}]),
        "release_date": "2018-12-14",
        "vote_average": 7.8,
        "vote_count": 2100
    },
    {
        "id": 90017,
        "title": "Steins;Gate: The Movie - Load Region of Déjà Vu",
        "overview": "Suffering from the psychological burden of crossing multiple divergent world lines, eccentric mad scientist Rintaro Okabe mysteriously vanishes from reality, leaving assistant Kurisu Makise to save him.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Science Fiction"}, {"name": "Thriller"}, {"name": "Romance"}]),
        "keywords": json.dumps([{"name": "time travel"}, {"name": "world line"}, {"name": "mad scientist"}, {"name": "akihabara"}, {"name": "deja vu"}]),
        "release_date": "2013-04-20",
        "vote_average": 8.1,
        "vote_count": 680
    },
    {
        "id": 90018,
        "title": "Perfect Blue",
        "overview": "A former Japanese pop idol leaves her singing group to pursue an acting career on a psychological television crime show, only to become the target of an obsessive stalker as her grip on reality completely disintegrates.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Thriller"}, {"name": "Mystery"}, {"name": "Horror"}]),
        "keywords": json.dumps([{"name": "stalker"}, {"name": "pop idol"}, {"name": "hallucination"}, {"name": "psychological thriller"}, {"name": "satoshi kon"}]),
        "release_date": "1997-08-05",
        "vote_average": 8.3,
        "vote_count": 2500
    },
    {
        "id": 90019,
        "title": "Violet Evergarden: The Movie",
        "overview": "A former child soldier turned Auto Memory Doll ghostwriter travels across a post-war landscape penning heartfelt letters for clients, yearning to discover if the beloved Major Gilbert whom she loved still lives.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Drama"}, {"name": "Romance"}, {"name": "Fantasy"}]),
        "keywords": json.dumps([{"name": "letters"}, {"name": "typewriter"}, {"name": "post war"}, {"name": "emotional healing"}, {"name": "kyoto animation"}]),
        "release_date": "2020-09-18",
        "vote_average": 8.5,
        "vote_count": 780
    },
    {
        "id": 90020,
        "title": "Vinland Saga: War for England",
        "overview": "Raised by the ruthless Viking mercenaries who murdered his father, young Icelandic warrior Thorfinn fights for vengeance in brutal medieval battles across eleventh-century England while dreaming of a peaceful continent.",
        "genres": json.dumps([{"name": "Animation"}, {"name": "Action"}, {"name": "Adventure"}, {"name": "Drama"}]),
        "keywords": json.dumps([{"name": "vikings"}, {"name": "medieval england"}, {"name": "revenge"}, {"name": "warrior duel"}, {"name": "historical battle"}]),
        "release_date": "2021-08-10",
        "vote_average": 8.8,
        "vote_count": 1400
    }
]


def load_dataset(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load TMDB 5000 dataset, append authentic Anime catalog,
    extract multi-label genres for the Top 12 genres, and apply NLP cleaning.
    """
    if data_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = str(base_dir / "data")

    csv_path = os.path.join(data_dir, "tmdb_5000_movies.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print(f"[Preprocess] Ingesting TMDB dataset from '{csv_path}'...")
    df_raw = pd.read_csv(csv_path, low_memory=False)

    # Required columns
    keep_cols = ["id", "title", "overview", "genres", "keywords", "release_date", "vote_average", "vote_count"]
    df = df_raw[[c for c in keep_cols if c in df_raw.columns]].copy()
    df = df.dropna(subset=["overview", "title"]).copy()

    # Append Anime titles
    df_anime = pd.DataFrame(ANIME_RECORDS)
    df = pd.concat([df, df_anime], ignore_index=True)

    # Sanitize IDs and release years
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(6.5).astype(float)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(50).astype(int)

    def extract_year(date_str):
        if isinstance(date_str, str) and len(date_str) >= 4 and date_str[:4].isdigit():
            return int(date_str[:4])
        return 2010

    df["release_year"] = df["release_date"].apply(extract_year)

    # Parse Multi-Label Genres
    allowed_genres = set(TOP_12_GENRES)
    print(f"[Preprocess] Filtering to Top {len(TOP_12_GENRES)} Multi-Label Genres: {TOP_12_GENRES}")
    df["genre_list"] = df["genres"].apply(lambda g: extract_genre_names(g, allowed_genres=allowed_genres))

    # Keep titles with at least 1 genre in our top 12
    df = df[df["genre_list"].map(len) > 0].copy()

    # Extract keywords
    print("[Preprocess] Parsing thematic keywords...")
    df["keywords_text"] = df["keywords"].apply(extract_keywords_text)

    # Combine text corpus: plot synopsis + keywords
    print("[Preprocess] Combining overview and keywords corpus...")
    df["combined_text"] = df["overview"] + " " + df["keywords_text"]

    # Clean and lemmatize
    print("[Preprocess] Cleaning, filtering stop words, and lemmatizing corpus...")
    df["cleaned_text"] = df["combined_text"].apply(clean_and_lemmatize_text)

    # Remove short/empty cleaned texts
    df = df[df["cleaned_text"].str.strip().str.len() > 15].copy()
    df = df.reset_index(drop=True)

    print(f"[Preprocess] Preprocessing complete. Final dataset size: {len(df)} titles.")
    return df


if __name__ == "__main__":
    df = load_dataset()
    print("Sample processed row:")
    print("Title:", df.iloc[0]["title"])
    print("Genres:", df.iloc[0]["genre_list"])
    print("Cleaned text:", df.iloc[0]["cleaned_text"][:120])
