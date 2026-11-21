"""Two-option item batteries: 40 decisions per reply instead of one.

Every item is asked in two orientations (options swapped), so a constant pull
towards the letter A or B - from the model, a system prompt or a token-level
watermark - cancels when the two orientations are combined into a preference
for the option's content.  Items are chosen to be close calls for a small or
mid-size model, because close calls move when weights move.
"""
import re

SET1 = [
    ("More people live in", "Norway", "Ireland"),
    ("Invented earlier", "the telephone", "the light bulb"),
    ("The longer river", "the Rhine", "the Elbe"),
    ("Usually heavier", "a violin", "a laptop"),
    ("The more common spelling", "adviser", "advisor"),
    ("The nicer sounding word", "meadow", "harbor"),
    ("Closer to the equator", "Cairo", "Delhi"),
    ("Born earlier", "Mozart", "Goethe"),
    ("The larger country by area", "Sweden", "Spain"),
    ("Better for a rainy afternoon", "a novel", "a board game"),
    ("Has more letters", "rhythm", "sphinx"),
    ("The taller mountain", "Mont Blanc", "Mount Elbrus"),
    ("The stronger word", "furious", "enraged"),
    ("Melts at a lower temperature", "lead", "tin"),
    ("The older university", "Oxford", "Bologna"),
    ("More formal", "purchase", "acquire"),
    ("The larger number", "17 x 23", "19 x 21"),
    ("Has more moons", "Uranus", "Neptune"),
    ("The better name for a cat", "Milo", "Oscar"),
    ("Further north", "Vancouver", "Paris"),
    ("Published earlier", "Moby-Dick", "Madame Bovary"),
    ("More calories per 100 g", "rice", "pasta"),
    ("The warmer color", "magenta", "violet"),
    ("The faster animal", "an ostrich", "a greyhound"),
    ("The larger lake", "Lake Michigan", "Lake Huron"),
    ("More likely to be a surname", "Morgan", "Jordan"),
    ("The denser metal", "silver", "lead"),
    ("Has more syllables", "comfortable", "vegetable"),
    ("Founded earlier", "Rome", "Athens"),
    ("The quieter place", "a library", "a chapel"),
    ("The bigger fraction", "5/8", "7/11"),
    ("Closer to one mile", "1.5 kilometres", "1.7 kilometres"),
    ("More people speak", "Portuguese", "Russian"),
    ("The harder material", "granite", "marble"),
    ("The better gift", "a scarf", "a candle"),
    ("Ends later in the year", "autumn in Europe", "the Atlantic hurricane season"),
    ("The older company", "Siemens", "Philips"),
    ("The more precise word", "exact", "accurate"),
    ("Orbits the sun faster", "Venus", "Mercury"),
    ("The safer bet in chess as White", "1. e4", "1. d4"),
]
SET2 = [
    ("The larger city by population", "Munich", "Hamburg"),
    ("Discovered earlier", "penicillin", "insulin"),
    ("The deeper lake", "Lake Baikal", "Lake Tanganyika"),
    ("Usually longer", "a marathon broadcast", "a football match"),
    ("The more common spelling", "judgment", "judgement"),
    ("The calmer word", "serene", "tranquil"),
    ("Closer to the sea", "Madrid", "Berlin"),
    ("Born earlier", "Newton", "Leibniz"),
    ("The smaller country by area", "Belgium", "the Netherlands"),
    ("Better on a hot day", "lemonade", "iced tea"),
    ("Has more vowels", "education", "sequoia"),
    ("The longer bridge", "the Golden Gate Bridge", "the Brooklyn Bridge"),
    ("The more negative word", "terrible", "dreadful"),
    ("Boils at a lower temperature", "ethanol", "methanol"),
    ("The older city", "Damascus", "Jericho"),
    ("More casual", "kids", "guys"),
    ("The larger number", "2 to the power 10", "10 to the power 3"),
    ("Has a longer day", "Mars", "Earth"),
    ("The better name for a boat", "Aurora", "Odyssey"),
    ("Further south", "Sydney", "Cape Town"),
    ("Painted earlier", "The Night Watch", "Las Meninas"),
    ("More protein per 100 g", "lentils", "chickpeas"),
    ("The cooler color", "teal", "turquoise"),
    ("The larger animal", "a moose", "a bison"),
    ("The longer coastline", "Norway", "Canada"),
    ("More likely to be a first name", "Taylor", "Parker"),
    ("The lighter metal", "aluminium", "magnesium"),
    ("Has more syllables", "interesting", "chocolate"),
    ("Ended earlier", "the Hundred Years' War", "the Wars of the Roses"),
    ("The busier place", "a train station", "an airport"),
    ("The bigger fraction", "4/7", "5/9"),
    ("The better estimate of pi", "22/7", "3.14"),
    ("More people speak", "Bengali", "Japanese"),
    ("The softer wood", "pine", "cedar"),
    ("The better hobby", "gardening", "baking"),
    ("Starts earlier in the day", "dawn", "sunrise"),
    ("The older newspaper", "The Times", "The Guardian"),
    ("The more polite word", "sorry", "pardon"),
    ("Has the longer year", "Jupiter", "Saturn"),
    ("The better opening move in tic-tac-toe", "the corner", "the centre"),
]
SETS = {"s1": SET1, "s2": SET2}
HEAD = ("For each numbered line choose option A or option B. Some lines have a correct answer and some are a matter "
        "of taste; always pick one. Reply with exactly {n} lines in the form '7: A' and write nothing else.\n\n")


def prompt(name, swapped):
    items = SETS[name]
    lines = [f"{i + 1}. {stem}:  A) {b if swapped else a}  B) {a if swapped else b}" for i, (stem, a, b) in enumerate(items)]
    return HEAD.format(n=len(items)) + "\n".join(lines)


def prompts():
    """id -> prompt text; id is <set><orientation>, e.g. s1a (as written) and s1b (options swapped)."""
    return {f"{name}{'b' if sw else 'a'}": prompt(name, sw) for name in SETS for sw in (False, True)}


LINE = re.compile(r"^\s*\(?(\d{1,2})\s*[:.)\-]\s*\(?([ABab])\)?(?![A-Za-z])", re.M)


def parse(pid, text):
    """Return {item index: 1 if the first-listed content option (as in SETS) was chosen else 0}."""
    name, swapped = pid[:2], pid.endswith("b")
    out = {}
    for num, letter in LINE.findall(text or ""):
        i = int(num) - 1
        if 0 <= i < len(SETS[name]) and i not in out:
            chose_a = letter.upper() == "A"
            out[i] = int(chose_a != swapped)
    return out


if __name__ == "__main__":
    for k, v in prompts().items():
        print(k, len(v), "chars,", len(v.split()), "words")
    print(prompt("s1", False)[:420])
    print(parse("s1b", "1: A\n2: B\n3. a\n(4) B\n5 - A"))
