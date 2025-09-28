#!/usr/bin/env python3
"""
NLP Text Augmentation Pipeline
Applies deterministic text transformations for data augmentation
"""

"""
# Use: E.g. Generate 5 documents with 70% randomization
output_files = generate_text_suite_nlp_augment(
    input_text_path="shakespeare.txt",
    output_suite_dir_path="./augmented_texts/",
    num_documents=5,
    randomization_level=0.7,
    seed=42,
    aug_1_sentence_spacing='random',
    aug_9_contractions='all',
    aug_10_capitalization='skip'
)
"""

"""
Topic: text augmentation and semantic equivalence of strings:
What is the minimal amount of language-augmentation needed to improve non-over-fitting learning for that document? The context here is changes that can be automated, especially deterministically.
by analogy with image augmentation: building a text preprocessing pipeline analogous to image augmentation techniques
"Language does not exist in three dimensional space, but serialized images don't either."
- 'rotation' as rephrasing
- panning: adding or removing filler-text before or after the content.
- cropping
- color/hue change/filter
Some changes (maybe like rotation and panning) are frame-shifting the same language.
Some changes are ~grammatically different but with the same meaning. (Won't vs/ Will not) (which also acts as a frame shift).

## Goals:
Python code should:
- be modular / functional, error handling, docstrings, etc.
- have user configuration parameters for:
-- how many documents to create
-- select which of the 12 type below to use
-- input a random seed (for repeatability)

If a lot of parameters/arguments, the simplest approach may be to have in the main function:

```python
generate_text_suite_nlp_augment(
input_text_path:
output_suite_dir_path:
1_skip_random_all='random';
1_variation='random';
...
10_skipall_skiprandom='skip'# special case
10_variation_upper_lower_random_skip='random' # special case
):
```
where '_skip_random_all='random';' means
='skip' is don't augment
='random' is use random see to apply to random cases
='all' is apply in all cases
_variation='random'; means it picks a random quantity or option. this is default and for MVP is the only real setting (e.g. not all options configured in detail for all things), no granular controls in version 1.
Random application of variation means that each instance has a random chance of being a given variation. Hence, "random." EVERY instance in EVERY document has a random chance.
e.g. randomization_level=0.5, # 50% (MVP applies to everything)
The user sets this ONE value, and it applies to everything that's set to 'random'.
So if the user sets:
randomization_level=0.3 → 30% chance at each instance
randomization_level=0.7 → 70% chance at each instance
This single parameter controls the probability for ALL augmentations that are set to 'random'.

incremental_random='random';, for MVP the only option is to select random variations (in future you could in theory do 'add one space' 'add two spaces' 'add three spaces'... etc.)
Output format: separate files, timestamp in names after original file name (in future can do other things too)
input path alice.txt (always .txt utf8 for MVP)
output path alice_nlp_augmented_timestamp.txt
Synonym substitutions:
- only matched to / applied to lowercase
- only inserting lowercase
(note: making whole text upper or lower case should be done last (after all other augmentations).
Error handling: because this is R&D not production-software, fail raise exception with full traceback in the case of any errors or problems. use try except with traceback wherever possible
for MVP no 3rd party libraries (no pip installs) any standard-library is fine:
- glob
- datetime
- re
- random
- traceback
etc.

Edge Cases of Random Application:
- for some (rare) augmentation areas, the augmentation applied extremely to the entire file,
e.g. all upper case, all lower case. In this case, if random is an option, random would mean skip or apply random. There is no 'case' within the file to apply randomly to (for MVP), it is whole file, or no augmentation. in future these might have a special probability or setting, e.g. 1 in suite is  upper, 1 in suite is lower... etc. this is an edge case that could be entirely skipped in MVP...but it is trivial to case-change the whole text... so no hard to implement.

# Types of Deterministic NLP Augmentation
"can be applied randomly" means, for each instance where it could be applied, there can be a (seeded) (pseudo)random 'coin flip' for whether to apply change to skip to next.
"can be applied variably" means, there is more than a binary choice, there are more than two ways this can be done. e.g. a choice of quantity used, or a choice of what to substitute.
1. add another space after a .?! before a capital letter
-- can be applied to all
-- can be applied randomly
-- can be applied variably
(future note , or if two, remove one.)
2. add space after \n (at beginning of new line)
-- can be applied to all
-- can be applied randomly
-- can be applied variably
3. add (or remove) spaces before/after symbols and punctuation = ()[]; : , ' "
-- can be applied to all
-- can be applied randomly
-- can be applied variably
(note: 1-3 may be able to be repeated at least 5 times (+5 spaces), ~incremented)
4. double newline \n\n
-- can be applied to all
-- can be applied randomly
5. two Single quotes vs. double quotes
-- can be be uniform: applied to all in one direction or the other
-- can be applied randomly
6. \t tab and N spaces (another good variable item 3,4,5,6,7 probably all fine)
-- can be applied to all
-- can be applied randomly
-- can be applied variably
7. single spaces: make all spaces double-spaces
-- can be applied to all
-- can be applied randomly
8. 'brackets': randomly change 'bracket' type (<[{ or }]>)
-- can be applied to all
-- can be applied randomly
-- can be applied variably
A. uniformly, e.g. all '(' -> '['
B. randomly (mismatched pairs)
brackets: options of open-types, options of closed-types. No open-closed switching.
Opening brackets: ( → [ or { or <
Closing brackets: ) → ] or } or >
Changing open to closed can change the meaning. We are not changing meaning.
"He (Tom)" is the same as "He [tom]".
"He] tom [But", is not bracketing the string 'tom'
9. change contractions to phrases and vice-versa
(won't, don't, can't shouldn't, he's, they're, it's)
-- can be applied to all
-- can be applied randomly
10. Capitalization:
- all lower case
- all upper case
(future note: unclear what effects randomization would have...
all-lower is truly practical,
all-upper is a bit abstract,)

11. Number/numeral swapping
"3" ↔ "three" (for small numbers)
"1st" ↔ "first"
"100" ↔ "one hundred"
(note, there may be edge cases to avoid...
'one' is often not a number
"When one goes to the store."

Is number to word safest?
-- can be applied to all
-- can be applied randomly

12. swapping synonymous strings that are common enough to be found, but not so common and terse as to have varied nuanced meanings (be/use have innumerable different meanings/uses)
-- can be applied to all
-- can be applied randomly
-- some are variable


Conservative list:


VERB SYNONYMS (Use-Equivalent):

begin = start
repair = mend
talk about = discuss
look at = observe
try = attempt
understand = comprehend


ADJECTIVE SYNONYMS (Use-Equivalent):

difficult "not easy"
easy = "not difficult"
big = large = huge
small = little = tiny
happy = glad = pleased
sad = unhappy = "not happy"
wrong = "not right"
slow=" not fast "
shouted = called
hurried = rushed
disappeared = vanished
affectionately = lovingly
conversation = " verbal exchange "
said the = spoke the = articulated the = stated the

he said = he stated
she said = she stated

MVP:
This is a basic test:
- no special case handling (url, phone, email, code, markdown, etc.)
- Just English, focusing on ASCII. (Maybe only 12 is highly language specific,
contractions may be sometimes not applicable. Romance-language may have some common patterns.)


# output file names
e.g.
input file path /shakespeare.txt
output file names
shakespeare_aug_{number}_{timestamp}.txt

Sequence of application:

for MVP, aside from the extreme all-document capitalization, there is no sequence (that I can think of) for other augmentations.


Again:
Random application of variation means that each instance has a random chance of being a given variation. Hence, "random." EVERY instance in EVERY document has a random chance.
e.g. randomization_level=0.5, # 50% (MVP applies to everything)
The user sets this ONE value, and it applies to everything that's set to 'random'.
So if the user sets:
randomization_level=0.3 → 30% chance at each instance
randomization_level=0.7 → 70% chance at each instance
This single parameter controls the probability for ALL augmentations that are set to 'random'.
For variations (e.g., "add 1-5 spaces") use the randomization_level to pick from those options.
For variations (e.g., which bracket type to use) use the randomization_level to pick from those options.
EVERY instance in EVERY document has a not-otherwise-specified-exception, random chance randomization_level=N MVP applies to everything.
AGAIN:
Random is not per document.

"""

import os
import re
import random
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class TextAugmenter:
    """Main class for applying NLP text augmentations"""

    def __init__(self, randomization_level: float = 0.5, seed: Optional[int] = None):
        """
        Initialize the TextAugmenter

        Args:
            randomization_level: Probability (0.0-1.0) for random augmentations
            seed: Random seed for reproducibility
        """
        self.randomization_level = randomization_level
        if seed is not None:
            random.seed(seed)

        # Synonym dictionary for conservative substitutions
        self.synonyms = {
            "begin": ["start"],
            "start": ["begin"],
            "repair": ["mend"],
            "mend": ["repair"],
            "talk about": ["discuss"],
            "discuss": ["talk about"],
            "look at": ["observe"],
            "observe": ["look at"],
            "try": ["attempt"],
            "attempt": ["try"],
            "understand": ["comprehend"],
            "comprehend": ["understand"],
            "difficult": ["not easy"],
            "easy": ["not difficult"],
            "big": ["large", "huge"],
            "large": ["big", "huge"],
            "huge": ["big", "large"],
            "small": ["little", "tiny"],
            "little": ["small", "tiny"],
            "tiny": ["small", "little"],
            "happy": ["glad", "pleased"],
            "glad": ["happy", "pleased"],
            "pleased": ["happy", "glad"],
            "sad": ["unhappy", "not happy"],
            "unhappy": ["sad", "not happy"],
            "wrong": ["not right"],
            "slow": ["not fast"],
            "shouted": ["called"],
            "called": ["shouted"],
            "hurried": ["rushed"],
            "rushed": ["hurried"],
            "disappeared": ["vanished"],
            "vanished": ["disappeared"],
            "affectionately": ["lovingly"],
            "lovingly": ["affectionately"],
            "conversation": ["verbal exchange"],
            "he said": ["he stated", "he spoke", "he articulated"],
            "she said": ["she stated", "she spoke", "she articulated"],
        }

        # Contraction mappings
        self.contractions = {
            "won't": "will not",
            "don't": "do not",
            "can't": "cannot",
            "shouldn't": "should not",
            "he's": "he is",
            "she's": "she is",
            "they're": "they are",
            "it's": "it is",
            "we're": "we are",
            "you're": "you are",
            "i'm": "i am",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
            "wouldn't": "would not",
            "couldn't": "could not",
        }

        # Number to word mappings (conservative list)
        self.numbers = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
            "10": "ten",
            "11": "eleven",
            "12": "twelve",
            "13": "thirteen",
            "14": "fourteen",
            "15": "fifteen",
            "16": "sixteen",
            "17": "seventeen",
            "18": "eighteen",
            "19": "nineteen",
            "20": "twenty",
            "1st": "first",
            "2nd": "second",
            "3rd": "third",
            "4th": "fourth",
            "5th": "fifth",
            "6th": "sixth",
            "7th": "seventh",
            "8th": "eighth",
            "9th": "ninth",
            "10th": "tenth",
        }

    def should_apply_random(self) -> bool:
        """Check if random augmentation should be applied based on randomization_level"""
        return random.random() < self.randomization_level

    def augment_sentence_spacing(self, text: str, mode: str) -> str:
        """Add extra spaces after sentence endings before capitals"""
        if mode == "skip":
            return text

        try:
            pattern = r"([.!?])(\s*)([A-Z])"

            def replace_func(match):
                punct, existing_space, capital = match.groups()
                if mode == "all" or (mode == "random" and self.should_apply_random()):
                    # Add 1-5 additional spaces
                    extra_spaces = " " * random.randint(1, 5)
                    return f"{punct}{existing_space}{extra_spaces}{capital}"
                return match.group(0)

            return re.sub(pattern, replace_func, text)
        except Exception as e:
            raise Exception(
                f"Error in augment_sentence_spacing: {str(e)}\n{traceback.format_exc()}"
            )

    def augment_newline_spacing(self, text: str, mode: str) -> str:
        """Add spaces after newlines"""
        if mode == "skip":
            return text

        try:
            lines = text.split("\n")
            result_lines = []

            for line in lines:
                if mode == "all" or (mode == "random" and self.should_apply_random()):
                    # Add 1-5 spaces at beginning of line
                    spaces = " " * random.randint(1, 5)
                    result_lines.append(f"{spaces}{line}")
                else:
                    result_lines.append(line)

            return "\n".join(result_lines)
        except Exception as e:
            raise Exception(
                f"Error in augment_newline_spacing: {str(e)}\n{traceback.format_exc()}"
            )

    def augment_punctuation_spacing(self, text: str, mode: str) -> str:
        """Add/remove spaces around punctuation and symbols"""
        if mode == "skip":
            return text

        try:
            punctuation = r'[=\(\)\[\];:,\'""]'
            pattern = f"({punctuation})"

            def replace_func(match):
                if mode == "all" or (mode == "random" and self.should_apply_random()):
                    punct = match.group(1)
                    spaces = " " * random.randint(1, 3)
                    # Randomly add spaces before, after, or both
                    choice = random.choice(["before", "after", "both"])
                    if choice == "before":
                        return f"{spaces}{punct}"
                    elif choice == "after":
                        return f"{punct}{spaces}"
                    else:  # both
                        return f"{spaces}{punct}{spaces}"
                return match.group(0)

            return re.sub(pattern, replace_func, text)
        except Exception as e:
            raise Exception(
                f"Error in augment_punctuation_spacing: {str(e)}\n{traceback.format_exc()}"
            )

    def augment_double_newlines(self, text: str, mode: str) -> str:
        """Convert single newlines to double newlines"""
        if mode == "skip":
            return text

        try:
            lines = text.split("\n")
            result_lines = []

            for i, line in enumerate(lines):
                result_lines.append(line)
                if i < len(lines) - 1:  # Not the last line
                    if mode == "all" or (
                        mode == "random" and self.should_apply_random()
                    ):
                        result_lines.append("")  # Add extra newline

            return "\n".join(result_lines)
        except Exception as e:
            raise Exception(
                f"Error in augment_double_newlines: {str(e)}\n{traceback.format_exc()}"
            )

    def augment_quote_style(self, text: str, mode: str) -> str:
        """Swap between single and double quotes"""
        if mode == "skip":
            return text

        try:

            def replace_quotes(match):
                quote = match.group(0)
                if mode == "all" or (mode == "random" and self.should_apply_random()):
                    return '"' if quote == "'" else "'"
                return quote

            # Replace single quotes
            text = re.sub(r"'", replace_quotes, text)
            # Replace double quotes
            text = re.sub(r'"', replace_quotes, text)

            return text
        except Exception as e:
            raise Exception(
                f"Error in augment_quote_style: {str(e)}\n{traceback.format_exc()}"
            )

    def augment_tabs_spaces(self, text: str, mode: str) -> str:
        """Convert tabs to spaces or vice versa"""
        if mode == "skip":
            return text

        try:
            if mode == "all" or (mode == "random" and self.should_apply_random()):
                # Convert tabs to random number of spaces (3-7)
                spaces = " " * random.randint(3, 7)
                return text.replace("\t", spaces)
            return text
        except Exception as e:
            raise Exception(
                f"Error in augment_tabs_spaces: {str(e)}\n{traceback.format_exc()}"
            )

    def augment_double_spaces(self, text: str, mode: str) -> str:
        """Convert single spaces to double spaces"""
        if mode == "skip":
            return text

        try:
            if mode == "all":
                return text.replace(" ", "  ")
            elif mode == "random":
                # Apply to each space randomly
                result = ""
                for char in text:
                    if char == " " and self.should_apply_random():
                        result += "  "
                    else:
                        result += char
                return result
            return text
        except Exception as e:
            raise Exception(
                f"Error in augment_double_spaces: {str(e)}\n{traceback.format_exc()}"
            )

    def augment_brackets(self, text: str, mode: str) -> str:
        """Change bracket types while preserving matching"""
        if mode == "skip":
            return text

        try:
            open_brackets = {
                "(": ["[", "{", "<"],
                "[": ["(", "{", "<"],
                "{": ["(", "[", "<"],
                "<": ["(", "[", "{"],
            }
            close_brackets = {
                ")": ["]", "}", ">"],
                "]": [")", "}", ">"],
                "}": [")", "]", ">"],
                ">": [")", "]", "}"],
            }

            def replace_bracket(match):
                bracket = match.group(0)
                if mode == "all" or (mode == "random" and self.should_apply_random()):
                    if bracket in open_brackets:
                        return random.choice(open_brackets[bracket])
                    elif bracket in close_brackets:
                        return random.choice(close_brackets[bracket])
                return bracket

            pattern = r"[(){}\[\]<>]"
            return re.sub(pattern, replace_bracket, text)
        except Exception as e:
            raise Exception(
                f"Error in augment_brackets: {str(e)}\n{traceback.format_exc()}"
            )

    def augment_contractions(self, text: str, mode: str) -> str:
        """Expand contractions or contract phrases"""
        if mode == "skip":
            return text

        try:
            # Create reverse mapping
            expansions = {v: k for k, v in self.contractions.items()}
            all_mappings = {**self.contractions, **expansions}

            # Sort by length (longest first) to avoid partial matches
            sorted_phrases = sorted(all_mappings.keys(), key=len, reverse=True)

            for phrase in sorted_phrases:
                if phrase.lower() in text.lower():

                    def replace_func(match):
                        if mode == "all" or (
                            mode == "random" and self.should_apply_random()
                        ):
                            return all_mappings[phrase.lower()]
                        return match.group(0)

                    pattern = re.escape(phrase.lower())
                    text = re.sub(pattern, replace_func, text, flags=re.IGNORECASE)

            return text
        except Exception as e:
            raise Exception(
                f"Error in augment_contractions: {str(e)}\n{traceback.format_exc()}"
            )

    def augment_capitalization(self, text: str, mode: str) -> str:
        """Apply document-level case changes"""
        if mode == "skip":
            return text

        try:
            if mode == "all":
                # For 'all', randomly choose upper or lower
                return text.upper() if random.random() < 0.5 else text.lower()
            elif mode == "random":
                if self.should_apply_random():
                    return text.upper() if random.random() < 0.5 else text.lower()
            return text
        except Exception as e:
            raise Exception(
                f"Error in augment_capitalization: {str(e)}\n{traceback.format_exc()}"
            )

    def augment_numbers(self, text: str, mode: str) -> str:
        """Convert numbers to words or vice versa"""
        if mode == "skip":
            return text

        try:
            # Create reverse mapping
            words_to_numbers = {v: k for k, v in self.numbers.items()}
            all_mappings = {**self.numbers, **words_to_numbers}

            # Sort by length (longest first)
            sorted_items = sorted(all_mappings.keys(), key=len, reverse=True)

            for item in sorted_items:
                pattern = r"\b" + re.escape(item) + r"\b"

                def replace_func(match):
                    if mode == "all" or (
                        mode == "random" and self.should_apply_random()
                    ):
                        return all_mappings[item]
                    return match.group(0)

                text = re.sub(pattern, replace_func, text, flags=re.IGNORECASE)

            return text
        except Exception as e:
            raise Exception(
                f"Error in augment_numbers: {str(e)}\n{traceback.format_exc()}"
            )

    def augment_synonyms(self, text: str, mode: str) -> str:
        """Replace words with synonyms (lowercase matching only)"""
        if mode == "skip":
            return text

        try:
            # Sort by length (longest first) to handle phrases before individual words
            sorted_phrases = sorted(self.synonyms.keys(), key=len, reverse=True)

            for phrase in sorted_phrases:
                # Create pattern for case-insensitive matching but preserve original case
                pattern = r"\b" + re.escape(phrase) + r"\b"

                def replace_func(match):
                    if mode == "all" or (
                        mode == "random" and self.should_apply_random()
                    ):
                        original = match.group(0)
                        synonym = random.choice(self.synonyms[phrase.lower()])

                        # Preserve case pattern of original
                        if original.isupper():
                            return synonym.upper()
                        elif original.istitle():
                            return synonym.title()
                        else:
                            return synonym.lower()
                    return match.group(0)

                text = re.sub(pattern, replace_func, text, flags=re.IGNORECASE)

            return text
        except Exception as e:
            raise Exception(
                f"Error in augment_synonyms: {str(e)}\n{traceback.format_exc()}"
            )

    def apply_augmentations(self, text: str, config: Dict[str, str]) -> str:
        """Apply all configured augmentations to text"""
        try:
            # Apply augmentations 1-9 and 11-12 first
            text = self.augment_sentence_spacing(
                text, config.get("1_sentence_spacing", "skip")
            )
            text = self.augment_newline_spacing(
                text, config.get("2_newline_spacing", "skip")
            )
            text = self.augment_punctuation_spacing(
                text, config.get("3_punctuation_spacing", "skip")
            )
            text = self.augment_double_newlines(
                text, config.get("4_double_newlines", "skip")
            )
            text = self.augment_quote_style(text, config.get("5_quote_style", "skip"))
            text = self.augment_tabs_spaces(text, config.get("6_tabs_spaces", "skip"))
            text = self.augment_double_spaces(
                text, config.get("7_double_spaces", "skip")
            )
            text = self.augment_brackets(text, config.get("8_brackets", "skip"))
            text = self.augment_contractions(text, config.get("9_contractions", "skip"))
            text = self.augment_numbers(text, config.get("11_numbers", "skip"))
            text = self.augment_synonyms(text, config.get("12_synonyms", "skip"))

            # Apply capitalization last (augmentation 10)
            text = self.augment_capitalization(
                text, config.get("10_capitalization", "skip")
            )

            return text
        except Exception as e:
            raise Exception(
                f"Error in apply_augmentations: {str(e)}\n{traceback.format_exc()}"
            )


def load_text_file(filepath: str) -> str:
    """Load text from UTF-8 file"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise Exception(
            f"Error loading file {filepath}: {str(e)}\n{traceback.format_exc()}"
        )


def save_text_file(text: str, filepath: str) -> None:
    """Save text to UTF-8 file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        raise Exception(
            f"Error saving file {filepath}: {str(e)}\n{traceback.format_exc()}"
        )


def generate_output_filename(input_path: str, output_dir: str, doc_number: int) -> str:
    """Generate timestamped output filename"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        filename = f"{base_name}_aug_{doc_number}_{timestamp}.txt"
        return os.path.join(output_dir, filename)
    except Exception as e:
        raise Exception(
            f"Error generating filename: {str(e)}\n{traceback.format_exc()}"
        )


def generate_text_suite_nlp_augment(
    input_text_path: str,
    output_suite_dir_path: str,
    num_documents: int = 5,
    randomization_level: float = 0.5,
    seed: Optional[int] = None,
    # Augmentation configuration
    aug_1_sentence_spacing: str = "random",
    aug_2_newline_spacing: str = "random",
    aug_3_punctuation_spacing: str = "random",
    aug_4_double_newlines: str = "random",
    aug_5_quote_style: str = "random",
    aug_6_tabs_spaces: str = "random",
    aug_7_double_spaces: str = "random",
    aug_8_brackets: str = "random",
    aug_9_contractions: str = "random",
    aug_10_capitalization: str = "skip",  # Document-level, often skip for readability
    aug_11_numbers: str = "random",
    aug_12_synonyms: str = "random",
) -> List[str]:
    """
    Generate a suite of augmented text documents

    Args:
        input_text_path: Path to input .txt file
        output_suite_dir_path: Directory for output files
        num_documents: Number of augmented documents to generate
        randomization_level: Probability (0.0-1.0) for random augmentations
        seed: Random seed for reproducibility
        aug_*: Configuration for each augmentation ('skip', 'random', 'all')

    Returns:
        List of output file paths

    Raises:
        Exception: For any errors during processing
    """
    try:
        # Load input text
        print(f"Loading input file: {input_text_path}")
        original_text = load_text_file(input_text_path)

        # Initialize augmenter
        augmenter = TextAugmenter(randomization_level=randomization_level, seed=seed)

        # Build configuration
        config = {
            "1_sentence_spacing": aug_1_sentence_spacing,
            "2_newline_spacing": aug_2_newline_spacing,
            "3_punctuation_spacing": aug_3_punctuation_spacing,
            "4_double_newlines": aug_4_double_newlines,
            "5_quote_style": aug_5_quote_style,
            "6_tabs_spaces": aug_6_tabs_spaces,
            "7_double_spaces": aug_7_double_spaces,
            "8_brackets": aug_8_brackets,
            "9_contractions": aug_9_contractions,
            "10_capitalization": aug_10_capitalization,
            "11_numbers": aug_11_numbers,
            "12_synonyms": aug_12_synonyms,
        }

        print(f"Configuration: {config}")
        print(f"Randomization level: {randomization_level}")
        print(f"Generating {num_documents} augmented documents...")

        output_files = []

        # Generate augmented documents
        for i in range(num_documents):
            print(f"Generating document {i + 1}/{num_documents}")

            # Apply augmentations
            augmented_text = augmenter.apply_augmentations(original_text, config)

            # Generate output filename
            output_path = generate_output_filename(
                input_text_path, output_suite_dir_path, i + 1
            )

            # Save augmented text
            save_text_file(augmented_text, output_path)
            output_files.append(output_path)

            print(f"Saved: {output_path}")

        print(f"Successfully generated {len(output_files)} augmented documents")
        return output_files

    except Exception as e:
        raise Exception(
            f"Error in generate_text_suite_nlp_augment: {str(e)}\n{traceback.format_exc()}"
        )


def get_directory_path():
    """
    Prompt user for directory path if not provided as command line argument.

    Returns:
        str: Valid directory path
    """
    if len(sys.argv) > 1:
        return sys.argv[1]

    while True:
        path = input("Enter directory path: ").strip()
        if os.path.isdir(path):
            return path
        print(f"Invalid directory: {path}")


def get_text_files(directory, extensions=None):
    """
    Get all text files from directory with specified extensions.

    Args:
        directory (str): Directory to search
        extensions (list): File extensions to include (default: ['.txt'])

    Returns:
        list: Sorted list of text files
    """
    if extensions is None:
        extensions = [".txt"]

    try:
        files = []
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in extensions):
                files.append(file)
        return sorted(files)
    except PermissionError:
        print(f"Permission denied accessing directory: {directory}")
        return []


def combine_text_files(
    directory=None, output_filename="corpus.txt", extensions=None, add_separators=True
):
    """
    Combine multiple text files from a directory into a single corpus file.

    Args:
        directory (str, optional): Directory containing text files. If None, prompts user.
        output_filename (str): Name of output file (default: "corpus.txt")
        extensions (list): File extensions to include (default: ['.txt'])
        add_separators (bool): Whether to add file separators (default: True)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get directory path
        if directory is None:
            directory = get_directory_path()

        if not os.path.isdir(directory):
            print(f"Directory not found: {directory}")
            return False

        # Get text files
        txt_files = get_text_files(directory, extensions)

        if not txt_files:
            print(f"No text files found in {directory}")
            return False

        # Ensure output file is not in the input list
        if output_filename in txt_files:
            txt_files.remove(output_filename)

        # Combine files
        output_path = os.path.join(directory, output_filename)
        files_processed = 0

        with open(output_path, "w", encoding="utf-8") as outfile:
            for txt_file in txt_files:
                file_path = os.path.join(directory, txt_file)
                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        if add_separators:
                            outfile.write(f"=== Content from {txt_file} ===\n")
                        content = infile.read().strip()
                        outfile.write(content)
                        if add_separators:
                            outfile.write("\n\n")
                        files_processed += 1
                        print(f"Processed: {txt_file}")

                except (UnicodeDecodeError, PermissionError) as e:
                    print(f"Error reading {txt_file}: {e}")
                    continue

        print(f"\nSuccess! Combined {files_processed} files into {output_path}")
        return True

    except Exception as e:
        print(f"Error combining files: {e}")
        return False


# import re
def extract_filename(path: str):
    # Define the regex pattern to capture the part before .txt
    pattern = r"(?<=/)[^/]+(?=\.txt\b)"

    # Search for the pattern in the path
    match = re.search(pattern, path)

    # Return the matched string if found, otherwise return None
    return match.group(0) if match else None


# # Example usage
# path = "there/here/file_name.txt"
# filename = extract_filename(path)
# print(filename)  # Output: file_name


def main():
    """Example usage of the text augmentation pipeline"""
    try:
        # Ask for input
        input_file = input("\nInput a path to a .txt file:\n//")
        file_name_only = extract_filename(input_file)

        print(f"test print {file_name_only}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"data/augmented_texts/{file_name_only}_{timestamp}/"
        # make sure exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory '{output_dir}' ensured to exist.")

        # Create sample input file if it doesn't exist
        if not os.path.exists(input_file):
            print("The Warning: File not found. Making demo file... one sec.")
            sample_text = """Hello world! This is a test document.

It contains multiple sentences. Some have contractions like don't, won't, and can't.
There are numbers like 3, 10, and 1st place.

"This has quotes," she said happily. The conversation was good.
They began to discuss the big problem (which was difficult).

He looked at the small issue	with tabs and spaces."""

            save_text_file(sample_text, input_file)
            print(f"Created sample input file: {input_file}")

        # Generate augmented documents
        output_files = generate_text_suite_nlp_augment(
            input_text_path=input_file,
            output_suite_dir_path=output_dir,
            num_documents=10,
            randomization_level=0.7,
            seed=42,
            # Enable most augmentations
            aug_1_sentence_spacing="random",
            aug_2_newline_spacing="random",
            aug_3_punctuation_spacing="random",
            aug_4_double_newlines="random",
            aug_5_quote_style="random",
            aug_6_tabs_spaces="all",
            aug_7_double_spaces="skip",  # Skip to avoid too much spacing
            aug_8_brackets="random",
            aug_9_contractions="random",
            aug_10_capitalization="skip",  # Skip for readability
            aug_11_numbers="random",
            aug_12_synonyms="random",
        )

        print("\nGenerated files:")
        for filepath in output_files:
            print(f"  - {filepath}")

        corpus_file_name = f"corpus_{file_name_only}_{timestamp}.txt"
        # Example usage with different extensions
        result_of_make_corpus = combine_text_files(
            output_dir,
            extensions=[".txt"],  # Include multiple file types
            output_filename=corpus_file_name,
        )
        print(f"Making corpus file...{output_dir}{corpus_file_name}")
        print(f"Magic 8-Ball, corpus creation succeded?? ...{result_of_make_corpus}!")

    except Exception as e:
        print(f"Error in main: {str(e)}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
