#!/usr/bin/env python
# coding: utf-8

import os
import json
import re
from typing import List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

# Pydantic schemas (your "original" structured outputs)
from mtg_schemas import MTGCard, MTGCardList, YesNoName, YesNoNameList, MTGNameOnly

# ============================================================================
# ENV + CLIENT SETUP
# ============================================================================

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeWarning("No usable OpenAI API key found in environment.")
openai_client = OpenAI(api_key=openai_api_key)

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    print("⚠ No usable OpenRouter API key found in environment. OpenRouter models will be disabled.")

openrouter_url = "https://openrouter.ai/api/v1"

clients = {
    "openai": openai_client,
    "openrouter": OpenAI(api_key=openrouter_api_key, base_url=openrouter_url) if openrouter_api_key else None,
}

# Model mappings
card_models = {
    "gpt-5.1": "openai",
    "gpt-4.1": "openai",
    "gpt-4.1-nano-2025-04-14": "openai",
    "gpt-4o-mini": "openai",
    "x-ai/grok-4-fast": "openrouter",
    "deepseek/deepseek-chat-v3.1": "openrouter",
    "meta-llama/llama-3.2-3b-instruct": "openrouter",
    "qwen/qwen3-vl-30b-a3b-instruct": "openrouter"
}

extract_models = card_models.copy()

# Filter out OpenRouter models if API key is not available
if not openrouter_api_key:
    card_models_available = {k: v for k, v in card_models.items() if v != "openrouter"}
    extract_models_available = {k: v for k, v in extract_models.items() if v != "openrouter"}
else:
    card_models_available = card_models
    extract_models_available = extract_models

# Card generation limits
MAX_NUM_CARDS = 60
MIN_NUM_CARDS = 2

system_prompt = (
    f"""You are a creative and imaginative designer of cards for the collectible/trading card game
    Magic: The Gathering. Respond only with a single JSON object that matches the schema.
    If the card has a non-null mana cost, try to match the mana cost with the potency of the card.
    I.e., creatures with high Power and/or Toughness should tend to cost more; and instants that
    cause more damage should tend to cost more. Keep in mind that Lands typically do not cost mana.
    Most (82%) MTG cards have a NaN (missing) Supertype value; the most common non-missing Supertype value is 'Legendary',
    accounting for 14% of all cards. It is OK to generate a card with a missing/None Supertype value!
    In fact, if the card is a common and/or low-powered creature or artifact, or if it isn't a creature or artifact to begin with,
    it might be best to just have Supertype with a value of None (missing).
    The top six most common Type values are (in decreasing order): Creature, Land, Instant, Sorcery, Enchantment, and Artifact.
    Creatures are the most common Type value, accounting for about 44% of all cards.
    Land cards are the next most common Type.
    A large proportion of (38%) cards have a missing Subtype.
    
    IMPORTANT: When generating cards, you must generate between {MIN_NUM_CARDS} and {MAX_NUM_CARDS} cards (inclusive) that have interesting, synergistic, and mutually reinforcing interactions.
    The minimum is {MIN_NUM_CARDS} cards because we are interested in interactions between cards.
    The maximum is {MAX_NUM_CARDS} cards. If the user requests more than {MAX_NUM_CARDS} cards or fewer than {MIN_NUM_CARDS} cards,
    or uses vague terms like "many" or "several", generate exactly {MAX_NUM_CARDS} cards if they ask for more, or {MIN_NUM_CARDS} cards if they ask for fewer or don't specify a number.
    
    CRITICAL: The cards you generate MUST have interesting, synergistic, and mutually reinforcing interactions. They should work together in meaningful ways, not just be individually interesting cards. The explanation field must describe these interactions clearly. Do not generate a set of cards that are merely individually interesting without interactions between them.
    
    SET NAME REQUIREMENT: You must provide a short descriptive name for the newly generated set of MTG cards in the Name field of the MTGCardList. This name should capture the theme, mechanic, or concept that ties the cards together.
    
    EXPLANATION REQUIREMENTS: The explanation field must:
    1. Mention each card in the MTGCardList at least once by its exact name.
    2. Only mention card names that actually exist in the cards array of the MTGCardList. Do not reference card names that are not in the generated set.
    3. Clearly describe how the cards interact with each other, using their exact names when referring to them."""
)

# ============================================================================
# CARD NAME DATABASE
# ============================================================================

try:
    CARD_NAMES_FILE = "cardnames.txt"
    with open(CARD_NAMES_FILE, "r", encoding="utf-8", errors="replace") as f:
        card_names = set(f.read().splitlines())
    print(f"✓ Loaded {len(card_names)} existing card names")
except FileNotFoundError:
    print("⚠ Card names file not found, starting with empty set")
    card_names = set()

# ============================================================================
# HELPER FUNCTIONS (ported from mtg_gradio_v9, adapted to Streamlit)
# ============================================================================

def get_client(model_name: str, model_dict: dict) -> OpenAI:
    """Get the appropriate client for a given model."""
    provider = model_dict.get(model_name)
    if provider is None:
        raise ValueError(f"Unknown model: {model_name}")
    client = clients.get(provider)
    if client is None:
        raise ValueError(
            f"Client not configured for provider: {provider}. "
            "Check that the corresponding API key is set."
        )
    return client


def ExtractCardCount(txt: str, extract_model: str) -> int:
    """Extract the number of cards requested from user text. Returns 0 if not specified."""
    # First, check for implicit count via named cards
    named_cards = ExtractNameIfAny(txt, extract_model)
    initial_count_from_named = len(named_cards) if named_cards else 0
    
    # Initialize initial_count
    initial_count = 0
    
    txt_lower = txt.lower()
    
    # Check for quantity words that indicate multiple cards (before checking singular patterns)
    # These should take precedence over singular patterns
    quantity_patterns = [
        r'\bmany\s+cards?\b',
        r'\bmultiple\s+cards?\b',
        r'\bseveral\s+cards?\b',
        r'\ba\s+set\s+of\s+many\s+cards?\b',
        r'\ba\s+set\s+of\s+cards?\b',
        r'\bgenerate\s+many\s+cards?\b',
        r'\bcreate\s+many\s+cards?\b',
        r'\bmake\s+many\s+cards?\b',
    ]
    for pattern in quantity_patterns:
        if re.search(pattern, txt_lower):
            # If quantity words are found, skip singular pattern check and go to LLM extraction
            # The LLM will handle interpreting "many" appropriately
            break
    else:
        # Only check for singular forms if no quantity words were found
        # Patterns that indicate singular (1 card): "a card", "an MTG card", "one card", "a new card", etc.
        # But only match if it's the main request, not in a sub-clause like "Include a card named..."
        # Using word boundaries and checking for request verbs before the pattern
        singular_patterns = [
            r'\b(?:generate|create|make|please\s+generate|please\s+create|please\s+make)\s+(?:a|an|one)\s+(?:new\s+)?(?:mtg\s+)?card\b',  # "generate a new MTG card", etc.
            r'^(?:please\s+)?(?:generate|create|make)\s+(?:a|an|one)\s+(?:new\s+)?(?:mtg\s+)?card\b',  # Start of text: "generate a card"
            r'\ba\s+new\s+mtg\s+card\b',  # "a new MTG card"
            r'\ba\s+new\s+card\b',  # "a new card"
            r'\ba\s+mtg\s+card\b',  # "a MTG card"
            r'\ban\s+new\s+mtg\s+card\b',  # "an new MTG card"
            r'\ban\s+new\s+card\b',  # "an new card"
            r'\ban\s+mtg\s+card\b',  # "an MTG card"
            r'\bone\s+new\s+mtg\s+card\b',  # "one new MTG card"
            r'\bone\s+new\s+card\b',  # "one new card"
            r'\bone\s+mtg\s+card\b',  # "one MTG card"
            r'\bone\s+card\b',  # "one card"
        ]
        for pattern in singular_patterns:
            if re.search(pattern, txt_lower):
                initial_count = 1
                break
    
    # Then check for explicit number in the text
    client = get_client(extract_model, extract_models)
    
    msg = f"""Here is some text.
<TEXT>
{txt}
</TEXT>
Extract the number of cards requested in this text. For example:
- "generate five cards" → 5
- "create 3 MTG cards" → 3
- "generate two cards" → 2
- "generate many cards" → {MAX_NUM_CARDS} (use maximum when "many", "multiple", "several" is specified)
- "create a set of many cards" → {MAX_NUM_CARDS} (use maximum when "many", "multiple", "several" is specified)
- "create a card" → 1
- "generate cards" (no number specified) → 0

Important: If the text says "many", "multiple", "several", or similar quantity words indicating multiple cards, return {MAX_NUM_CARDS}.
If the text explicitly requests a specific number, return that number.
If the text requests a single card (using "a card", "one card", etc.), return 1.
If no number is specified and no quantity words are used, return 0.

Respond with ONLY the number (0 if not specified), nothing else.
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": msg},
    ]
    
    try:
        # Use a simple completion to extract the number
        completion = client.chat.completions.create(
            model=extract_model,
            messages=messages,
            temperature=0.2,
            max_tokens=10,
        )
        
        response = completion.choices[0].message.content.strip()
        # Try to extract number from response
        numbers = re.findall(r'\d+', response)
        initial_count = 0
        if numbers:
            initial_count = int(numbers[0])
        else:
            # Check for word numbers
            word_to_num = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
            }
            response_lower = response.lower()
            for word, num in word_to_num.items():
                if word in response_lower:
                    initial_count = num
                    break
        
        # Final check: Have LLM count cards by analyzing full context
        # This trumps previous methods if it finds a higher count
        count_msg = f"""Here is a request to generate a set of MTG cards.

<CONTEXT>
{txt}
</CONTEXT>

Please count the number of cards requested. Consider:
- Explicit numbers mentioned (e.g., "5 cards", "three cards")
- Quantity words like "many", "multiple", "several" (these indicate multiple cards)
- Lists of specific card names or types requested
- Phrases like "a set of cards", "generate cards", etc.

If the request asks for "many" cards or lists many specific cards/types, count all the cards that are being requested.
If no specific number is given and no quantity words are used, return 0.

Respond with ONLY the number, nothing else."""
        
        try:
            count_completion = client.chat.completions.create(
                model=extract_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that counts the number of items requested in text."},
                    {"role": "user", "content": count_msg}
                ],
                temperature=0.2,
                max_tokens=10,
            )
            
            count_response = count_completion.choices[0].message.content.strip()
            # Try to extract number from response
            count_numbers = re.findall(r'\d+', count_response)
            context_count = 0
            if count_numbers:
                context_count = int(count_numbers[0])
            else:
                # Check for word numbers
                word_to_num = {
                    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
                }
                count_response_lower = count_response.lower()
                for word, num in word_to_num.items():
                    if word in count_response_lower:
                        context_count = num
                        break
            
            # Use the context count if it's greater than the initial count
            # Also compare with named cards count
            max_count = max(initial_count, initial_count_from_named)
            if context_count > max_count:
                return context_count
            else:
                return max_count
                
        except Exception as e:
            print(f"Warning: Context-based card count extraction failed: {e}")
            # Fall back to initial count
            return initial_count
            
    except Exception as e:
        print(f"Warning: Card count extraction failed: {e}")
        return 0


def ExtractNameIfAny(txt: str, extract_model: str) -> List[str]:
    """Extract all card names from user text if specified. Returns a list of card names, or empty list if none found."""
    client = get_client(extract_model, extract_models)

    msg = f"""Here is some text.
<TEXT>
{txt}
</TEXT>
If the text includes a request to specify the name(s) of one or more items (e.g., cards), extract ALL the specified names.
For example, if the text says "create cards named 'Test' and 'Example'", extract both 'Test' and 'Example'.
If the text says "create a card named 'Test'", extract 'Test'.
If no specific names are requested, return an empty list.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": msg},
    ]

    try:
        completion = client.beta.chat.completions.parse(
            model=extract_model,
            messages=messages,
            response_format=YesNoNameList,
            temperature=0.2,
        )

        parsed = completion.choices[0].message.parsed
        # Extract all names from the list where YesNo == "Yes"
        card_names_list = [
            item.Name for item in parsed.items 
            if item.YesNo == "Yes" and item.Name and item.Name.strip()
        ]
        return card_names_list
    except Exception as e:
        print(f"Warning: Name extraction failed: {e}")
        return []


def generate_unique_name_for_card(parsed_card, used_names, extract_model):
    """
    Ask the LLM to generate a new, unique card name,
    consistent with the card's other attributes.
    """

    client = get_client(extract_model, extract_models)

    card_info = json.dumps(parsed_card.model_dump(), indent=2)

    prompt = f"""
You must generate a NEW, UNIQUE name for this Magic: The Gathering card.

Here are all of the card's attributes except the name:
<card>
{card_info}
</card>

Requirements:
- Do NOT reuse any name in the following list:
{list(used_names)}
- The new name MUST NOT match any existing card name.
- The new name MUST match the style, color identity, type, subtype, flavor,
  and general theme of the provided card.
- Respond ONLY with a single JSON object containing the field "Name".
"""

    completion = client.beta.chat.completions.parse(
        model=extract_model,
        messages=[{"role": "user", "content": prompt}],
        response_format=MTGNameOnly,
        temperature=0.4,     # low temperature is best for names
    )

    return completion.choices[0].message.parsed.Name


def clean_explanation_quotes(parsed_list: MTGCardList, card_model: str) -> str:
    """
    Use the LLM to remove quotes (single or double) around card names and set name in the explanation.
    Returns the cleaned explanation, or the original if no cleaning was needed.
    """
    # Extract card names from the parsed list
    card_names = [card.Name for card in parsed_list.cards]
    
    # Include the set name as well
    all_names = []
    if parsed_list.Name:
        all_names.append(parsed_list.Name)
    all_names.extend(card_names)
    
    if not all_names:
        return parsed_list.explanation
    
    # Create a prompt for the LLM to clean the explanation
    names_list = "\n".join([f"- {name}" for name in all_names])
    
    prompt = f"""You are given an explanation text about Magic: The Gathering cards and a list of names (set name and card names).

Names in the set (including the set name and all card names):
{names_list}

Explanation text:
{parsed_list.explanation}

Task: Remove all single quotes (') and double quotes (") that enclose any of these names in the explanation text. 
All names (set name and card names) should appear without any quotes around them. Do not change any other part of the text.
If a name appears with quotes like "Name" or 'Name', change it to just Name (no quotes).
If the explanation already has no quotes around these names, return it unchanged.

Return ONLY the cleaned explanation text, nothing else."""

    try:
        client = get_client(card_model, card_models)
        
        completion = client.chat.completions.create(
            model=card_model,
            messages=[
                {"role": "system", "content": "You are a text editor that removes quotes around card names in explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent cleaning
            max_tokens=2000,
        )
        
        cleaned_explanation = completion.choices[0].message.content.strip()
        
        # If the cleaned explanation is empty or seems wrong, return original
        if not cleaned_explanation or len(cleaned_explanation) < len(parsed_list.explanation) * 0.5:
            print("⚠ Explanation cleaning may have failed, using original")
            return parsed_list.explanation
        
        return cleaned_explanation
        
    except Exception as e:
        print(f"⚠ Failed to clean explanation quotes: {e}")
        return parsed_list.explanation  # Return original on error


def CreateCard(msg: str, card_model: str, extract_model: str, temp: float):
    """Main function to create MTG cards (can be multiple)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": msg},
    ]

    # Check if any requested names already exist
    requested_names = ExtractNameIfAny(msg, extract_model)
    if requested_names:
            duplicate_names = [name for name in requested_names if name in card_names]
            if duplicate_names:
                names_str = ", ".join([f"'{name}'" for name in duplicate_names])
                return (
                    f"❌ Sorry, the following name(s) have already been used: {names_str}. "
                    "Please select other names or leave names unspecified.",
                    "",
                    "",
                    []
                )

    # Try to create cards (with retries for duplicate names)
    max_card_attempts = 5     # regenerate cards up to 5 times
    for attempt in range(max_card_attempts):
        try:
            client = get_client(card_model, card_models)

            completion = client.beta.chat.completions.parse(
                model=card_model,
                messages=messages,
                response_format=MTGCardList,
                temperature=temp,
            )

            parsed_list: MTGCardList = completion.choices[0].message.parsed
            cards = parsed_list.cards
            set_name = parsed_list.Name
            # Get original explanation before any renaming
            original_explanation = parsed_list.explanation
            
            # Track names in the current batch to ensure uniqueness within the batch
            batch_names = set()
            # Track name mappings for updating explanation (old_name -> new_name)
            name_mappings = {}
            
            # Check if any generated card names are duplicates and regenerate them
            for i, card in enumerate(cards):
                original_name = card.Name
                # Check if name is duplicate in existing card database
                if card.Name in card_names:
                    # Keep trying to generate a unique name until one is found
                    max_name_attempts = 10  # Try up to 10 times to generate a unique name
                    name_found = False
                    for name_attempt in range(max_name_attempts):
                        try:
                            # Combine existing names and current batch names to avoid duplicates
                            all_used_names = card_names | batch_names
                            new_name = generate_unique_name_for_card(
                                parsed_card=card,
                                used_names=all_used_names,
                                extract_model=extract_model,
                            )
                            # Verify the regenerated name is actually unique
                            if new_name in card_names:
                                print(f"⚠ Regenerated name '{new_name}' (attempt {name_attempt + 1}) is still a duplicate in card_names, retrying...")
                                continue  # Try again
                            if new_name in batch_names:
                                print(f"⚠ Regenerated name '{new_name}' (attempt {name_attempt + 1}) conflicts with another card in this batch, retrying...")
                                continue  # Try again
                            # Unique name found!
                            card.Name = new_name
                            # Track the name change
                            if original_name != new_name:
                                name_mappings[original_name] = new_name
                            name_found = True
                            break
                        except Exception as e:
                            print(f"⚠ Failed to generate replacement name (attempt {name_attempt + 1}): {e}")
                            if name_attempt == max_name_attempts - 1:
                                # Last attempt failed, give up and retry entire card generation
                                raise ValueError("Failed to generate unique name after multiple attempts")
                            continue  # Try again
                    
                    if not name_found:
                        raise ValueError("Failed to generate unique name after multiple attempts")
                
                # Check if name is duplicate within the current batch
                if card.Name in batch_names:
                    # Keep trying to generate a unique name until one is found
                    max_name_attempts = 10  # Try up to 10 times to generate a unique name
                    name_found = False
                    for name_attempt in range(max_name_attempts):
                        try:
                            # Combine existing names and current batch names to avoid duplicates
                            all_used_names = card_names | batch_names
                            new_name = generate_unique_name_for_card(
                                parsed_card=card,
                                used_names=all_used_names,
                                extract_model=extract_model,
                            )
                            # Verify the regenerated name is actually unique
                            if new_name in card_names:
                                print(f"⚠ Regenerated name '{new_name}' (attempt {name_attempt + 1}) is still a duplicate in card_names, retrying...")
                                continue  # Try again
                            if new_name in batch_names:
                                print(f"⚠ Regenerated name '{new_name}' (attempt {name_attempt + 1}) conflicts with another card in this batch, retrying...")
                                continue  # Try again
                            # Unique name found!
                            card.Name = new_name
                            # Track the name change
                            if original_name != new_name:
                                name_mappings[original_name] = new_name
                            name_found = True
                            break
                        except Exception as e:
                            print(f"⚠ Failed to generate replacement name for batch duplicate (attempt {name_attempt + 1}): {e}")
                            if name_attempt == max_name_attempts - 1:
                                # Last attempt failed, give up and retry entire card generation
                                raise ValueError("Failed to generate unique name after multiple attempts")
                            continue  # Try again
                    
                    if not name_found:
                        raise ValueError("Failed to generate unique name after multiple attempts")
                
                # Add the (now unique) name to batch tracking
                batch_names.add(card.Name)
            
            # Update explanation with new card names if any cards were renamed
            if name_mappings:
                explanation = original_explanation
                # Replace old names with new names in the explanation
                # Sort by length (longest first) to handle cases where one name contains another
                sorted_mappings = sorted(name_mappings.items(), key=lambda x: len(x[0]), reverse=True)
                for old_name, new_name in sorted_mappings:
                    # Use word boundaries to replace whole words only
                    escaped_old_name = re.escape(old_name)
                    pattern = r'\b' + escaped_old_name + r'\b'
                    explanation = re.sub(pattern, new_name, explanation)
                # Update parsed_list.explanation so clean_explanation_quotes uses the updated version
                parsed_list.explanation = explanation
            else:
                explanation = original_explanation
            
            # Clean the explanation to remove quotes around card names (after updating names)
            # Note: parsed_list.cards already have the updated names since they're the same objects
            explanation = clean_explanation_quotes(parsed_list, card_model)

            # Success - format all cards (without explanation)
            formatted_cards = []
            for i, card in enumerate(cards):
                card_json = json.dumps(card.model_dump(), indent=4, ensure_ascii=False)
                pretty_text = format_card_info(card_json)
                # Add horizontal rule before each card except the first
                separator = "\n---\n" if i > 0 else ""
                formatted_cards.append(f"{separator}GENERATED CARD #{i+1}:\n\n{pretty_text}\n")
            
            cards_text = "\n".join(formatted_cards)
            
            # Extract card names for bolding in explanation
            card_names_list = [card.Name for card in cards]
            
            # Return cards text, set name, explanation, and card names separately
            return cards_text, set_name, explanation, card_names_list

        except ValidationError as ve:
            return f"❌ Validation Error: {ve}", "", "", []

        except Exception as e:
            print(f"❌ Unexpected error while generating cards: {e}")
            continue

    return "❌ Failed to generate safe cards after several attempts.", "", "", []

def bold_card_names(text: str, card_names: List[str], set_name: str = "") -> str:
    """
    Make all card names and set name in the text bold using markdown syntax.
    Removes quotes around names when bolding them.
    Uses word boundaries to avoid partial matches.
    """
    result = text
    
    # Combine set name with card names, prioritizing set name
    all_names = []
    if set_name:
        all_names.append(set_name)
    if card_names:
        all_names.extend(card_names)
    
    if not all_names:
        return result
    
    # Sort by length (longest first) to handle cases where one name contains another
    sorted_names = sorted(all_names, key=len, reverse=True)
    
    for name in sorted_names:
        if name:  # Skip empty names
            # Escape special regex characters in the name
            escaped_name = re.escape(name)
            
            # Pattern 1: Name with double quotes - remove quotes and bold
            # Match: "name" ensuring name is a complete word (not part of a larger word)
            # Use word boundaries around the name itself
            pattern_double_quotes = r'"\b' + escaped_name + r'\b"'
            result = re.sub(pattern_double_quotes, f'**{name}**', result)
            
            # Pattern 2: Name with single quotes - remove quotes and bold
            # Match: 'name' ensuring name is a complete word (not part of a larger word)
            pattern_single_quotes = r"'\b" + escaped_name + r"\b'"
            result = re.sub(pattern_single_quotes, f'**{name}**', result)
            
            # Pattern 3: Name without quotes - just bold it
            # Use negative lookbehind/lookahead to avoid matching already-bolded text
            # Match word boundary, then name, then word boundary, but not if surrounded by **
            pattern_no_quotes = r'(?<!\*)\b' + escaped_name + r'\b(?!\*)'
            result = re.sub(pattern_no_quotes, f'**{name}**', result)
    
    return result

# pretty-printing function
def format_card_info(raw_json: str) -> str:
    """
    Transform the raw JSON dump into a nicer human-readable block:
    - Remove quotes and commas
    - Rename keys (OriginalText → Original Text, etc.)
    - Flatten Colors list into comma-separated string
    - Convert None → None
    """
    try:
        data = json.loads(raw_json)
    except Exception:
        return raw_json  # fallback
    
    # Key renaming map
    rename = {
        "OriginalText": "Original Text",
        "FlavorText": "Flavor Text",
        "ManaCost": "Mana Cost",
    }

    # Build formatted lines
    lines = []

    for key, value in data.items():
        # Rename key if applicable
        pretty_key = rename.get(key, key)

        # Process Colors list
        if key == "Colors":
            if isinstance(value, list):
                value_str = ", ".join(value)
            else:
                value_str = "None"
        # Process None
        elif value is None:
            value_str = "None"
        else:
            value_str = str(value)

        # Remove quotes from value_str (clean but safe)
        value_str = value_str.replace('"', "")

        # Format each field on its own line with HTML line break for single spacing
        lines.append(f"{pretty_key}: {value_str}")

    # Join with HTML line breaks to ensure single line spacing in markdown
    return "<br>".join(lines)


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="MTG Card Set Generator", layout="wide")

st.title("🎴 MTG Card Set Generator")
st.markdown(
    "Generate a set of custom MTG cards with interesting, synergistic, and mutually reinforcing interactions."
)

# Initialize session state for card info and explanation
if "card_info" not in st.session_state:
    st.session_state["card_info"] = ""
if "card_explanation" not in st.session_state:
    st.session_state["card_explanation"] = ""
if "card_set_name" not in st.session_state:
    st.session_state["card_set_name"] = ""
if "card_names_list" not in st.session_state:
    st.session_state["card_names_list"] = []

# Main layout with two columns
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### User Prompt")
    user_prompt = st.text_area(
        "Card Description",
        value="Please generate two new MTG cards.",
        height=120,
    )
    
    generate_btn = st.button("Generate Cards", type="primary", use_container_width=True)
    
    # Error message placeholder
    error_placeholder = st.empty()
    
    # Settings section (vertically)
    st.markdown("---")
    st.markdown("#### ⚙️ Settings")
    card_model_choice = st.selectbox(
        "Card Generation Model",
        options=list(card_models_available.keys()),
        index=list(card_models_available.keys()).index("gpt-4o-mini") if "gpt-4o-mini" in card_models_available else 0,
    )
    extract_model_choice = st.selectbox(
        "Name Extraction Model",
        options=list(extract_models_available.keys()),
        index=list(extract_models_available.keys()).index("gpt-4.1-nano-2025-04-14") if "gpt-4.1-nano-2025-04-14" in extract_models_available else 0,
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.8, step=0.1)

with col_right:
    # Explanation widget at the top
    st.markdown("#### Explanation")
    if st.session_state["card_explanation"]:
        # Display set name as H2 heading if available
        if st.session_state["card_set_name"]:
            st.markdown(f"## {st.session_state['card_set_name']}")
        # Bold card names and set name in the explanation
        explanation_with_bold = bold_card_names(
            st.session_state["card_explanation"],
            st.session_state["card_names_list"],
            st.session_state.get("card_set_name", "")
        )
        st.markdown(explanation_with_bold)
    else:
        st.info("Explanation of card interactions will appear here after generation.")
    
    st.markdown("---")
    
    # Card Information below
    st.markdown("#### Card Information")
    if st.session_state["card_info"]:
        raw = st.session_state["card_info"]

        # Check if text is already formatted (starts with card separator or horizontal rule)
        if raw.startswith("########################") or raw.startswith("---"):
            # Replace hash marks with horizontal rules if present
            pretty = raw.replace("########################", "---")
        else:
            # Remove optional prefix like "✓ Generated Card:" 
            if raw.startswith("✓ Generated Card:"):
                # Split on the first '{'
                _, json_part = raw.split("{", 1)
                raw_json = "{" + json_part.strip()
            else:
                raw_json = raw

            pretty = format_card_info(raw_json)
        
        st.markdown(pretty, unsafe_allow_html=True)
    else:
        st.info("Card details will appear here after generation.")

# On submit
if generate_btn:
    if not user_prompt.strip():
        error_placeholder.warning("Please enter a description for the cards.")
    else:
        # Check the number of cards requested
        try:
            requested_count = ExtractCardCount(user_prompt.strip(), extract_model_choice)
            if requested_count > MAX_NUM_CARDS:
                error_placeholder.error(
                    f"❌ At most {MAX_NUM_CARDS} cards can be generated at any one time. "
                    f"Your request asks for {requested_count} cards. Please reduce the number and try again."
                )
            elif requested_count == 1:
                error_placeholder.error(
                    f"❌ A minimum of {MIN_NUM_CARDS} cards must be requested, since cards with an interesting interaction between them will be generated. "
                    f"Your request asks for 1 card. Please request at least {MIN_NUM_CARDS} cards."
                )
            elif requested_count > 0 and requested_count < MIN_NUM_CARDS:
                error_placeholder.error(
                    f"❌ A minimum of {MIN_NUM_CARDS} cards must be requested, since cards with an interesting interaction between them will be generated. "
                    f"Your request asks for {requested_count} cards. Please request at least {MIN_NUM_CARDS} cards."
                )
            else:
                try:
                    with st.spinner("Generating cards..."):
                        card_info, set_name, explanation, card_names_list = CreateCard(
                            msg=user_prompt.strip(),
                            card_model=card_model_choice,
                            extract_model=extract_model_choice,
                            temp=temperature,
                        )

                    if card_info.startswith("❌"):
                        error_placeholder.error(card_info)
                    else:
                        st.session_state["card_info"] = card_info
                        st.session_state["card_set_name"] = set_name
                        st.session_state["card_explanation"] = explanation
                        st.session_state["card_names_list"] = card_names_list
                        st.success("Cards generated successfully!")
                        st.rerun()   # ← Force UI to update immediately
                except Exception as e:
                    error_placeholder.error(f"Unexpected error: {e}")
        except Exception as e:
            error_placeholder.error(f"Error checking card count: {e}")

