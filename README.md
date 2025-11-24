# MTG AI Card Set Generator
Generate a set of custom MTG cards with interesting, synergistic, and mutually reinforcing interactions.

Deployment on Hugging Face Spaces: [MTG AI Card Set Generator](https://huggingface.co/spaces/kestrel256/mtg-ai-card-set-generator)

# Magic: The Gathering Card Set Generator
## Some Requirements
To try out this MTG card generator, you'll need at the very least an OpenAI API key, since images are generated with OpenAI models via OpenAI's API. The default LLM for generating the card specifications (e.g., mana cost, power, toughness) is `gpt-4.1-nano-2025-04-14`, and if you want to use that LLM then that's another reason you'll also need that OpenAI API key. If you want to try some of the LLMs outside of OpenAI like Llama or Grok-4, you'll need an OpenRouter API key in addition to the OpenAI API key, since interactions with these other LLM models are routed through OpenRouter. The Python code assumes that OpenAI API key and the OpenRouter API key are stored in a local `.env` file.

After setting up your `OPENAI_API_KEY` as described above, click on the **Generate Card** button, perhaps with the default prompt `Please generate two new MTG cards.`.
If all goes well, the new card's information should appear in the text box in the upper right of the app.
And an A.I.-generated image should appear to the left of the card's information. Any errors should appear in a pink box that appears below the text box for the prompt.

To see this app in action, visit the deployment on Hugging Face Spaces: [MTG AI Card Set Generator](https://huggingface.co/spaces/kestrel256/mtg-ai-card-set-generator)

## Some History

This project is an extension of an earlier project that generated single MTG cards;
see the [MTG Card Generator](https://github.com/maisogjm/mtg-ai-card-generator) (Hugging Faces Spaces deployment: [MTG Card Generator](https://huggingface.co/spaces/kestrel256/mtg-ai-card-generator)).
The intent here is to generate *sets* of novel MTG cards that have interesting, synergistic, and mutually reinforcing interactions.
After the cards are generated, in addition to giving information on each card, an explanation is provided to describe the card interactions.

I have found that occasionally an explanation is generated that mentions cards that weren't actually generated,
despite having inserted a directive in the system prompt not to do so.
I suppose that a call to an LLM editor can be added to verify the explanation, but have refrained from doing so right now.
Perhaps later models such as ``gpt-5.1`` might be better at following negative prompts.
For now I am relying on a directive in the system prompt to nudge the LLM to generate an explanation that:
1. Mentions each card in the newly generated set of cards at least once by name.
2. Doesn't mention names of cards that aren't among the newly generated set of cards.
3. Clearly describes how the cards interact with each other, using their exact names when referring to them.


I have also found that the open-weight models occasionally give JSON validation errors, as if they occasionally have trouble conforming to the Pydantic structured output.
The `gpt-4o-mini` model seems to be much more reliable.

In this implementation, I have capped the number of cards generated at any one time to 12.
If you want to try generating larger sets, clone this repository and in your own copy increase the value of the parameter `MAX_NUM_CARDS`.

No A.I. artwork is generate in this project, since I thought that the point of this project was the interactions between cards,
and generating images is relatively costly and time-consuming compared to plain text.

## Suggestions
Try some of the following prompts.

- Please generate new MTG cards named "Gog" and "Magog".
- Please generate new MTG cards named 'Scylla' and 'Charybdis'.
- Please generate new MTG cards named "Rock" and "A Hard Place".
- Please generate new MTG cards named 'Frying Pan' and 'Fire'.
- Please generate new MTG cards named 'Rock', 'Paper', and 'Scissors'.
- Please generate new MTG cards that use Poison Counters in an interesting way.
- Please generate new MTG cards that all use the Saga mechanic. Have their Sagas interact in an interesting way.
- Please generate two new MTG cards named "Red in Tooth" and "Red in Claw" (a reference to "red in tooth and claw"; Tennyson), with an interesting interaction between the two cards.
- Please generate two new MTG cards named "Predator" and "Prey" (a reference to "red in tooth and claw"; Tennyson), with an interesting interaction between the two cards.
- Please generate two MTG cards that, together, demonstrate a Symmetrical Effect Breaker.
- Please generate three MTG cards named after the Three Musketeers.
- Please generate three MTG cards named after the three Greek Fates.
- Please generate three MTG cards named after the three Viking Norns.
- Please generate four new MTG cards based on the Four Seasons. Name this set of MTG Cards 'The Four Seasons'.
- Please generate four new MTG cards based on the Four Horsemen of the Apocalypse. Append "(Horseman of the Apocalypse)" to the name of all four cards.
- Please generate four new MTG cards based on the Marvel Comics superhero team "The Fantastic Four". Name these new cards after the members of that superhero team.
- Please generate four new MTG cards based on the Marvel Comics supervillain team "The U-Foes". Name these new cards after the members of that supervillain team.
- Please generate five new MTG cards that are all creatures that interact in an interesting way.
- Please generate five new MTG cards: one red card, one blue card, one green card, one white card, and one black card; with an interesting interaction between the five cards.
- Please generate 12 new MTG cards based on the 12 Months of the Year. Name this set of MTG Cards 'The 12 Months of the Year'.
- Please generate 12 new MTG cards based on the 12 Signs of the (western) Zodiac, e.g., Cancer, Leo, Capricorn, etc.. Name this set of MTG Cards '12 Signs of the Zodiac'.

In one experiment, I tried asking *Please generate new MTG cards based on the Kingdoms of Life.*.
I thought I would get one card for each of kingdoms recognized in the modern 6-kingdom model.
But instead I got a set of green cards that had a sylvan theme, with names like **Verdant Guardian**, **Sylvan Sentinel's Vigil**, and **Forest's Bounty**.
I found I had to be more specific, like this:

```
In the modern 6-kingdom model, the Kingdoms of Life are:
(1) Animalia
(2) Plantae
(3) Fungi
(4) Protista
(5) Bacteria (Eubacteria)
(6) Archaea (Archaebacteria)
Please generate a set of 6 new MTG cards based on the modern 6-kingdom model. Name this set of cards "The Kingdoms of Life"```
