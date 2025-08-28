class SelectionPrompts:
    GUIDELINES = f"""
    ## Overview 
You are an assistant for a fact-checking and credibility verification system. You will be given a sentence from a news or web article, along
with surrounding context and the paragraph it comes from. Your task is to determine whether the sentence contains a verifiable and checkable information that can be
decomposed into individual claims. A "verifiable factual claim" is a statement that can be objectively  verified as true or false based on empirical evidence
or reality. The sentence must be sufficiently specific, meaning a fact-checker should have a clear idea of what evidence they would need
to support or refute it. You are not being asked whether the sentence is true or false. Your task is to assess whether the sentence has the
potential to be verified, based only on the sentence itself and the context.

Decomposed claim MUST:
- Contain one discrete factual proposition that can be independently verified
- Be specific enough to check against a source.
- Be fully self-contained, no pronouns like "he," "it," "they," or vague references like "the event," "this," "that".
- Include attribution if the original statement involves someone saying or asserting something.
- Be substantive enough to warrant separate verification and avoid trivially obvious claims.

A decomposed claim MUST NOT:
- Contain opinions, speculation, beliefs, or suggestions, e.g., "should be," "might lead to…".
- Be redundantly granular, e.g., don't split "dangerous driving causing bodily harm" into separate claims.
- Create multiple claims that say essentially the same thing with slight variations.
- Depend on context that is not present in the claim itself.
- Rewrite or simplify in ways that lose the original meaning.

Avoid over-decomposition:
- Don't split compound descriptors that form a single concept: "dangerous driving causing bodily harm" is one charge.
- Don't create multiple versions of the same claim: "X expressed hope", "X expressed hope for Y", "X expressed hope for Y at time Z". 
- Don't separate time/location from the main action unless they're independently verifiablefactual claim.

Here is what to consider:
- DO NOT include generic statements, opinions, speculations, interpretations, or recommendations as verifiable claims.
- Context is crucial. If the sentence is vague or ambiguous on its own, check whether the surrounding paragraph clarifies it.
- Ignore citations or references like "[1]" or "according to experts.".
- If a claim is made by a named entity e.g., “the health ministry stated”, in most cases it should be verifiable if the attribution is included.
- Introductory or summary sentences that just set up or wrap up other content e.g., "There are several important trends to consider.", usually do not contain verifiable claims.
- Even if part of the sentence is unverifiable, you may still extract the verifiable part and discard the unverifiable part.
- Do NOT infer missing information. Only rewrite what is explicitly present in the sentence or clarified in the paragraph/context.
- Keep rewrites as short and close to the original as possible, while removing unverifiable parts.
- Do not paraphrase a sentence unless you're removing unverifiable content. If the sentence is already concise and specific, return it as-is.

## Examples
Sentence: “California and New York implemented incentives for renewable energy adoption, highlighting the broader importance of sustainability.”
→ Label: Verifiable
→ Rewritten: “California and New York implemented incentives for renewable energy adoption.”

Sentence: “AI could lead to major breakthroughs in education.”
→ Label: Not Verifiable
→ Rewritten: “None”

Sentence: “John is the CEO of Company X.”
→ Label: Verifiable
→ Rewritten: “John is the CEO of Company X.”

Sentence: "Many experts believe climate change is the most serious threat to humanity."
→ Label: Not Verifiable  
→ Rewritten: None

Response
Return the following:
- Original sentence
- One of the following verifiability label
   - verifiable - contains at least one verifiable factual claim
   - not_verifiable - contains no verifiable factual content. Use "not_verifiable" for opinions, suggestions, emotional appeals, or sentences that cannot be objectively checked.
   - unclear - ambiguous or borderline. Use "unclear" only when the sentence may be verifiable with more context, but is ambiguous or vague as-is
- Rewritten sentence (optionally)
   - If the original sentence mixes verifiable and unverifiable content, return only the verifiable part.
   - If no rewrite is needed, return the original sentence as the rewritten_sentence.
   - If the sentence is unclear but contains a potentially verifiable portion, try to extract that part and return it as the rewritten_sentence.
   - If ambiguity prevents any confident rewrite, return "none".

Return the following content in JSON format:
{{
  "original_sentence": "<insert the original sentence>",
  "verification_label": "<one of the mentioned labels - verifiable, not_verifiable or unclear>",
  "rewritten_sentence": "<rewritten version if possible, the original sentence if no rewrite is needed, or 'none' if the sentence contains no verifiable content>"
}}

Use this format consistently for each sentence evaluated.
    """

    @staticmethod
    def get_prompt(sentence: str, context: str, paragraph: str) -> str:
        return f"""
You are an assistant for a fact-checking system. You will be given a sentence from a news or web article, along with the full paragraph 
it belongs to, and the surrounding sentences (if available). Your task is to determine whether the sentence contains at least one 
specific and verifiable factual proposition. If so, return a cleaned-up version that contains only the verifiable information.

Definitions and Rules:
- A verifiable proposition is a statement that can be objectively verified as true or false based on empirical evidence or reality.
- It does not matter whether the proposition is true or false.
- It does *not matter whether the proposition is important or relevant.
- Do not include: opinions, vague generalizations, recommendations, speculation, or interpretations.
- Do not rely* on citations, sources, or references like "[1]" to determine verifiability.
- Context matters. If the sentence is vague on its own, use the surrounding paragraph and nearby sentences to clarify it.
- If the sentence is an introductory or summary statement e.g., “There are several trends to note…”, it is not verifiable unless it includes a specific factual assertion.
- If the sentence contains both verifiable and unverifiable parts, rewrite it to retain only the verifiable content.
- Do NOT infer missing information. Only rewrite what is explicitly present in the sentence or clarified in the paragraph/context.
- Keep rewrites as short and close to the original as possible, while removing unverifiable parts.
- Only rewrite a sentence if the ambiguity is clearly resolvable based on evidence from the paragraph or context. If unsure, return "none".
- If the sentence is already self-contained and clear, return "remains_unchanged". Avoid unnecessary paraphrasing.
- If the sentence includes attribution e.g. “according to the ministry”, preserve it in the rewritten version.

Sentences labeled "unclear" may be passed to a disambiguation step. If the ambiguity is referential or structural and could potentially 
be resolved using context, preserve the wording and return the full sentence, not "none", so disambiguation can try to resolve it.

Examples:

- "California and New York implemented incentives for renewable energy adoption, highlighting the importance of sustainability."  
  → Contains verifiable proposition  
  → Rewritten: "California and New York implemented incentives for renewable energy adoption."

- "AI could lead to major breakthroughs in medicine."  
  → Does NOT contain verifiable proposition  
  → Rewritten: None

- "Jane is the CEO of Company X and is a great leader."  
  → Contains verifiable proposition  
  → Rewritten: "Jane is the CEO of Company X."

- “The ministry says 180 people, including 93 children, have died from malnutrition since the start of the war.”  
  → Label: Verifiable  
  → Rewritten: “The ministry says 180 people, including 93 children, have died from malnutrition since the start of the war.”
  
Now, follow the steps below carefully.
4-step reasoning:
1. Reflect on the definition of a verifiable proposition and the exclusion criteria, opinions, vague claims, etc..
2. Objectively describe what the sentence says and how it fits into the surrounding paragraph.
3. Consider all interpretations: Is the sentence stating a concrete, specific, and checkable fact? Or is it making a generalization, opinion, 
or setup statement?
4. If it is verifiable, decide whether any rewriting is needed to remove unverifiable parts. If yes, return the revised sentence; if not, return the original sentence as the rewritten_sentence

Response
Return your answer in valid JSON format like this:

{{
  "original_sentence": "<insert the original sentence>",
  "verification_label": "<verifiable | not_verifiable | unclear>",
  "rewritten_sentence": "<rewritten version, or original sentence, or 'none'>"
}}


Sentence:
{sentence}

Context:
{context}

Paragraph:
{paragraph}
    """


class DisambiguationPrompts:
    GUIDELINES = """
You are an assistant in a fact-checking pipeline. Your task is to clarify ambiguous factual claims from news articles. You will be given:

- A potentially ambiguous sentence
- The paragraph it came from
- The surrounding sentence context

Your goal is to resolve ambiguity and return a clear, self-contained factual statement — **but only if the ambiguity can be resolved using 
the provided context**.

Types of ambiguity:
- **Referential ambiguity**: unclear references like “he,” “she,” “it,” “this,” or vague time expressions like “recently.”
- **Structural ambiguity**: multiple ways the sentence could be interpreted syntactically.

Rules:
- Use ONLY the sentence, context, and paragraph. Do NOT rely on external knowledge or assumptions.
- If ambiguity can be resolved, rewrite the sentence clearly using the available information.
- If the sentence is already clear, return `"remains_unchanged"`.
- You may slightly restructure the sentence only if necessary to resolve ambiguity e.g., to embed a clarified referent or time frame.
- Preserve original phrasing as much as possible, but prioritize clarity over rigid syntax matching.
- If the input sentence is "none", treat it as a signal that the original sentence was unclear and no partial claim could be extracted. 
Try resolving the original sentence instead.
- Do not assume timelines, causality, or speaker intent unless it is explicitly stated in the context.
- Avoid combining information across multiple sentences unless a direct referent is being resolved.
- If the sentence is already self-contained and clear, return "remains_unchanged". Avoid unnecessary paraphrasing.
- Disambiguation means resolving unclear references, time expressions, or syntax. It does NOT mean rewriting for clarity or conciseness 
unless ambiguity is present.
- If the sentence contains vague or referential expressions (e.g. "she", "the district", "recently"), resolve them using context when 
the referent is clearly stated nearby.
- You may resolve references using previous or next sentences if they contain clear identifiers.
- Only return `"none"` if the ambiguity remains even after checking paragraph and context.
- Be conservative with structural ambiguity, but don’t over-reject simple referent substitutions.

Do NOT:
- Guess or invent missing facts
- Add emotional or speculative language
- Keep vague referents in the rewrite
- Do not change numerical values, named entities, or attributions unless disambiguation demands it.
- Do NOT change or substitute named entities (people, locations, organizations) unless a direct clarification from context requires it.

Output should be:
- One clear factual sentence (or "none")
- A short reason explaining how ambiguity was resolved or why it couldn’t be

Your rewrites should be minimal and preserve the original meaning as much as possible.
    """

    @staticmethod
    def get_prompt(sentence: str, context: str, paragraph: str) -> str:
        return f"""
You are an assistant in a fact-checking system. You will be given a potentially ambiguous sentence from a news article, along with its 
surrounding context and the full paragraph it appeared in. Your task is to determine whether the sentence can be rewritten into a clear, 
self-contained, and verifiable factual statement using only the provided information.

Your job:
1. Check for **referential ambiguity** — e.g., unclear subjects like "he", "she", "it", "this", or vague time expressions like "recently", 
"last week".
2. Check for **structural ambiguity** — e.g., multiple possible meanings of how clauses relate.
3. Determine whether the ambiguity can be resolved using only the sentence, context, and paragraph. If yes, rewrite the sentence into a 
**clear, standalone factual claim**. If no, return "none".

If the sentence already contains a clear, self-contained, and unambiguous factual claim, return "remains_unchanged" — even if it's slightly vague 
or long, as long as it's unambiguous.

Do NOT:
- Add new information not present in the inputs.
- Rely on prior knowledge, assumptions, or unstated facts.
- Retain emotional, speculative, or opinion-based phrasing.
- Use vague references without grounding them (e.g., "she", "they", "it").

---

### Output Format (in JSON):

{{
  "original_sentence": "<the input sentence>",
  "disambiguated_sentence": "<rephrased sentence with ambiguity resolved, 'none', or 'remains_unchanged'>",
  "reason": "<brief explanation of what was ambiguous and how it was or was not resolved>"
}}

---

### Examples

**Input:**
Sentence: "She led the initiative."
Context: "The article described how Maria Alvarez organized the protest. She led the initiative with strong community support."
→ Output:
{{
  "original_sentence": "She led the initiative.",
  "disambiguated_sentence": "Maria Alvarez led the initiative.",
  "reason": "'She' refers to Maria Alvarez based on the context. The rewrite removes referential ambiguity."
}}

**Input:**
Sentence: "Kemp's lawyer Michael Tudori said she was relieved after pleading guilty."
Context: "Alicia Kemp appeared in court. Her lawyer Michael Tudori spoke to reporters."
→ Output:
{{
  "original_sentence": "Kemp's lawyer Michael Tudori said she was relieved after pleading guilty.",
  "disambiguated_sentence": "Kemp's lawyer Michael Tudori said Kemp was relieved after pleading guilty.",
  "reason": "'She' refers to Kemp, not the lawyer. The subject of 'pleading guilty' must be Kemp, not Michael Tudori."
}}

**Input:**
Sentence: "It happened recently."
Context: "No previous sentence defines what 'it' refers to or when 'recently' was."
→ Output:
{{
  "original_sentence": "It happened recently.",
  "disambiguated_sentence": "none",
  "reason": "The sentence contains temporal and referential ambiguity ('it', 'recently') that cannot be resolved from the context."
}}

**CRITICAL RULE FOR PRONOUNS:**
- Always identify WHO the pronoun refers to by looking at the subject/agent of the action
- 'She/he pleaded guilty' = the defendant pleaded, not the lawyer
- 'She/he said' = the speaker said something
- When in doubt about agency (who did what), preserve the logical subject-verb relationship

**Input:**
Sentence: "1,800 people still reside in the district."
Context: "Earlier context clearly states 'the Korabel island district'."
→ Output:
{{
  "disambiguated_sentence": "1,800 people still reside in the Korabel island district.",
  "reason": "'the district' refers to 'the Korabel island district' based on prior context. Only the referent was resolved, rest remains
  unchanged."
}}

**Input:**
Sentence: "The ministry confirmed it."
Context: "The paragraph mentions that the Health Ministry confirmed the death toll."
→ Output:
{{
  "disambiguated_sentence": "The Health Ministry confirmed the death toll.",
  "reason": "Resolved referent 'it' using the paragraph. No other changes needed."
}}

**Input:**
Sentence: "It happened last week."
Context: "On July 24, the spokesperson confirmed the hack. It happened last week."
→ Output:
{{
  "disambiguated_sentence": "The hack happened on July 24.",
  "reason": "'last week' refers to July 24 based on explicit date in context."
}}

---

### Input

Sentence:
{sentence}

Context:
{context}

Paragraph:
{paragraph}
    """


class DecompositionPrompts:
    GUIDELINES = f"""
You are a fact-checking assistant tasked with **decomposing complex factual claims into specific, verifiable, and decontextualized propositions**.

You will be given a single **factual claim** (e.g. from a rewritten sentence after selection/disambiguation). This claim may contain:
- Multiple factual assertions
- Implicit references or pronouns
- Compound or complex sentence structures

Your goal is to extract **atomic claims** that can each be verified independently.

IMPORTANT RULES:
1. Include specific temporal information (dates, times) when available
2. Include specific geographical information (cities, states, countries)
3. Include specific numerical information (ages, quantities)
4. Include specific names and titles when mentioned
5. Each sub-claim should be independently verifiable
6. Avoid vague pronouns - use specific names/entities
7. Preserve the original context and specificity

EXAMPLE:
Original: "The suspect shot dead two officers during a raid yesterday"
Bad decomposition: "The suspect shot dead two officers"
Good decomposition: "On August 26, 2025, a suspect shot dead two Victoria Police officers during a raid in Porepunkah, Victoria, Australia"

A well-formed decomposed claim MUST:
- Contain one discrete factual proposition
- Be specific enough to check against a source
- Be verifiable based on evidence (true or false)
- Be fully self-contained (no pronouns like “he,” “it,” “they,” or vague references like “the event,” “this,” “that”)
- Include attribution if the original statement involves someone saying or asserting something

A decomposed claim MUST NOT:
- Contain opinions, speculation, beliefs, or suggestions (e.g., “should be,” “might lead to…”)
- Merge multiple propositions into one statement
- Depend on context that is not present in the claim itself
- Rewrite or simplify in ways that lose the original meaning

---

Decomposition Strategy

1. **Split** compound sentences into individual facts (e.g., joined by "and", "but", "while", etc.)
2. **Preserve attribution** when someone said, reported, claimed, or alleged something.
3. **Disambiguate referents**: If the sentence says “he” or “this group,” resolve it using prior disambiguation steps.
4. **Don’t infer** beyond the statement: stick to exactly what is written.

---

Examples

Original:  
"Israel bombed Rafah and later blocked humanitarian aid from entering the city."

Decomposed:
- "Israel bombed Rafah."
- "Israel blocked humanitarian aid from entering Rafah."

---

Original:  
"UN officials reported that 8,000 civilians had been killed and that hospitals were running out of fuel."

Decomposed:
- "UN officials reported that 8,000 civilians had been killed."
- "UN officials reported that hospitals were running out of fuel."

---

Original:  
"She said the new policy was rushed and violated previous agreements."

Decomposed:
- "She said the new policy was rushed."
- "She said the new policy violated previous agreements."

---

Original:  
"Activists argue that Israel’s response has been disproportionate."

Reject: This is an opinion and cannot be verified as a factual statement.

Accept (if rephrased by disambiguation step to contain a verifiable statement like “Activists say X happened.”)

---
Final Output Format

Your output should be a JSON object like the following:

```json
{{
  "original_claim": "<the original input claim>",
  "decomposed_claims": [
    "<claim 1>",
    "<claim 2>",
    ...
  ]
}}
If there are no valid decomposed claims, return:
{{
  "original_claim": "<the original input claim>",
  "decomposed_claims": []
}}
  """

    @staticmethod
    def get_prompt(claim: str) -> str:
        return f"""
  You are an assistant for a claim verification system. You will be given a factual claim, which may contain multiple facts, ambiguous references, or complex constructions.

Your task is to **decompose** this claim into a list of **specific, verifiable, and decontextualized factual propositions**.

A **decomposed proposition** must satisfy the following:

### It is:
1. **Specific** – not vague, general, or speculative.
2. **Verifiable** – can be objectively verified as true or false via factual evidence.
3. **Decontextualized** – understandable without relying on any outside context (e.g. no ambiguous pronouns like “she,” “it,” “this”).

### It must NOT:
- Contain opinions, hypotheticals, or subjective language.
- Omit important attribution (e.g., if the claim is made by a person or organization, preserve that attribution).
- Include multiple facts in one proposition — keep them **atomic** and **discrete**.
- Add or infer information not present in the original claim.

---

### Examples

**Claim:**  
"UN officials say 5,000 people have died and that Israel blocked journalists from entering."

**Decomposed Propositions:**  
- "UN officials say 5,000 people have died."  
- "UN officials say that Israel blocked journalists from entering."

---

**Claim:**  
"Elon Musk said Starlink helped Ukraine, and that other companies should do more."

**Decomposed Propositions:**  
- "Elon Musk said Starlink helped Ukraine."  
- "Elon Musk said that other companies should do more."

---

**Claim:**  
"She called the policy ineffective and said it had worsened poverty."

**Decomposed Propositions:**  
- "She said the policy is ineffective."  
- "She said the policy had worsened poverty."

(*Note: In real processing, pronouns like “she” should be resolved prior to this step using disambiguation.*)

---

### Important

If the claim contains **no specific and verifiable information**, return an empty list for `"decomposed_claims"`.

---

## Final Output Format (JSON)

You must return a valid JSON object structured like this:

```json
{{
  "original_claim": "<insert original claim>",
  "decomposed_claims": [
    "<proposition 1>",
    "<proposition 2>",
    ...
  ]
}}

Claim:
{claim}
  """
