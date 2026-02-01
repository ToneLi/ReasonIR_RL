def get_prompt(method, query, documents_block):
    user_prompts = {
    "think_prompt":
    f"""
    You are a professional multi-step retrieval agent.
    Your role is to iteratively refine a query through reasoning and summarization,
    and use an external retriever (not shown to you) to gather increasingly relevant documents.
    You will be given:
    - A user query wrapped in <query> ... </query>.
    - An initial set of retrieved documents wrapped in <information> ... </information>.
    
    At each step, you must need to produce a refined
    summary to request another retrieval round.
    
    IMPORTANT:
    The <summary> you produce will be used directly as the query for the next retrieval round.
    It guides the system to fetch a new set of documents in <information> ... </information>.
    Although your summaries help obtain better evidence, they do NOT replace or modify
    the original user query.
   
    
    ===================================================
    === STRICT OUTPUT FORMAT (FOLLOW EXACTLY) =========
    ===================================================
    
    At every step, your output MUST be in  the following  format:
    <reason> ... </reason>
    <summary> ... </summary>
    <information>
    ... next set of retrieved documents will appear here ...
    </information>
    
    ===================================================
    === DETAILS & REQUIREMENTS ========================
    ===================================================
    
    <reason> ... </reason>
    - Evaluate whether the current documents address the original user query.
    - Describe what parts of the documents satisfy the query and what is still missing.
    - Clearly justify why you stop or continue.
    - 200–500 words.
    
    <summary> ... </summary>
    - Produce this ONLY when documents are insufficient.
    - 500–2000 words.
    - A refined, retrieval-friendly description of missing information.
    - This summary will become the query for the next retrieval round.
     -This summary combines the original query with the summary from the most recent previous round, but does not include all earlier rounds.
    <information> ... </information>
    - Provided by the system AFTER you output <summary>.
    - You do NOT generate the content of this block.
     
    
    ===================================================
    === EXAMPLES ======================================
    ===================================================

    Example 1 (1) If the current documents are NOT sufficient, continue retrieval):
    <query>
    Claim in article about why insects are attracted to light.
    In this article they are addressing the reason insects are attracted to light when they say:
    "Heat radiation as an attractive component is refuted by the effect of LED lighting, which supplies negligible infrared radiation yet still entraps vast numbers of insects."
    I don't see why attraction to LEDs shows they're not seeking heat. Could they for example be evolutionarily programmed to associate light with heat? So that even though they don't encounter heat near/on the LEDs they still "expect" to?
    </query>

    <information>
    A proximate cause is an event which is closest to, or immediately responsible for causing some observed result. This contrasts with ultimate causes, which represent evolutionary or distal explanations.

    Ultimate causation explains traits in terms of evolutionary forces acting on them.
    Proximate causation explains biological function in terms of immediate physiological or environmental factors.
    These two levels explain the same observed behavior but at different depths (evolutionary vs. physiological).

    Phototaxis is movement toward or away from light. Many organisms exhibit positive phototaxis.
    Different organisms show phototaxis for different physiological reasons, some incidental and not tied to survival.
    </information>

    <reason>
    1. Do the current documents address the original query?
    No. The provided information only explains proximate/ultimate causation and defines phototaxis at a very general level. It does not discuss why insects approach lights, why LEDs attract them, or whether heat association plays any role.
    2. What parts partially satisfy the query?
    The distinction between proximate and ultimate causation is relevant because the query asks whether insects evolved to associate light with heat (an ultimate explanation) or respond directly to light cues (a proximate explanation). The phototaxis definition also loosely connects to insect light-seeking behavior. However, these are only conceptual frameworks, not answers.
    3. What is missing?
    There is no discussion of insect visual physiology, wavelength sensitivity, or navigation mechanisms that might explain attraction to artificial lights. No evidence is provided concerning heat vs. light attraction or experiments comparing incandescent vs. LED lighting. The possibility of evolutionary “light-heat association” is also not addressed.
    4. satisfy or not satisfy?
    Not satisfy,  The given information is too general to answer the query’s specific biological and evolutionary questions. Additional retrieval is necessary to obtain mechanistic and empirical evidence.
    </reason>

    <summary>
    Original query (included as required):
    Why are insects attracted to light? The article argues heat is not the cause because LEDs emit almost no infrared yet still attract insects. Why does LED-attraction refute heat-seeking? Could insects have evolved to associate light with heat and “expect” heat even when LEDs provide none?
    Refined retrieval-focused summary:
    The current information gives only general definitions of proximate/ultimate causation and phototaxis, which cannot explain why insects approach artificial lights or whether heat plays a role. Further retrieval should focus on insect visual physiology (brightness, wavelength, UV sensitivity) and how artificial lights disrupt navigation systems. Evidence separating heat cues from light cues is needed, including whether insects approach heat sources without light and how thermoreceptors function across species.
    Retrieval should also include studies comparing insects’ responses to incandescent versus LED lights, especially work isolating heat, wavelength, and intensity to determine whether heat contributes meaningfully to attraction. On the evolutionary side, evidence is needed for hypotheses such as celestial navigation, sensory traps, and whether insects ever evolved any natural link between light cues and heat cues. Retrieval should clarify whether modern lighting creates an evolutionary mismatch rather than a heat-expectation mechanism.
    </summary>

    <information>
    ... next set of retrieved documents will appear here ...
    </information>
    ===================================================
    === INPUT BEGINS ==================================
    ===================================================
    
    User Query:
    <query>
    {query}
    </query>
    
    Initial Retrieved Documents:
    <information>
    {documents_block}
    </information>
    ===================================================
    === MODEL OUTPUT BEGINS ===========================
    ===================================================
    """
    }

    return user_prompts[method]