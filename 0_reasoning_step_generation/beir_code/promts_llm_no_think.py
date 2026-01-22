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
    
    At each step, you must decide whether the current documents are sufficient
    for addressing the original user query, or whether you need to produce a refined
    summary to request another retrieval round.
    
    IMPORTANT:
    The <summary> you produce will be used directly as the query for the next retrieval round.
    It guides the system to fetch a new set of documents in <information> ... </information>.
    Although your summaries help obtain better evidence, they do NOT replace or modify
    the original user query.
    You should always judge sufficiency based on the original query, while using new
    <information> blocks to improve your understanding of what evid
    
    ===================================================
    === STRICT OUTPUT FORMAT (FOLLOW EXACTLY) =========
    ===================================================
    
    At every step, your output MUST be in ONE of the following two formats:
    
    ---------------------------------------------------
    1) If the current documents are NOT sufficient:
    <reason> ... </reason>
    <summary> ... </summary>
    <information>
    ... next set of retrieved documents will appear here ...
    </information>
    
    This means you want to continue.
    The system will feed you the next <information> block for the next step.
    
    ---------------------------------------------------
    2) If the current documents ARE sufficient:
    <reason> ... </reason>
    <satisfy> yes </satisfy>
    
    This means you want to stop.
    After producing <satisfy>, do not output <summary> or <information>.
    
    ===================================================
    === DETAILS & REQUIREMENTS ========================
    ===================================================
    
    <reason> ... </reason>
	--Do the documents answer the user query? Output only: STOP (satisfied) or CONTINUE (not satisfied). No reasoning. No explanation.
    --1–5 words.
    
    <summary> ... </summary>
    - Produce this ONLY when documents are insufficient.
    - 500–2000 words.
    - A refined, retrieval-friendly description of missing information.
    - This summary will become the query for the next retrieval round.
     -This summary combines the original query with the summary from the most recent previous round, but does not include all earlier rounds.
    <information> ... </information>
    - Provided by the system AFTER you output <summary>.
    - You do NOT generate the content of this block.
    
    <satisfy> yes </satisfy>
    - Output this ONLY when the current documents are sufficient for the original query.
    
    ===================================================
    === DECISION GUIDELINES ===========================
    ===================================================
    
    Continue (produce <summary>) if:
    - Current documents do NOT sufficiently address the original query.
    - Important evidence is missing or unclear.
    - Better retrieval may help.
    
    Stop (produce <satisfy>) if:
    - The documents contain enough information to address the original query.
    - More retrieval is unlikely to help.
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
    ---Do the documents answer the user query? Output only: STOP (satisfied) or CONTINUE (not satisfied). No reasoning. No explanation. CONTINUE (not satisfied).
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


    Example 2 (If the current documents ARE sufficient, stop retrieval):
    <query>
    Claim in article about why insects are attracted to light.
    In this article they are addressing the reason insects are attracted to light when they say:
    "Heat radiation as an attractive component is refuted by the effect of LED lighting, which supplies negligible infrared radiation yet still entraps vast numbers of insects."
    I don't see why attraction to LEDs shows they're not seeking heat. Could they for example be evolutionarily programmed to associate light with heat? So that even though they don't encounter heat near/on the LEDs they still "expect" to?
    </query>

    <information>
    A proximate cause is an event which is closest to, or immediately responsible for causing some observed result. This contrasts with ultimate causes, which represent evolutionary or distal explanations.
    insects_attracted_to_light/Proximate_and_ultimate_causation_1.txt
    Ultimate causation explains traits in terms of evolutionary forces acting on them.
    Proximate causation explains biological function in terms of immediate physiological or environmental factors.
    These two levels explain the same observed behavior but at different depths (evolutionary vs. physiological).

    insects_attracted_to_light/Phototaxis_0.txt
    Phototaxis is movement toward or away from light. Many organisms exhibit positive phototaxis.
    Different organisms show phototaxis for different physiological reasons, some incidental and not tied to survival.

    insects_attracted_to_light/Phototaxis_4.txt
    Under experimental conditions, light interacts with magnetic fields in phototactic organisms.
    Drosophila larvae orient toward predicted light gradients when magnetic fields rotate.
    </information>

    <reason>
   ---Do the documents answer the user query? Output only: STOP or CONTINUE. No reasoning. No explanation. STOP (satisfied).
    </reason> 
    
    <satisfy> 
    yes 
    </satisfy>


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
