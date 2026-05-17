
# FEVER
# FEVER_FILTER = "{claim}{content}Given a claim and evidence, decide if the evidence is enough to determine whether the claim is true or false."
# FEVER_MAP = "{claim}{content}Explain how the evidence supports or unsupports the claim."

# FEVER_FILTER = "{data}Given a claim and several evidence, determine if any of the evidence can support the claim"
# FEVER_MAP = "{data}Which evidence can support the claim?"
FEVER_FILTER = "{data}Given a claim and evidence, decide if the evidence is enough to determine whether the claim is true or false."
FEVER_MAP = "{data}Explain how the evidence supports or unsupports the claim. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters."

FEVER_FACTOOL_QUERY_MAP = (
    "Generate two concise Wikipedia search queries that would retrieve evidence "
    "needed to verify the claim.\n\n"
    "Example 1\n"
    "Claim: Graham Neubig is a professor at MIT.\n"
    "Output:\n"
    "1. Graham Neubig current position\n"
    "2. Is Graham Neubig a professor at MIT?\n\n"
    "Example 2\n"
    "Claim: The Eiffel Tower is located in Berlin.\n"
    "Output:\n"
    "1. Eiffel Tower location\n"
    "2. Is the Eiffel Tower in Berlin?\n\n"
    "Example 3\n"
    "Claim: The film Titanic was directed by James Cameron.\n"
    "Output:\n"
    "1. Titanic film director\n"
    "2. James Cameron directed Titanic\n\n"
    "{claim}\n"
    "Return exactly two numbered search queries and no other text."
)
FEVER_FACTOOL_SUPPORT_FILTER = (
    "{claim}{content}Based on the retrieved Wikipedia content, is the claim supported? "
    "Answer true only if the Wikipedia content clearly supports the claim."
)

# ENRON 1
FILTER_ENRON_FRAUD = (
    '{contents}The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'

)
FILTER_ENRON_NOT_NEWS = (
    '{contents}The email is not quoting from a news article or an article written by someone outside of Enron.'
)

MAP_ENRON_EXPLANATION = (
    '{contents}Explain briefly why this email is related to a fraudulent scheme, using the email contents provided in the context. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.'
)


FILTER_ENRON_FRAUD_2 = (
    '{contents}The email is normal scheme (i.e., Not "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'

)
FILTER_ENRON_NOT_NEWS_2 = (
    '{contents}The email includes people associated with Enron.'
)

MAP_ENRON_EXPLANATION_2 = (
    '{contents}Summarize the email briefly. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.'
)
# ARXIV

# TOPK - MAP
ARXIV_CASE_1_TOPK = ("{abstract} Provide the most relevant papers to Image Semantic Segmentation research")
ARXIV_CASE_1_MAP = (
    '{abstract}Explain why is the paper relevant to Image Semantic Segmentation research. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.'
    )
ARXIV_CASE_1_AGG = (
    '{abstract}"Summarize the key common ideas and contributions across the paper abstracts"'
    )



# FILTER - JOIN - FILTER - MAP -

ARXIV_CASE_2_FILTER = ("{abstract}Is the paper relevant to AI robotics research?")
ARXIV_CASE_2_JOIN = (
    '{abstract}{robotic_abstract}Do these two papers study similar topic?'
)
ARXIV_CASE_2_MAP = (
    '{abstract}Explain their relationship in terms of shared topics, methods, and differences. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.'
)

ARXIV_CASE_3_FILTER_1 = ("{abstract}Is this abstract about AI or machine learning?")
ARXIV_CASE_3_FILTER_2 = ("{abstract}Is this abstract describing theoretical work?")
ARXIV_CASE_3_FILTER_3 = ("{abstract}Does this abstract mention using an image dataset?")
ARXIV_CASE_3_MAP = ("{abstract}Summarize the abstract in 1–2 sentences, focusing on the main method and key result. Do not add filler, repeated punctuation, or repeated characters.")

#MOVIE PLOT
RESUME_CASE_1_FILTER = ("{resume}Does this resume show software development or programming experience?")
RESUME_CASE_1_MAP_1 = ("{resume}Summarize the resume. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.")
RESUME_CASE_1_JOIN = ("{resume}{job}Is this candidate a plausible fit for this software/developer job based on skills, tools, and experience?")
RESUME_CASE_1_MAP = ("{resume}{job}Explain why this candidate is suitable for the position briefly. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.")


RESUME_CASE_2_TOPK = ("{resume}I am finding a candidate who knows C++ and Machine Learning or Statistical Analysis.")
RESUME_CASE_2_FILTER = ("{resume}Does this candidate have more than 5 years of experience?")
RESUME_CASE_2_MAP = ("{resume}Can you extract the technical skills of the candidate? Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.")


RESUME_CASE_3_MAP = ("{resume}Summarize the resume. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.")
RESUME_CASE_3_FILTER = ("{resume}Does this resume show software development or programming experience?")

# BioDEX reaction classification
BIODEX_CASE_1_JOIN = (
    "{article}{reaction_label}Does the biomedical article describe a patient who experienced this adverse drug reaction?"
)
BIODEX_CASE_1_MAP = (
    "{article}{reaction_label}Explain briefly why this reaction label applies to the article. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters."
)


# Contract-NLI
CONTRACT_NLI_VALID_CONTRACT = (
    "{contract}Is this document a valid contract or agreement text with enough substantive "
    "clauses to evaluate confidentiality obligations?"
)
CONTRACT_NLI_ENTAILMENT_JOIN = (
    "{contract}{hypothesis}Given the contract and hypothesis, does the contract entail the hypothesis? "
    "Answer true only if the contract clearly supports the hypothesis."
)
CONTRACT_NLI_EXPLAIN_ENTAILMENT = (
    "{contract}{hypothesis}Explain briefly why the contract entails the hypothesis. Cite the relevant "
    "obligation or clause in one concise sentence. Do not add filler, repeated "
    "punctuation, or repeated characters."
)


# MEDEC medical error detection and correction
MEDEC_ERROR_FILTER = (
    "{data}This is the MEDEC benchmark for medical error detection in clinical notes. "
    "The note may contain one subtle deliberately injected medical error in one numbered sentence, "
    "such as an incorrect diagnosis, pathogen, treatment, test, anatomy, epidemiology, or clinical fact. "
    "About half of the notes contain an error. Compare each numbered sentence against the rest of the "
    "case context and expected medical knowledge. Answer true if any numbered sentence is medically "
    "inconsistent or likely erroneous. Answer false only if all numbered sentences are medically consistent. "
    "Return only TRUE or FALSE."
)
MEDEC_ERROR_SENTENCE_ID_MAP = (
    "{data}Identify the numbered sentence that contains the medical error. "
    "Return only the sentence ID as an integer, with no explanation."
)
MEDEC_CORRECTED_SENTENCE_MAP = (
    "{data}Generate the corrected version of the erroneous numbered sentence. "
    "Return only the corrected sentence, with no explanation."
)
