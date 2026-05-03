
# FEVER
FEVER_FILTER = "Given a claim and evidence, decide if the evidence is enough to determine whether the claim is true or false."
FEVER_MAP = "Explain how the evidence supports or unsupports the claim."


# ENRON 1
FILTER_ENRON_FRAUD = (
    'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'

)
FILTER_ENRON_NOT_NEWS = (
    'The email is not quoting from a news article or an article written by someone outside of Enron.'
)

MAP_ENRON_EXPLANATION = (
    'Explain briefly why this email is related to a fraudulent scheme, using the email contents provided in the context.'
)


FILTER_ENRON_FRAUD_2 = (
    'The email is normal scheme (i.e., Not "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'

)
FILTER_ENRON_NOT_NEWS_2 = (
    'The email includes people associated with Enron.'
)

MAP_ENRON_EXPLANATION_2 = (
    'Summarize the email briefly.'
)
# ARXIV

# TOPK - MAP
CASE_1_TOPK_ARXIV = ("Provide the most relevant papers to Image Semantic Segmentation research")
CASE_1_MAP_ARXIV = (
    'Explain why is the paper relevant to Image Semantic Segmentation research'
    )
CASE_1_AGG_ARXIV = (
    '"Summarize the key common ideas and contributions across the paper abstracts"'
    )

# FILTER - JOIN - FILTER - MAP -
CASE_2_FILTER_ARXIV = ("Is the paper relevant to AI robotic research?")
CASE_2_JOIN_ARXIV = (
    'Do these two papers study similar topic?'
)
CASE_2_MAP_ARXIV = (
    'Explain their relationship in terms of shared topics, methods, and differences.'
)

CASE_3_FILTER_1 = ("Is this abstract about AI or machine learning?")
CASE_3_FILTER_2 = ("Is this abstract describing theoretical work?")
CASE_3_FILTER_3 = ("Does this abstract mention using an image dataset?")
CASE_3_MAP_ARXIV = ("Summarize the abstract in 1–2 sentences, focusing on the main method and key result.")



RESUME_CASE_1_FILTER = ("Does this resume show software development or programming experience?")
RESUME_CASE_1_MAP_1 = ("Summarize the resume")
RESUME_CASE_1_JOIN = ("Is this candidate a plausible fit for this software/developer job based on skills, tools, and experience?")
RESUME_CASE_1_MAP = ("Explain why this candidate is suitable for the position briefly")

RESUME_CASE_2_TOPK = ("I am finding a candidate who knows C++ and Machine Learning or Statistical Analysis.")
RESUME_CASE_2_FILTER = ("Does this candidate have more than 5 years of experience?")
RESUME_CASE_2_MAP = ("Can you extract the technical skills of the candidate?")

