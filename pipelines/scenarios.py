
# FEVER
# FEVER_FILTER = "{claim}{content}Given a claim and evidence, decide if the evidence is enough to determine whether the claim is true or false."
# FEVER_MAP = "{claim}{content}Explain how the evidence supports or unsupports the claim."

FEVER_FILTER = "{data}Given a claim and several evidence, determine if any of the evidence can support the claim"
FEVER_MAP = "{data}Which evidence can support the claim?"

# ENRON 1
FILTER_ENRON_FRAUD = (
    '{contents}The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'

)
FILTER_ENRON_NOT_NEWS = (
    '{contents}The email is not quoting from a news article or an article written by someone outside of Enron.'
)

MAP_ENRON_EXPLANATION = (
    '{contents}Explain briefly why this email is related to a fraudulent scheme, using the email contents provided in the context.'
)


FILTER_ENRON_FRAUD_2 = (
    '{contents}The email is normal scheme (i.e., Not "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'

)
FILTER_ENRON_NOT_NEWS_2 = (
    '{contents}The email includes people associated with Enron.'
)

MAP_ENRON_EXPLANATION_2 = (
    '{contents}Summarize the email briefly.'
)
# ARXIV

# TOPK - MAP
ARXIV_CASE_1_TOPK = ("{abstract} Provide the most relevant papers to Image Semantic Segmentation research")
ARXIV_CASE_1_MAP = (
    '{abstract}Explain why is the paper relevant to Image Semantic Segmentation research'
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
    '{abstract}Explain their relationship in terms of shared topics, methods, and differences.'
)

ARXIV_CASE_3_FILTER_1 = ("{abstract}Is this abstract about AI or machine learning?")
ARXIV_CASE_3_FILTER_2 = ("{abstract}Is this abstract describing theoretical work?")
ARXIV_CASE_3_FILTER_3 = ("{abstract}Does this abstract mention using an image dataset?")
ARXIV_CASE_3_MAP = ("{abstract}Summarize the abstract in 1–2 sentences, focusing on the main method and key result.")

#MOVIE PLOT
RESUME_CASE_1_FILTER = ("{resume}Does this candidate programming experience?")
RESUME_CASE_1_JOIN = ("{resume}{job}Is this candidate suitable for this position?")
RESUME_CASE_1_MAP = ("{resume}{job}Explain why this candidate is suitable for the position briefly")


RESUME_CASE_2_TOPK = ("{resume}I am finding a candidate who knows C++ and Machine Learning or Statistical Analysis.")
RESUME_CASE_2_FILTER = ("{resume}Does this candidate have more than 5 years of experience?")
RESUME_CASE_2_MAP = ("{resume}Can you extract the technical skills of the candidate?")
