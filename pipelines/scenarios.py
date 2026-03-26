

FEVER_FILTER = "{claim}{content}Given a claim and evidence, decide if the evidence is enough to determine whether the claim is true or false."
FEVER_MAP = "{claim}{content}Explain how the evidence supports or unsupports the claim."


FILTER_ENRON_FRAUD = (
    "{contents}\n"
    'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy"). '
    "Answer TRUE if it does, FALSE otherwise. Output TRUE or FALSE only.\n"
)

FILTER_ENRON_NOT_NEWS = (
    "{contents}\n"
    "The email is not quoting from a news article or an article written by someone outside of Enron. "
    "Answer TRUE if it is NOT quoting such an article, FALSE otherwise. Output TRUE or FALSE only.\n"
)

MAP_ENRON_EXPLANATION = (
    "{contents}\n"
    "Explain briefly why this email is related to a fraudulent scheme, using the email contents provided in the context."
)

