"""Reporter identity stage IDs used both for execution and routing.

Ambiguous model review and search remain reserved names until implemented.
A deferred judgment names its destination stage, not a routing bucket.
"""

REPORTER_ROOT_LOOKUP = "reporter_root_lookup"
REPORTER_ROOT_LOOKUP_AMBIGUOUS = "reporter_root_lookup_ambiguous"
REPORTER_ROOT_LOOKUP_UNIQUE_LLM = "reporter_root_lookup_unique_llm"
REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM = "reporter_root_lookup_ambiguous_llm"
REPORTER_ROOT_LOOKUP_LARGE_CANDIDATE_REVIEW = "reporter_root_lookup_large_candidate_review"
REPORTER_ROOT_SEARCH = "reporter_root_search"
