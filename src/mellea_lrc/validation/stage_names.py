"""Reporter identity stage IDs used both for execution and routing.

The review and search stages are reserved names until their implementations
exist. A deferred judgment names its destination stage, not a routing bucket.
"""

REPORTER_ROOT_EXACT_LOOKUP = "reporter_root_exact_lookup"
REPORTER_ROOT_EXACT_AMBIGUITY = "reporter_root_exact_ambiguity"
REPORTER_ROOT_EXACT_REVIEW = "reporter_root_exact_review"
REPORTER_ROOT_EXACT_AMBIGUITY_REVIEW = "reporter_root_exact_ambiguity_review"
REPORTER_ROOT_LARGE_CANDIDATE_REVIEW = "reporter_root_large_candidate_review"
REPORTER_ROOT_SEARCH = "reporter_root_search"
