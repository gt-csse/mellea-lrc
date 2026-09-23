"""Reviewers: one module per question, each answering about one candidate.

A reviewer is given a candidate and the window around it and returns an
:class:`~mellea_lrc.extraction.adjudication.types.Adjudication`. It never decides what is
in the document, only whether characters already there mean what a generator
proposed -- which is the property that lets an answer be grounded back into the
text rather than trusted.
"""
