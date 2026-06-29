
---
## Description

When `resolving` the citations, we can determine whether a `Full-case` citation is missing by deduction. If we find a `short`, `supra`, or `Id`, citation with no reference to a `Full-citaion` then we can deduce that we either missed it, or there is a mistake with their names. 

We can implement a feed-back loop where we go back to search for a full citation, when we know that is should be in the text. 

---
## Recall Recovery

### Dangling Citations

If there is a short-case, ID, or supra citation that doesn't belong to a full-case citation, that could indicate that we missed a full-case citation. It could also indicate a hallucination.