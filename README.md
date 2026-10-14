# Knowledge Map

A personal map of full-stack development, AI, and data science. Open this folder as an Obsidian vault and use Graph View to see it.

## Areas

- [[fullstack/_index|Full-Stack Development]]
- [[ai/_index|AI / Machine Learning]]
- [[datascience/_index|Data Science]]
- [[overlap/_index|Overlap: where the areas meet]]

## How this works

- One concept = one note, named in `kebab-case.md` and placed in its area folder.
- Every note starts from `_templates/concept.md` (frontmatter: `area`, `confidence`, `related`, `status`, `updated`).
- Link concepts with `[[wiki-links]]`. Links create the graph edges.
- Confidence: 1 = heard of it, 3 = can explain it, 5 = can teach it.
- Notes are written during Q&A, then committed and pushed with git.

## Review queue

Low-confidence notes (needs the Dataview plugin):

```dataview
TABLE area, confidence, updated
FROM ""
WHERE confidence AND confidence <= 2
SORT confidence ASC
```
