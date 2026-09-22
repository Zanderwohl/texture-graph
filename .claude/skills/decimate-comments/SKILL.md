---
name: decimate-comments
description: Cut comments back to the essential why. Use when asked for a comment pass, a comment-slashing pass, to decimate or slash comments, or to cut comments down.
---

# Decimate comments

Cut down on comments severely, keeping only the most essential *why* information in each
comment. A comment is not longer than the function it describes. Remove information about what
things used to do or used to be. Write in simple, straightforward style, without metaphor or
simile, for a human.

`CLAUDE.md`'s **Comments** section is the standing rule; this is the pass that enforces it.

## Delete

- Restatements of the code, in any dress: `/// Returns the name.`, a doc that lists the
  parameters the signature already names.
- History. "used to be", "the first version", "the bug was", "this used to race". The log
  remembers, and a reader with the current code in front of them cannot use any of it.
- Argument for a decision the code already makes — the same point in three sentences, then
  again in the test's doc.
- Metaphor, simile, and bold used for emphasis rather than for a warning.
- Design-document prose. Link to the document instead.

## Keep

- **Why**, where the why is not derivable: a constraint, a discarded alternative, a bug this
  shape prevents.
- Measured numbers and where they came from — the threshold someone found by looking.
- Units, ranges and frames the type does not carry.
- Numerical hazards: cancellation, saturation, a tolerance and its reason.
- Invariants a caller must uphold.
- Module docs. They carry the shared context so the items below need not repeat it.
- A test's doc, cut to one or two sentences saying what breaks.

**Do not delete a comment you do not understand.** It is the one most likely to be load-bearing.
Read the code until it is either derivable — then cut it — or not.

## Scope

State which files before starting, and leave the rest. Two things are out of bounds unless
asked: code moved verbatim from elsewhere, where the comments travelled with it and the
verbatim move is the property worth keeping, and anything vendored.

Prefer making the code say it — a named constant, a smaller function, a better type removes the
comment that explained it. That is a code change: say so rather than slipping it in.

## Check

Count before and after; the count is the review artifact.

```bash
grep -rc '^\s*//' <files>
```

Work in batches with `cargo test` between them — a bulk edit that takes a line of code with it
is otherwise found much later. The diff should be nearly all deletions.
