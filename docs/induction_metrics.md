# Induction Head Metrics

This project uses three complementary measurements to identify induction-like
attention heads in GPT-2.

## Data Setup

The synthetic sequence has this structure:

```text
[BOS] + random prefix + pattern + same pattern again
```

The repeated pattern creates an induction task. If the model sees a token in the
second copy, it can look back to the first copy, find the same token, and use the
token that followed it there as evidence for what should come next now.

Example:

```text
... A B C D ... A B C D
```

When the model is at the second `A`, a useful induction head should attend to
the `B` after the first `A`.

## 1. Previous Token Score

This metric asks:

```text
Does this head mostly look one token back?
```

For an attention pattern

```text
A[head, query_position, key_position]
```

the value

```text
A[h, i, i - 1]
```

means:

```text
At position i, head h attends to the previous position i - 1.
```

The score is the average of the diagonal immediately below the main diagonal:

```text
PreviousTokenScore(h) = mean_i A[h, i, i - 1]
```

The first token is naturally skipped because position `0` has no previous token.

Intuition:

```text
A high score means the head behaves like a "look one step back" head.
```

This is useful, but not enough to prove induction behavior. A head can look at
the previous token without using the repeated pattern.

## 2. Induction Score

This metric asks:

```text
When the current token appeared before, does this head look at the token that
followed its earlier occurrence?
```

Suppose:

```text
tokens[22] == tokens[12]
```

Then the induction target key is:

```text
12 + 1 = 13
```

So we read:

```text
A[h, 22, 13]
```

More generally:

```text
InductionScore(h) = mean_r A[h, query_r, source_r + 1]
```

where:

```text
query_r  = position in the second pattern
source_r = matching position in the first pattern
```

Intuition:

```text
A high score means the head is looking at "what came after this same token last
time."
```

This is the attention-side signature of an induction head.

## 3. Logit Attribution Score

Attention tells us where a head looks, but not whether the head actually helps
the model predict the right next token. Logit attribution asks:

```text
Does this head's output increase the logit of the correct next token?
```

TransformerLens `hook_z` gives each head's output before it is projected back
into the residual stream:

```text
z[layer, position, head, d_head]
```

For one head, we first project through that layer's output matrix:

```text
o = z W_O
```

Then we compare how much this residual contribution supports the correct next
token versus the rest of the vocabulary:

```text
LogitAttribution = logit(correct_token) - mean(logit(other_tokens))
```

Equivalently, with unembedding matrix `W_U` and vocabulary size `V`, the code
uses the optimized direction:

```text
d_y = (V * W_U[:, y] - sum_v W_U[:, v]) / (V - 1)
```

Then:

```text
LogitAttribution = o · d_y
```

This avoids constructing a large full-vocabulary logit tensor while computing
the same value.

Intuition:

```text
A high positive score means this head is not only attending to useful evidence;
its output is pushing the model toward the correct next token.
```

## How To Read The Metrics Together

A strong induction head should usually have:

```text
high induction_score
positive/high logit_attribution_score
```

The previous-token score is helpful context:

```text
high previous_token_score
```

means the head often looks one step back, but this alone is not sufficient to
identify induction behavior.

In short:

```text
Previous Token Score  = "Does it look one token back?"
Induction Score       = "Does it look at the token after the earlier match?"
Logit Attribution     = "Does that information push the correct next token?"
```
