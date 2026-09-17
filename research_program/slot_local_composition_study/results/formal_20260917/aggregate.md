# Slot-local receiver representation aggregation

- runs: 128
- split: heldout goal combination

| representation | live heldout | silent heldout | live−silent | zero-shot passes |
|---|---:|---:|---:|---:|
| `history` | 0.254 | -0.066 | +0.319 [+0.268,+0.371] | 0/32 |
| `slot_local` | 0.328 | -0.049 | +0.377 [+0.271,+0.483] | 9/32 |

| contrast | mean | 95% CI |
|---|---:|---:|
| `slot_local_minus_history|live|heldout` | +0.074 | [-0.013,+0.162] |
| `representation_interaction|heldout` | +0.058 | [-0.033,+0.149] |

The slot-local arm is an architectural representation intervention. It tests whether staged action timing is sufficient to induce factor-wise receiver states; it does not supply a semantic dictionary.
