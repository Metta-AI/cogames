# Tutorial Spec

## Approach

- **`cogames tutorial play`**: Interactive CLI command, launches mettascope GUI
- **Everything else**: Notebooks in `notebooks/`, linked from README.ipynb

## `cogames tutorial play`

Improvement: Terminal prompts wait for specific in-game events (not just Enter). Clear success/fail feedback.

## Notebooks

| Notebook                      | Purpose                                     |
| ----------------------------- | ------------------------------------------- |
| `notebooks/make-policy.ipynb` | Policy interface, create and test a policy  |
| `notebooks/train.ipynb`       | Training, interpreting metrics, checkpoints |
| `notebooks/tournament.ipynb`  | Submit policies, view results               |

## Testing

- `cogames tutorial play --non-interactive` for CI
- Notebooks tested via nbdev
- Nightly: Full tutorial flow including tournament submission
