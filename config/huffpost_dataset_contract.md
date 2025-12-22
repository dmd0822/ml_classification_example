# HuffPost dataset contract (reference)

This project trains a multi-class classifier to predict `category` from a derived `text` column.

## Raw input expectations

The raw dataset is expected to provide at least the following fields per record:

- `category` (required): label target; non-empty string
- `headline` (required): headline text; string (may be empty/null)
- `short_description` (required): short description text; string (may be empty/null)

If additional fields exist (e.g., `date`, `link`, `authors`), they are treated as optional metadata and are not required for the reference baseline.

## Derived fields

### `text`

A derived text field built deterministically from raw fields:

- `text = headline + separator + short_description`
- `separator` is defined in the config (default: `[SEP]`)
- Nulls are replaced using config defaults (default: empty string)
- Output is whitespace-normalized and trimmed (config-driven)

## Validation rules (fail fast)

At minimum, preprocessing should enforce:

- `category` must be present and non-empty for all training rows
- `text` must be non-empty after normalization (or the row is dropped)
- Label cardinality must be >= 2 to run a classification experiment

## Where this is configured

See [config/huffpost_category_text.json](huffpost_category_text.json).
