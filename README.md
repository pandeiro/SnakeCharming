# SnakeCharming

Single-page HTML/JS harness and lesson plans for introducing students to Python coding exercises.

## Structure

```
- index.html
- doc
  - <non-lesson-prefix>-*.md  // helper docs (not shown)
- lessons
  - <lesson-name>.md          // lesson plans
```

## Producing New Lessons

1. Use `doc/prompt-spec.md` and play with the output.
2. When finished, save results `lessons/<name-of-lesson>.md` and add to `LESSONS` const at the top
of the first `<script></script>` block in `index.html`.

## Deployment

Static site, wherever you want it.

To automate, put instructions in `./deploy.sh`.
