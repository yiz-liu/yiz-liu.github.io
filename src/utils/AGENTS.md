# SHARED UTILITY NOTES

## OVERVIEW
Cross-cutting helper boundary for content lookup, locale-aware URLs, and date formatting.

## STRUCTURE
```text
src/utils/
├── content-utils.ts  # collection grouping, fallback, draft filtering
├── time.ts           # shared date-formatting helpers
└── url-utils.ts      # locale-aware links and blog asset paths
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Blog/spec loading | `content-utils.ts` | draft filtering, fallback, grouping |
| Locale-aware links | `url-utils.ts` | default-locale omission handled here |
| Date formatting | `time.ts` | shared display helpers |

## CONVENTIONS
- Prefer these helpers over reimplementing logic in pages or components.
- `getBlogEntrySort(lang)` is the source of truth for blog list ordering, draft visibility, translation grouping, and `isFallback` tagging.
- `getSpec(lang, spec)` is the source of truth for spec-page fallback behavior.
- `baseUrl()` and `getRelativeLocaleUrl()` should own URL construction.
- `blogCoverUrl()` is the expected path normalizer for blog images.

## ANTI-PATTERNS
- Do not duplicate locale fallback logic in routes or components.
- Do not build route URLs with string concatenation when `getRelativeLocaleUrl()` exists.
- Do not bypass `blogCoverUrl()` when resolving post-relative images.
- Do not change draft filtering semantics casually; dev intentionally differs from prod.

## NOTES
- `content-utils.ts` is one of the highest-leverage files in the repo; route behavior across home, archives, blog detail, about, and RSS depends on it.
- `url-utils.ts` mirrors `astro.config.mjs` i18n rules; keep them aligned if locale config changes.
