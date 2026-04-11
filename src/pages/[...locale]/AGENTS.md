# ROUTE BOUNDARY NOTES

## OVERVIEW
Primary localized route boundary for user-facing pages.

## STRUCTURE
```text
src/pages/[...locale]/
├── [...page].astro   # localized home + pagination
├── about.astro       # localized spec-backed singleton page
├── archives.astro    # localized archive page with Svelte island
└── blog/[...id].astro # localized blog detail pages
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Home pagination | `[...page].astro` | paginates `getBlogEntrySort()` output |
| Blog detail routing | `blog/[...id].astro` | route ids come from normalized folder path |
| About page content | `about.astro` | loads `getSpec(currentLang, 'about')` |
| Archive behavior | `archives.astro` | passes data to `ArchivePanel.svelte` |
| Locale URL rules | `src/utils/url-utils.ts` | use helper, never build prefixes manually |
| Translation fallback | `src/utils/content-utils.ts` | blog/spec fallback behavior |

## CONVENTIONS
- Every route resolves `currentLang` from `Astro.currentLocale || i18n!.defaultLocale`.
- `getStaticPaths()` usually maps all locales but passes `undefined` for the default locale param.
- Route files call shared helpers (`getBlogEntrySort`, `getSpec`, `getRelativeLocaleUrl`) instead of duplicating locale logic.
- `MainPageLayout` is the standard wrapper for all rendered pages here.
- Blog detail pages surface `draft` and `isFallback` state in the UI; keep that behavior when editing route logic.

## ANTI-PATTERNS
- Do not replace this boundary with flat per-locale route files without also updating URL helpers and fallback logic.
- Do not hardcode `/en/` or `/zh-cn/` path prefixes; default locale intentionally has no prefix.
- Do not read content collections directly from every route when a shared utility already owns the behavior.
- Do not treat `entry.id` as a filename; here it is the normalized folder slug.

## NOTES
- `about.astro` throws if `spec/about/<locale>` and default fallback are both missing.
- `archives.astro` uses `client:only="svelte"` for archive filtering, so route data shape must stay compatible with `ArchivePanel.svelte`.
- `rss.xml.ts` lives outside this folder but follows the same content-selection assumptions.
