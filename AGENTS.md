# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-11 Asia/Shanghai
**Commit:** 57ce96b
**Branch:** main

## OVERVIEW
Astro-based personal notes site for AI infrastructure and inference systems. Core stack: Astro 6, TypeScript, selective Svelte islands, Tailwind v4, Astro content collections, Pagefind search, GitHub Pages deploy.

## STRUCTURE
```text
./
├── src/                    # app code, content collections, routes, layouts, utilities
│   ├── pages/[...locale]/  # locale-scoped routes; default locale omitted from URL
│   ├── content/blog/       # blog authoring tree: <slug>/<locale>.md
│   ├── content/spec/       # singleton content pages such as about/<locale>.md
│   ├── layouts/            # global document shell + page chrome wrapper
│   ├── utils/              # locale-aware content, URL, and date helpers
│   └── components/         # Astro components plus small Svelte islands
├── script/                 # authoring helper: new post generator
├── public/                 # static public assets
└── .github/workflows/      # GitHub Pages deployment
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Build/dev commands | `package.json` | `build` includes Pagefind indexing |
| Framework behavior | `astro.config.mjs` | i18n, Svelte integration, markdown plugin |
| Site-wide settings | `src/config.ts` | title, page size, profile, license |
| Localized routes | `src/pages/[...locale]/` | primary page boundary |
| Blog/spec schemas | `src/content.config.ts` | blog schema, loader rules |
| Blog/spec lookup logic | `src/utils/content-utils.ts` | locale fallback, grouping, draft filtering |
| Locale-aware links/assets | `src/utils/url-utils.ts` | URL prefixing, blog asset paths |
| Translation contract | `src/i18n/key.ts` + `src/i18n/language/*` | typed nested keys |
| Global shell/chrome | `src/layouts/Layout.astro` + `src/layouts/MainPageLayout.astro` | router, loading overlay, header/footer/search |
| Search behavior | `src/components/Header.astro` + `src/components/misc/Search.astro` | lazy Pagefind bootstrap |
| Post authoring workflow | `script/newpost.js` | canonical way to create blog files |

## CODE MAP
| Symbol / Entry | Type | Location | Role |
|----------------|------|----------|------|
| `getBlogEntrySort` | function | `src/utils/content-utils.ts` | group multilingual posts, filter drafts, mark fallback |
| `getSpec` | function | `src/utils/content-utils.ts` | load spec page with default-locale fallback |
| `getRelativeLocaleUrl` | function | `src/utils/url-utils.ts` | build locale-aware links |
| `[...page].astro#getStaticPaths` | route entry | `src/pages/[...locale]/[...page].astro` | paginate localized home/index pages |
| `[...id].astro#getStaticPaths` | route entry | `src/pages/[...locale]/blog/[...id].astro` | generate localized blog detail pages |
| `about.astro#getStaticPaths` | route entry | `src/pages/[...locale]/about.astro` | localized singleton page from `spec/about/*` |
| `archives.astro#getStaticPaths` | route entry | `src/pages/[...locale]/archives.astro` | localized archive page using Svelte island |
| `Layout.astro` | layout | `src/layouts/Layout.astro` | global head, theme bootstrap, Astro transitions |
| `MainPageLayout.astro` | layout | `src/layouts/MainPageLayout.astro` | shared header/footer/search wrapper |
| `rss.xml.ts` | endpoint | `src/pages/rss.xml.ts` | RSS from default-locale blog entries |

## CONVENTIONS
- Default locale is `zh-cn`; URLs for the default locale are intentionally unprefixed (`prefixDefaultLocale: false`).
- Route code usually derives `currentLang` from `Astro.currentLocale || i18n!.defaultLocale` and then calls `i18nit(currentLang)`.
- Use path aliases (`@components`, `@utils`, `@layouts`, `@i18n`, `@styles`, `@/*`) instead of deep relative imports.
- Tailwind v4 is used with CSS variables and arbitrary values; there is no visible Tailwind config file.
- Blog entries live under `src/content/blog/<slug>/<locale>.md`; spec pages live under `src/content/spec/<name>/<locale>.md`.
- Missing non-default translations are allowed; UI surfaces fallback state rather than failing build-time.
- Astro pages/layouts own the shell; Svelte is reserved for islands such as TOC and archive filtering.

## ANTI-PATTERNS (THIS PROJECT)
- Do not edit generated or ignored output: `dist/`, `.astro/`, `node_modules/`.
- Do not debug search from `pnpm dev`; Pagefind only exists after `pnpm build && pnpm preview`.
- Do not hand-roll locale URLs; use `getRelativeLocaleUrl()`.
- Do not assume every post/spec has both locales; fallback-to-default is intentional.
- Do not bypass `script/newpost.js` when creating blog posts unless you also preserve the required schema/path shape.
- Do not add client scripts that only initialize on `DOMContentLoaded`; `ClientRouter` means `astro:page-load` / `astro:after-swap` matter.

## UNIQUE STYLES
- Chinese-first repo: default content, comments, and route assumptions center on `zh-cn`.
- Page chrome is composed centrally through `MainPageLayout.astro`, not repeated per page.
- Search is lazy-loaded from `Header.astro` and consumed by `Search.astro` through custom events and shared window state.
- Markdown rendering is heavily enhanced: spoilers, copy buttons, PhotoSwipe wrapping, Pagefind body markers.
- Reading-time metadata comes from a custom remark plugin, then appears in cards and blog detail pages.

## COMMANDS
```bash
pnpm install
pnpm dev
pnpm build
pnpm preview
pnpm newpost <folder> [lang]
```

For non-interactive shells in this environment, `node` / `npm` / `pnpm` may be unavailable until nvm is sourced. Use:

```bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
```

After that, project commands like `pnpm build` should resolve normally.

## NOTES
- Deploy is GitHub Actions -> GitHub Pages on push to `main` via `.github/workflows/deploy.yml`.
- There is no dedicated `test`, `lint`, `typecheck`, or `check` script; `pnpm build` is the main validation path.
- `src/content/spec` looks like “specs” but is site content, not tests.

## ROUTE-BOUNDARY SPECIFICS (`src/pages/[...locale]`)
- `getStaticPaths()` in this boundary usually maps all locales but passes `undefined` for the default-locale param.
- Do not replace this boundary with flat per-locale route files without also updating URL helpers and fallback logic.
- Do not treat `entry.id` as a filename here; it is the normalized folder slug.
- `archives.astro` uses `client:only="svelte"`, so its route data shape must stay compatible with `ArchivePanel.svelte`.
- `about.astro` throws if both `spec/about/<locale>` and the default-locale fallback are missing.

## KNOWLEDGE-BASE FILES
- Child knowledge-base files exist at `src/content/blog` and `src/utils` for local rules.
