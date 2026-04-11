# BLOG AUTHORING NOTES

## OVERVIEW
Multilingual blog content tree; one folder per post, one markdown file per locale.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add new post | `script/newpost.js` | canonical generator |
| Validate frontmatter | `src/content.config.ts` | blog Zod schema |
| Understand route identity | `src/utils/content-utils.ts` | groups sibling locale files by folder |
| Understand asset resolution | `src/utils/url-utils.ts` | `blogCoverUrl()` joins folder slug + relative image |
| Example content | `graphs-in-vllm-ascend/zh-cn.md` | representative single-locale post |

## CONVENTIONS
- Path shape is `src/content/blog/<post-slug>/<locale>.md`.
- Locale comes from the filename (`zh-cn.md`, `en.md`), not frontmatter.
- Required frontmatter: `title`, `pubDate`, `slugId`; optional but expected fields: `description`, `image`, `category`, `draft`.
- `slugId` should match the folder-based logical slug used for the post.
- Missing non-default translations are allowed; utilities fall back to the default locale and mark the entry as fallback.
- Relative cover/image paths are resolved relative to the post folder.

## ANTI-PATTERNS
- Do not flatten posts into `src/content/blog/<locale>.md`; the folder boundary is meaningful.
- Do not rename locale files to non-locale names.
- Do not create underscore-prefixed markdown files if you expect Astro to load them; the loader excludes them.
- Do not assume drafts behave the same in dev and prod; prod filters `draft: true` out.
- Do not omit frontmatter fields that the schema or UI expects.

## NOTES
- `pnpm newpost <folder> [lang]` creates the expected folder/file shape and valid starter frontmatter.
- If a post exists only in `zh-cn`, the English route may still render the default-locale content with fallback messaging.
- The current tree is small, but the logic already assumes sibling translations under a shared folder slug.
