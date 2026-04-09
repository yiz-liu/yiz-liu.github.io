# Yizhou's Notes

Personal notes on AI infrastructure, inference systems, and engineering trade-offs.

Originally derived from [Momo](https://github.com/Motues/Momo), now maintained as Yizhou's Notes.

## Development

```bash
pnpm install
pnpm dev
```

## Build

```bash
pnpm build
pnpm preview
```

## Create a new post

```bash
pnpm newpost <folder> [lang]
```

Example:

```bash
pnpm newpost vllm/paged-attention zh-cn
pnpm newpost vllm/paged-attention en
```

The script creates files under `src/content/blog/<folder>/<lang>.md` using the current frontmatter schema.
