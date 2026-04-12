// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from "@tailwindcss/vite";
import icon from 'astro-icon';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { remarkReadingTime } from './src/plugins/remark-reading-time.mjs';
import mermaid from 'astro-mermaid';
import astroD2 from 'astro-d2';

import svelte from "@astrojs/svelte";
export default defineConfig({
  site: 'https://yiz-liu.github.io/', // Root URL of site
  i18n: {
    locales: ['zh-cn', 'en'],
    defaultLocale: 'zh-cn',
    routing: {
      prefixDefaultLocale: false,
      redirectToDefaultLocale: false
    }
  },
  integrations: [icon({
    include: {
      "fa6-brands": ["*"],
      "fa6-solid": ["*"],
      "simple-icons": ["*"],
      "vscode-icons": ["*"],
      "material-symbols": ["*"]
    }
  }), svelte(), mermaid({ 
    theme: 'forest',
    autoTheme: true
  }), astroD2({
    theme: { default: '0', dark: '200' },
    inline: false,
    experimental: { useD2js: true }
  })],
  markdown: {
    shikiConfig: {
      theme: 'one-dark-pro', // code theme
      // theme: 'github-dark',
      wrap: false
    },
    remarkPlugins: [remarkMath, remarkReadingTime],
    rehypePlugins: [rehypeKatex]
  },
  vite: {
    plugins: [tailwindcss()]
  }
});
