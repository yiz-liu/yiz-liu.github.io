import type {
    SiteConfig,
    ProfileConfig,
    LicenseConfig
} from "./types/config"

export const siteConfig: SiteConfig = {
    title: "Yizhou's Notes",
    subTitle: "Notes on AI infrastructure, inference systems, and engineering trade-offs.",

    favicon: "/favicon/favicon.svg", // Path of the favicon, relative to the /public directory

    pageSize: 6, // Number of posts per page
    toc: {
        enable: true,
        depth: 3 // Max depth of the table of contents, between 1 and 4
    },
    blogNavi: {
        enable: true // Whether to enable blog navigation in the blog footer
    }
}

export const profileConfig: ProfileConfig = {
    avatar: "/favicon/favicon.ico", // Relative to the /src directory. Relative to the /public directory if it starts with '/'
    name: "Yizhou Liu",
    description: "Notes on AI infrastructure, inference systems, and engineering trade-offs.",
    indexPage: "https://yiz-liu.github.io/",
    startYear: 2026,
}

export const licenseConfig: LicenseConfig = {
	enable: true,
	name: "CC BY-NC-SA 4.0",
	url: "https://creativecommons.org/licenses/by-nc-sa/4.0/",
};
