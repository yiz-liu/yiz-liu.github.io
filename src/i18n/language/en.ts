import type { Translation } from "@i18n/key";

const translation: Translation = {
    header: {
        home: "Home",
        archive: "Archive",
        about: "About",
    },
    cover: {
        title: {
            home: "Yizhou's Notes",
            archive: "Archive",
            about: "About",
        },
        subTitle: {
            home: "Notes on AI infrastructure, inference systems, and engineering trade-offs.",
            archive: "Total of {count} articles",
            about: "A concise intro to this personal technical site",
        }
    },
    toc: "Contents",
    category: "Category",
    pageNavigation: {
        previous: "Prev",
        next: "Next",
        currentPage: "Page {currentPage} of {totalPages}",
    },
    button: {
        switchDarkMode: "Switch Dark Mode",
        backToTop: "Back to Top",
        backToBottom: "Back to Bottom",
        meun: "Menu",
        toc: "Contents",
    },
    search: {
        placeholder: "Enter keywords to start searching",
        noresult: "No results found.",
        error: "Search error occurred. Please try again later."
    },
    license: {
        author: "Author",
        license: "License",
        publishon: "Published on"
    },
    blogNavi: {
        next: "Next Blog",
        prev: "Previous Blog"
    },
    pagecard: {
        words: "words",
        minutes: "min read",
        uncategorized: "Uncategorized"
    },
    langNote: {
        note: "Note: ",
        description: "This page does not support English, using the default language version"
    },
    draftNote: {
        warning: "Draft Warning: ",
        description: "This article is a draft and only appears in the testing environment. It will not be displayed in the production environment."
    },
    page404: {
        title: "Page not found / 页面未找到",
        subTitle: "The page may have moved, or it does not exist yet. / 页面可能已移动，或尚未发布。",
        backToHome: "Home",
        backToPreview: "Previous Page",
        errorCode: "Error Code: 404",
        notice: "Perhaps you can try:"
    },
    themeInfo: {
        light: "Switch to Light Mode",
        dark: "Switch to Dark Mode",
        system: "Switch to System Mode"
    }
}

export default translation;
