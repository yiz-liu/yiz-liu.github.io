export interface Translation {
    header: {
        home: string;
        archive: string;
        about: string;
    };
    cover: {
        title: {
            home: string;
            archive: string;
            about: string;
        };
        subTitle: {
            home: string;
            archive: string;
            about: string;
        };
    };
    toc:string;
    category: string;
    pageNavigation: {
        previous: string;
        next: string;
        currentPage: string;
    };
    button: {
        switchDarkMode: string;
        backToTop: string;
        backToBottom: string;
        meun: string;
        toc: string;
    }
    search: {
        placeholder: string;
        noresult: string;
        error: string;
    };
    license: {
        author: string;
        license: string;
        publishon: string;
    };
    blogNavi: {
        next: string;
        prev: string;
    },
    pagecard: {
        words: string;
        minutes: string;
        uncategorized: string;
    }
    langNote: {
        note: string;
        description: string;
    },
    draftNote: {
        warning: string;
        description: string;
    },
    page404: {
        title: string;
        subTitle: string;
        backToHome: string;
        backToPreview: string;
        errorCode: string;
        notice: string;
    },
    themeInfo: {
        light: string;
        dark: string;
        system: string;
    }
}
