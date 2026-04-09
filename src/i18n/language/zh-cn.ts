import type { Translation } from "@i18n/key";

const translation: Translation = {
    header: {
        home: "首页",
        archive: "归档",
        about: "关于",
    },
    cover: {
        title: {
            home: "Yizhou's Notes",
            archive: "文章归档",
            about: "关于",
        },
        subTitle: {
            home: "记录 AI 基础设施、推理系统与工程权衡",
            archive: "共 {count} 篇文章",
            about: "个人技术博客简介",
        }
    },
    toc: "目录",
    category: "分类",
    pageNavigation: {
        previous: "上一页",
        next: "下一页",
        currentPage: "第 {currentPage} 页，共 {totalPages} 页",
    },
    button: {
        switchDarkMode: "切换明暗模式",
        backToTop: "回到顶部",
        backToBottom: "回到底部",
        meun: "菜单",
        toc: "目录",
    },
    search: {
        placeholder: "输入关键词开始搜索",
        noresult: "未找到相关结果",
        error: "搜索出现错误，请稍后重试"
    },
    license: {
        author: "作者",
        license: "许可协议",
        publishon: "发布时间"
    },
    blogNavi: {
        next: "下一篇",
        prev: "上一篇"
    },
    pagecard: {
        words: "字",
        minutes: "分钟",
        uncategorized: "未分类"
    },
    langNote: {
        note: "注意：",
        description: "当前页面不支持简体中文，使用默认语言版本"
    },
    draftNote: {
        warning: "草稿警告：",
        description: "此文章为草稿，只出现在测试环境，生产环境将不会显示。"
    },
    page404: {
        title: "Page not found / 页面未找到",
        subTitle: "The page may have moved, or it does not exist yet. / 页面可能已移动，或尚未发布。",
        backToHome: "返回首页",
        backToPreview: "返回上一页",
        errorCode: "错误代码：404",
        notice: "你可以试试："
    },
    themeInfo: {
        light: "切换到 浅色 模式",
        dark: "切换到 深色 模式",
        system: "切换到 跟随系统 模式"
    }
}

export default translation;
