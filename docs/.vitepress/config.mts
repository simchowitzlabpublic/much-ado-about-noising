import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Much Ado About Noising Documentation",
  description: "Much Ado About Noising - A PyTorch framework for behavior cloning with flow models",
  base: '/much-ado-about-noising/',
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Getting Started', link: '/getting-started/quick_start' },
      { text: 'Development', link: '/development/add_new_task' },
      { text: 'GitHub', link: 'https://github.com/simchowitzlabpublic/much-ado-about-noising' }
    ],

    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Quick Start', link: '/getting-started/quick_start' },
          { text: 'Architecture & Design', link: '/getting-started/design' },
          { text: 'Configuration', link: '/getting-started/configuration' }
        ]
      },
      {
        text: 'Development',
        items: [
          { text: 'Adding a New Task', link: '/development/add_new_task' },
          { text: 'Adding a New Method', link: '/development/add_new_method' },
          { text: 'Network Architecture', link: '/development/networks' }
        ]
      },
      {
        text: 'API Reference',
        items: [
          { text: 'Core Components', link: '/api/core' },
          { text: 'Losses & Samplers', link: '/api/losses' },
          { text: 'Networks', link: '/api/networks' },
          { text: 'Datasets', link: '/api/datasets' },
          { text: 'Environments', link: '/api/environments' }
        ]
      },
      {
        text: 'Help',
        items: [
          { text: 'Troubleshooting', link: '/help/troubleshooting' },
          { text: 'FAQ', link: '/help/faq' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/simchowitzlabpublic/much-ado-about-noising' }
    ],

    search: {
      provider: 'local'
    },

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2025-present'
    }
  }
})
