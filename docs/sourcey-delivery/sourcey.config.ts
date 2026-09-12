import { defineConfig, doxygen, markdown } from "sourcey";

const commit = "35a8adbfbafd0958c14a1797d710a5cb9559fd9a";

export default defineConfig({
  name: "iree-metal",
  siteUrl: "https://nikliolios.com",
  baseUrl: "/iree-metal/sourcey-docs",
  prettyUrls: false,
  repo: "https://github.com/niklio/iree-metal",
  theme: {
    preset: "default",
    colors: {
      primary: "#6d5dfc",
      light: "#8d82ff",
      dark: "#5145cd",
    },
  },
  navigation: {
    tabs: [
      {
        tab: "Guide",
        slug: "",
        source: markdown({
          groups: [
            {
              group: "Start here",
              pages: ["introduction", "architecture", "building"],
            },
          ],
        }),
      },
      {
        tab: "API reference",
        slug: "api",
        source: doxygen({
          xml: "./build/doxygen/xml",
          language: "cpp",
          index: "structured",
          sourceUrl: `https://github.com/niklio/iree-metal/blob/${commit}/{fullPath}#L{line}`,
        }),
      },
    ],
  },
  navbar: {
    links: [
      { type: "github", href: "https://github.com/niklio/iree-metal" },
      { label: "Nik Liolios", href: "https://nikliolios.com" },
    ],
    primary: {
      type: "button",
      label: "View source",
      href: `https://github.com/niklio/iree-metal/tree/${commit}`,
    },
  },
  footer: {
    links: [
      { label: "Apache-2.0 license", href: `https://github.com/niklio/iree-metal/blob/${commit}/LICENSE` },
      { label: "Pinned source revision", href: `https://github.com/niklio/iree-metal/tree/${commit}` },
    ],
  },
  changelog: false,
  search: {
    featured: ["introduction", "architecture", "building"],
  },
});
