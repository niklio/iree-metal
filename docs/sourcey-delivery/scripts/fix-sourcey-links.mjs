import { readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

async function htmlFiles(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await htmlFiles(path));
    if (entry.isFile() && path.endsWith(".html")) files.push(path);
  }
  return files;
}

let replacements = 0;
for (const file of await htmlFiles("dist")) {
  const before = await readFile(file, "utf8");
  const after = before.replaceAll("api/index.html", "api.html");
  if (after !== before) {
    replacements += before.split("api/index.html").length - 1;
    await writeFile(file, after);
  }
}

console.log(`Normalized ${replacements} API landing-page links.`);
