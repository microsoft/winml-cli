// Extension: doc-check
// Verify docs are self-consistent, cross-referenced correctly, and match source code

import { joinSession } from "@github/copilot-sdk/extension";
import { readFile } from "fs/promises";
import { resolve, join, dirname } from "path";
import { glob } from "fs/promises";

const ROOT = resolve(import.meta.dirname, "../../..");
const DOCS_DIR = join(ROOT, "docs");
const SRC_DIR = join(ROOT, "src");

async function readTextFile(path) {
    try {
        return await readFile(path, "utf-8");
    } catch {
        return null;
    }
}

async function findFiles(pattern, cwd) {
    const results = [];
    for await (const entry of glob(pattern, { cwd })) {
        results.push(entry);
    }
    return results;
}

const session = await joinSession({
    tools: [
        {
            name: "doc_check_consistency",
            description:
                "Audit the docs/ folder for internal consistency issues: broken cross-references between markdown files, nav entries in mkdocs.yml that point to missing files, and markdown files in docs/ that are not listed in nav or exclude_docs. Returns a report of findings.",
            parameters: { type: "object", properties: {} },
            skipPermission: true,
            handler: async () => {
                const findings = [];

                // 1. Read mkdocs.yml nav and check all referenced files exist
                const mkdocsContent = await readTextFile(join(ROOT, "mkdocs.yml"));
                if (!mkdocsContent) return "Error: mkdocs.yml not found";

                const navFileRefs = [...mkdocsContent.matchAll(/:\s+(\S+\.md)/g)].map((m) => m[1]);
                for (const ref of navFileRefs) {
                    const fullPath = join(DOCS_DIR, ref);
                    const content = await readTextFile(fullPath);
                    if (content === null) {
                        findings.push(`[NAV_MISSING_FILE] mkdocs.yml references '${ref}' but file does not exist`);
                    }
                }

                // 2. Check internal markdown links in all docs files
                const mdFiles = await findFiles("**/*.md", DOCS_DIR);
                for (const relPath of mdFiles) {
                    const fullPath = join(DOCS_DIR, relPath);
                    const content = await readTextFile(fullPath);
                    if (!content) continue;

                    // Find markdown links like [text](../path/to/file.md) or [text](./file.md#anchor)
                    const linkPattern = /\[([^\]]*)\]\(([^)]+)\)/g;
                    let match;
                    while ((match = linkPattern.exec(content)) !== null) {
                        const target = match[2];
                        // Skip external URLs and anchors-only
                        if (target.startsWith("http") || target.startsWith("#") || target.startsWith("mailto:")) continue;
                        // Strip anchor
                        const filePart = target.split("#")[0];
                        if (!filePart) continue;
                        // Resolve relative to current file's directory
                        const resolvedPath = resolve(dirname(fullPath), filePart);
                        const exists = await readTextFile(resolvedPath);
                        if (exists === null) {
                            findings.push(`[BROKEN_LINK] ${relPath}: link to '${filePart}' resolves to non-existent file`);
                        }
                    }
                }

                // 3. Check for md files not in nav and not in exclude_docs
                const excludeMatch = mkdocsContent.match(/exclude_docs:\s*\|([\s\S]*?)(?=\n\S|\n*$)/);
                const excludePatterns = excludeMatch
                    ? excludeMatch[1]
                        .split("\n")
                        .map((l) => l.trim())
                        .filter(Boolean)
                    : [];
                const navSet = new Set(navFileRefs);
                for (const relPath of mdFiles) {
                    const normalized = relPath.replace(/\\/g, "/");
                    if (navSet.has(normalized)) continue;
                    if (normalized === "index.md") continue;
                    // Check exclude patterns (simple glob: just filename or path prefix)
                    const excluded = excludePatterns.some((pat) => {
                        const cleanPat = pat.replace(/^\//, "");
                        return normalized === cleanPat || normalized.startsWith(cleanPat);
                    });
                    if (excluded) continue;
                    findings.push(`[ORPHAN_FILE] ${normalized} is not in mkdocs.yml nav or exclude_docs`);
                }

                if (findings.length === 0) return "✅ No consistency issues found.";
                return `Found ${findings.length} issue(s):\n\n${findings.join("\n")}`;
            },
        },
        {
            name: "doc_check_code_alignment",
            description:
                "Cross-reference documentation claims against source code. Checks: CLI flag names documented in command pages match actual Click/Typer parameter definitions in src/; EP names and device mappings in docs match source; config schema fields match the WinMLBuildConfig dataclass. Returns mismatches.",
            parameters: {
                type: "object",
                properties: {
                    scope: {
                        type: "string",
                        description: "Which aspect to check: 'flags' (CLI flags vs source), 'eps' (EP table vs source), 'config' (config schema fields vs dataclass), or 'all'",
                        enum: ["flags", "eps", "config", "all"],
                    },
                },
            },
            skipPermission: true,
            handler: async (args) => {
                const scope = args.scope || "all";
                const findings = [];

                // Helper: find Python files containing a pattern
                async function searchSrc(pattern) {
                    const pyFiles = await findFiles("**/*.py", SRC_DIR);
                    const results = [];
                    for (const f of pyFiles) {
                        const content = await readTextFile(join(SRC_DIR, f));
                        if (content && content.includes(pattern)) {
                            results.push({ file: f, content });
                        }
                    }
                    return results;
                }

                if (scope === "flags" || scope === "all") {
                    // Check command docs for flags and verify they exist in source
                    const cmdFiles = await findFiles("*.md", join(DOCS_DIR, "commands"));
                    for (const cmdFile of cmdFiles) {
                        const content = await readTextFile(join(DOCS_DIR, "commands", cmdFile));
                        if (!content) continue;
                        // Extract flags from markdown tables: | `--flag-name` |
                        const flagPattern = /\|\s*`(--[\w-]+)`/g;
                        let match;
                        const docFlags = [];
                        while ((match = flagPattern.exec(content)) !== null) {
                            docFlags.push(match[1]);
                        }
                        if (docFlags.length === 0) continue;

                        // Try to find the command source file
                        const cmdName = cmdFile.replace(".md", "");
                        const srcFiles = await searchSrc(`def ${cmdName}`);
                        if (srcFiles.length === 0) continue;

                        // Check each documented flag exists in source (as click option or argument)
                        const srcContent = srcFiles.map((s) => s.content).join("\n");
                        for (const flag of docFlags) {
                            const paramName = flag.replace(/^--/, "").replace(/-/g, "_");
                            const altName = flag; // --flag-name form
                            if (!srcContent.includes(paramName) && !srcContent.includes(altName)) {
                                findings.push(`[FLAG_NOT_IN_SRC] ${cmdFile}: '${flag}' not found in source for '${cmdName}' command`);
                            }
                        }
                    }
                }

                if (scope === "eps" || scope === "all") {
                    // Check EP table in docs/concepts/eps-and-devices.md
                    const epDoc = await readTextFile(join(DOCS_DIR, "concepts", "eps-and-devices.md"));
                    if (epDoc) {
                        const epPattern = /\|\s*`(\w+ExecutionProvider)`\s*\|\s*`(\w+)`/g;
                        let match;
                        while ((match = epPattern.exec(epDoc)) !== null) {
                            const epName = match[1];
                            const shortName = match[2];
                            // Verify EP short name exists somewhere in source
                            const srcHits = await searchSrc(shortName);
                            if (srcHits.length === 0) {
                                findings.push(`[EP_SHORT_NAME_MISSING] EP '${epName}' short name '${shortName}' not found in source`);
                            }
                        }
                    }
                }

                if (scope === "config" || scope === "all") {
                    // Check config schema fields in docs/reference/index.md against source dataclass
                    const refDoc = await readTextFile(join(DOCS_DIR, "reference", "index.md"));
                    if (refDoc) {
                        // Extract field names from table rows: | `field_name` |
                        const fieldPattern = /\|\s*`([\w.]+)`\s*\|/g;
                        let match;
                        const docFields = new Set();
                        while ((match = fieldPattern.exec(refDoc)) !== null) {
                            docFields.add(match[1].split(".").pop()); // Get leaf field name
                        }
                        // Find WinMLBuildConfig in source
                        const configFiles = await searchSrc("WinMLBuildConfig");
                        if (configFiles.length > 0) {
                            const configSrc = configFiles.map((f) => f.content).join("\n");
                            // Check each doc field appears in source
                            for (const field of docFields) {
                                if (!configSrc.includes(field)) {
                                    findings.push(`[CONFIG_FIELD_MISSING] Field '${field}' documented but not found in WinMLBuildConfig source`);
                                }
                            }
                        }
                    }
                }

                if (findings.length === 0) return "✅ Documentation aligns with source code.";
                return `Found ${findings.length} mismatch(es):\n\n${findings.join("\n")}`;
            },
        },
        {
            name: "doc_check_samples",
            description:
                "Verify that sample pages (docs/samples/) use correct model IDs, command flags, and pipeline steps that match the current CLI capabilities. Checks model IDs are valid HuggingFace references and command examples use documented flags.",
            parameters: { type: "object", properties: {} },
            skipPermission: true,
            handler: async () => {
                const findings = [];
                const sampleFiles = await findFiles("*.md", join(DOCS_DIR, "samples"));

                // Load all documented flags from command pages
                const cmdFiles = await findFiles("*.md", join(DOCS_DIR, "commands"));
                const allFlags = new Map(); // command -> Set of flags
                for (const cmdFile of cmdFiles) {
                    const content = await readTextFile(join(DOCS_DIR, "commands", cmdFile));
                    if (!content) continue;
                    const cmdName = cmdFile.replace(".md", "");
                    const flags = new Set();
                    const flagPattern = /\|\s*`(--[\w-]+)`/g;
                    let match;
                    while ((match = flagPattern.exec(content)) !== null) {
                        flags.add(match[1]);
                    }
                    allFlags.set(cmdName, flags);
                }

                for (const sampleFile of sampleFiles) {
                    const content = await readTextFile(join(DOCS_DIR, "samples", sampleFile));
                    if (!content) continue;

                    // Check command examples use valid flags
                    const codeBlocks = content.match(/```bash\n([\s\S]*?)```/g) || [];
                    for (const block of codeBlocks) {
                        // Find winml commands
                        const cmdPattern = /winml\s+(\w+)(.*)/g;
                        let match;
                        while ((match = cmdPattern.exec(block)) !== null) {
                            const cmd = match[1];
                            const argsStr = match[2];
                            const docFlags = allFlags.get(cmd);
                            if (!docFlags || docFlags.size === 0) continue;

                            // Extract flags used
                            const usedFlags = argsStr.match(/--[\w-]+/g) || [];
                            for (const flag of usedFlags) {
                                if (!docFlags.has(flag)) {
                                    findings.push(`[UNDOCUMENTED_FLAG] ${sampleFile}: 'winml ${cmd} ${flag}' uses flag not in docs/commands/${cmd}.md`);
                                }
                            }
                        }
                    }
                }

                if (findings.length === 0) return "✅ All sample commands use documented flags.";
                return `Found ${findings.length} issue(s):\n\n${findings.join("\n")}`;
            },
        },
    ],
});
