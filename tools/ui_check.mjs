// Browser checks for docs/index.html against a SYNTHETIC fixture site (never the live data).
// Usage: node tools/ui_check.mjs <site-dir>   (site-dir contains index.html and data/latest.json)
// Needs the `playwright` package (dev-only; e.g. a global install) and a Chromium build.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require("playwright")); } catch {
  try { ({ chromium } = require(path.join(process.execPath, "../../lib/node_modules/playwright"))); } catch {
    console.log("SKIP: playwright not installed"); process.exit(0);
  }
}
const site = path.resolve(process.argv[2] || "site");
let mode = "ok";   // ok | missing | schema | broken
const server = http.createServer((req, res) => {
  const url = req.url.split("?")[0];
  if (url.startsWith("/data/latest.json")) {
    if (mode === "missing") { res.writeHead(404); return res.end(); }
    if (mode === "schema") { res.writeHead(200, { "content-type": "application/json" }); return res.end(JSON.stringify({ display_schema: "asx_research_display_v99" })); }
    if (mode === "broken") { req.socket.destroy(); return; }
  }
  const f = path.join(site, url === "/" ? "index.html" : url);
  if (!fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "content-type": f.endsWith(".html") ? "text/html" : "application/json" });
  fs.createReadStream(f).pipe(res);
}).listen(0);
const base = `http://127.0.0.1:${server.address().port}/`;
const exe = fs.existsSync("/opt/pw-browsers/chromium") ? "/opt/pw-browsers/chromium" : undefined;
const browser = await chromium.launch(exe ? { executablePath: fs.realpathSync(exe) } : {});
const results = [];
const check = (name, ok, detail = "") => { results.push({ name, ok, detail }); console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`); };

async function page(width = 1200, opts = {}) {
  const ctx = await browser.newContext({ viewport: { width, height: 900 }, ...opts });
  await ctx.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.abort());
  const p = await ctx.newPage();
  p.on("pageerror", (e) => check("no page errors", false, e.message));
  return p;
}
const ready = (p) => p.waitForFunction(() => document.getElementById("load-state").hidden || document.getElementById("load-state").dataset.state !== "loading");

// 1. Today shows result, scenario, reason, limitation and no-execution status
let p = await page();
await p.goto(base + "#/today"); await ready(p);
const today = await p.evaluate(() => ({ h: document.getElementById("result-h").textContent, reason: document.getElementById("result-reason").textContent,
  limit: document.getElementById("result-limit").textContent, scen: document.getElementById("scenario").selectedOptions[0].text, body: document.body.innerText }));
check("Today: result, reason and limitation visible", !!(today.h && today.reason && today.limit), today.h);
check("Today: scenario visibly a simulation", /simulated model portfolio/i.test(today.body) && !!today.scen, today.scen);
check("Today: 'This program places no trades.' shown", today.body.includes("This program places no trades."));
check("Synthetic data flagged", /Synthetic demo data/.test(today.body));

// 2. Navigation, inspection never changes Today, Back/Forward and reload restore state
await p.click('nav a[data-view="performance"]');
await p.waitForSelector("#view-performance:not([hidden])");
check("Nav: URL reflects view", p.url().includes("#/performance"));
const btns = await p.$$('#hist-body button:has-text("View performance")');
await btns[2].click();
await p.waitForFunction(() => location.hash.includes("m="));
const insp = await p.evaluate(() => ({ flag: !document.getElementById("insp-flag").hidden, h: document.getElementById("insp-h").textContent, hist: document.getElementById("hist-h").textContent }));
check("Inspecting another method label shown", insp.flag, insp.h);
check("History chart follows the inspected method", insp.hist.includes(insp.h), insp.hist);
const inspUrl = p.url();
await p.click('nav a[data-view="today"]');
await p.waitForSelector("#view-today:not([hidden])");
const after = await p.evaluate(() => document.getElementById("result-h").textContent);
check("Inspecting a method does not change Today", after === today.h);
await p.goBack(); await p.waitForSelector("#view-performance:not([hidden])");
check("Back restores the inspected method", p.url() === inspUrl);
await p.reload(); await ready(p);
check("Reload restores public state", await p.evaluate(() => !document.getElementById("view-performance").hidden && location.hash.includes("m=") && !document.getElementById("insp-flag").hidden));

// 3. Keyboard sort keeps focus, sets aria-sort, announces; ranks unchanged
await p.focus('#hist-body button[data-sort="mdd_daily"]');
const rankMap = () => p.$$eval("#hist-body tbody tr", (t) => Object.fromEntries(t.map((x) => [x.dataset.method, x.cells[0].textContent])));
const ranksBefore = await rankMap();
await p.keyboard.press("Enter");
await p.waitForTimeout(100);
const sortState = await p.evaluate(() => ({ focus: document.activeElement && document.activeElement.dataset.sort, aria: document.activeElement.closest("th").getAttribute("aria-sort"), live: document.getElementById("announce").textContent }));
const ranksAfter = await rankMap();
check("Keyboard sort keeps focus on the header button", sortState.focus === "mdd_daily");
check("Sorted header exposes aria-sort", ["ascending", "descending"].includes(sortState.aria), sortState.aria);
check("Sort is announced", /Sorted by/.test(sortState.live), sortState.live);
const common = Object.keys(ranksAfter).filter((k) => k in ranksBefore);
check("Sorting does not change rank numbers", common.length > 0 && common.every((k) => ranksBefore[k] === ranksAfter[k]), `${common.length} rows compared`);
const totals = await p.evaluate(() => document.getElementById("hist-body").innerText.match(/Showing (\d+) of (\d+)/));
check("Top 10 initially with explicit total", totals && Number(totals[1]) <= 10 && Number(totals[2]) >= Number(totals[1]), totals && totals[0]);

// 4. Evidence view: groups, fallback notice, score not called a probability
await p.click("#tab-ev"); await p.waitForSelector("#panel-ev:not([hidden])");
const ev = await p.evaluate(() => document.getElementById("panel-ev").innerText);
check("Evidence view explains qualification first", ev.includes("Qualification first; evidence score second"));
check("No-qualifier fallback notice present", ev.includes("No stock method qualifies"));
check("Groups listed in order", ev.indexOf("Passed current research checks") < ev.indexOf("Paper tracking") && ev.indexOf("Paper tracking") < ev.indexOf("Exploratory / not qualified"));
check("Score never labelled a chance of profit", !/chance of profit\b(?! or)/i.test(ev.replace("not a chance of profit", "")));

// 5. Contrast of text in both themes (normal text >= 4.5:1)
async function contrastFails(pg) {
  return pg.evaluate(() => {
    const parse = (c) => (c.match(/[\d.]+/g) || []).map(Number);
    const lum = ([r, g, b]) => [r, g, b].map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }).reduce((a, v, i) => a + v * [0.2126, 0.7152, 0.0722][i], 0);
    const bg = (n) => { while (n && n.nodeType === 1) { const c = parse(getComputedStyle(n).backgroundColor); if (c.length >= 3 && (c.length < 4 || c[3] > 0.5)) return c; n = n.parentElement; } return parse(getComputedStyle(document.body).backgroundColor); };
    const bad = [];
    for (const n of document.querySelectorAll("main *, header *, footer *")) {
      if (!n.offsetParent || ![...n.childNodes].some((c) => c.nodeType === 3 && c.textContent.trim())) continue;
      const st = getComputedStyle(n); if (st.visibility === "hidden") continue;
      const a = lum(parse(st.color)), b = lum(bg(n));
      const ratio = (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
      const large = parseFloat(st.fontSize) >= 24 || (parseFloat(st.fontSize) >= 18.66 && Number(st.fontWeight) >= 700);
      if (ratio < (large ? 3 : 4.5)) bad.push(`${n.tagName}.${n.className}: ${ratio.toFixed(2)} "${n.textContent.trim().slice(0, 30)}"`);
    }
    return bad;
  });
}
for (const theme of ["light", "dark"]) {
  for (const view of ["today", "performance", "research", "how"]) {
    await p.evaluate((t) => { document.documentElement.dataset.theme = t; }, theme);
    await p.goto(base + `#/${view}`); await p.waitForTimeout(150);
    const bad = await contrastFails(p);
    check(`Contrast ${theme} ${view}`, bad.length === 0, bad.slice(0, 3).join(" | "));
  }
}
await p.evaluate(() => { document.documentElement.removeAttribute("data-theme"); });

// 6. Theme toggle keeps focus
await p.goto(base + "#/today"); await ready(p);
await p.focus("#theme-btn"); await p.keyboard.press("Enter"); await p.waitForTimeout(50);
check("Theme toggle keeps focus", await p.evaluate(() => document.activeElement.id === "theme-btn"));

// 7. Phone width / 400% zoom (320 CSS px) and 200% text: no page-level horizontal scroll
for (const [w, label, big] of [[320, "320px / 400% zoom", false], [640, "200% text", true]]) {
  const q = await page(w);
  for (const view of ["today", "performance", "research", "how"]) {
    await q.goto(base + `#/${view}`); await ready(q);
    if (big) await q.addStyleTag({ content: "html{font-size:200%}" });
    await q.waitForTimeout(100);
    const over = await q.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    check(`${label} ${view}: no horizontal page scroll`, over <= 1, `${over}px`);
  }
  await q.context().close();
}
// 8. Touch targets
const small = await p.evaluate(() => [...document.querySelectorAll("nav a, .btn, select, .linkbtn")].filter((n) => n.offsetParent && n.getBoundingClientRect().height < 43.5).map((n) => n.textContent.trim().slice(0, 20)));
check("Important touch targets >= 44px", small.length === 0, small.join(", "));
// 9. Landmarks and chart alternative
const lm = await p.evaluate(() => ({ main: !!document.querySelector("main#main"), nav: !!document.querySelector("nav[aria-label]"), skip: !!document.querySelector("a.skip[href='#main']") }));
check("Main, navigation and skip link present", lm.main && lm.nav && lm.skip);
await p.goto(base + "#/performance"); await ready(p);
check("Chart has a data-table alternative and keyboard slider", await p.evaluate(() => !!document.querySelector("#h-table table") && !!document.querySelector("#h-chart [role=slider][tabindex='0']")));

// 10. Failure states are distinct
for (const [m, text] of [["missing", "No results have been published yet"], ["schema", "unknown format"], ["broken", "Could not reach"]]) {
  mode = m; const q = await page(); await q.goto(base + "#/today"); await q.waitForTimeout(400);
  const t = await q.evaluate(() => document.getElementById("load-state").innerText);
  check(`State: ${m}`, t.includes(text) && t.includes("Try again"), t.slice(0, 80)); await q.context().close();
}
mode = "ok";
await browser.close(); server.close();
const failed = results.filter((r) => !r.ok);
console.log(`${results.length - failed.length}/${results.length} UI checks passed`);
process.exit(failed.length ? 1 : 0);
