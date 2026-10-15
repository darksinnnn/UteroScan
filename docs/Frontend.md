# UteroScan — Frontend Design & Build Plan

**Companion to:** `PRD.md`, `Architecture.md`, `Implementation_Plan.md`, `Agents.md`
**Status:** This unblocks the previously-deferred Phase C (UI) from `PRD.md` Section 2. Backend model (V1, `resnet50_stage2_best.pth`) is locked and ready to serve — this document plans the frontend only.

---

## 0. The Core Instruction — Read This First, Antigravity

**The single biggest risk on this phase is producing something that reads as "AI-generated" or "template-assembled."** Every decision below exists to prevent that. Before writing a single line of UI code:

- Do not use default Tailwind/Bootstrap component shapes verbatim (default card shadows, default button radii, default spacing scale). Every component needs at least one deliberate, non-default choice.
- Do not center everything. Do not make every section symmetric. Real design has intentional asymmetry, varied section rhythm, and generous unclaimed whitespace.
- Do not reach for blue/teal medical-clinical colors by default. See Section 1 — this is a deliberate departure.
- If a section looks like it could be swapped into any generic SaaS/health startup template, it has failed the brief. Every section should feel specific to *this* project.
- When unsure, look at fewer, higher-quality references rather than pattern-matching to "healthcare website" clichés.

---

## 1. Visual Direction — Palette & Typography

### Why beige, not clinical blue/teal
Healthcare web design in 2025–2026 has been visibly moving away from the "hospital blue and white" default — the sites that read as more premium and differentiated are the ones using warm neutrals, beige, blush, and soft earth tones instead. That's the direction here: **warm, calm, editorial** — closer to a premium wellness/research brand than a hospital portal. This also happens to suit a diagnostic *AI tool* well: it should feel considered and trustworthy, not sterile or alarmist.

### Palette (build the actual hex values from this direction, don't reuse it verbatim)
- **Base/background:** warm off-white to light beige — think unbleached paper, oat, linen. Avoid pure white (`#FFFFFF`) — it reads as generic/default. Something like a warm `#F7F3EC` / `#F2EBE0` range.
- **Secondary surface:** a slightly deeper beige/sand for cards and section breaks — enough contrast to separate content without a hard border or drop shadow.
- **Ink/text:** near-black warm charcoal, not pure `#000000` — something like a deep espresso/graphite (`#211E1A` range) for primary text, with a muted warm gray for secondary text.
- **Accent (pick exactly one, used sparingly):** a single confident accent color for CTAs, active states, and data highlights — candidates: a deep clinical-but-warm terracotta/rust, a muted forest green, or a deep navy used *only* as an accent (not a base color). Do not use more than one accent hue site-wide.
- **Data/diagnostic color (separate from the brand accent):** reserve a distinct, deliberately different color pairing for anything showing UF/NUF results (e.g. confidence bars, result badges) so diagnostic output is never visually confused with marketing UI. This should feel clinical-precise even while the rest of the site feels warm.

### Typography
- One serif or high-contrast display face for headlines (this is what makes a beige palette feel premium rather than flat) + one clean grotesk/sans for body and UI text. Do not use a single font family for everything — that's a strong "template" tell.
- Generous line-height on body copy (1.5–1.7), tight tracking on large display headlines.
- Avoid default system font stacks and avoid the current over-used "Inter everywhere" look — pick something with more character for headlines specifically.

### What to explicitly avoid
- Purple-to-blue gradients (the single most obvious "AI-generated startup site" tell in 2024–2026)
- Glassmorphism/frosted-glass cards as a primary motif
- Overly rounded (pill-shaped) buttons everywhere — pick one consistent radius language and stick to it
- Emoji as icons
- Stock medical photography (stethoscopes, generic doctor-patient handshake photos) — use abstract/editorial imagery, real anonymized preprocessing-pipeline visuals (you already have great ones — the before/after CLAHE comparisons, the ROC curves, styled properly), or custom illustration instead

---

## 2. Research Task for Antigravity (Do This Before Building)

Before writing code, gather concrete visual references so the design is grounded in real, current work rather than generic AI-guessed aesthetics:

1. Search Pinterest, Awwwards, Dribbble, and Land-book for: "beige minimal SaaS website", "editorial medical website design", "warm neutral dashboard UI", "health tech product page 2026". Save 8–12 reference screenshots.
2. Specifically look at how premium health/wellness/biotech sites (e.g. modern fertility/diagnostics startups, longevity/health-data companies) use beige/neutral palettes with a single accent color — this is a strong current pattern worth studying directly rather than reinventing.
3. Look at how Rive and Spline are used on real product sites (their own showcase pages are good references) to understand *tasteful, restrained* use — a common mistake is over-using 3D/animation until it feels like a demo reel rather than a product.
4. Report back a short mood-board summary (not full images necessarily, but concrete notes: "site X uses a diagonal-cut hero section," "site Y uses a single looping Rive icon next to each feature, not full-screen animation") before starting implementation, so we can align on direction before code is written.

---

## 3. Motion & Interaction Stack

### Lenis (smooth scroll) — foundation layer
- Use for the base scroll experience site-wide: `npm i lenis`, driven via `requestAnimationFrame`.
- Keep `lerp`/`duration` subtle (default ~0.1 lerp, ~1.2s duration is a reasonable starting point) — the goal is a slightly weighted, premium scroll feel, not a sluggish or disorienting one. Over-tuned smooth-scroll is another common "trying too hard" tell.
- Pair with `IntersectionObserver`-driven scroll reveals (fade/slide-in for sections as they enter viewport) — Lenis natively supports IntersectionObserver, so use that rather than a heavier scroll-animation library on top.
- Respect `prefers-reduced-motion` — disable/soften smooth scroll and scroll-triggered animation for users who request it. This is a real accessibility requirement, not optional polish.

### Rive — for small, purposeful interactive animation
Use Rive for **small, contained, meaningful animations**, not decoration for its own sake:
- An animated icon set for feature callouts (e.g. a small looping/hover-triggered icon next to "Preprocessing," "AI Classification," "Confidence Score" feature blocks)
- A subtle animated state indicator for the upload flow (idle → uploading → processing → result), so the AI pipeline feels alive and responsive during actual inference, not just a static spinner
- Do NOT use Rive for full-page hero animation unless it's genuinely central to explaining the product — restraint is the differentiator here

### Spline — for one deliberate 3D moment, not everywhere
- One 3D element, most likely in the hero section — something abstract that evokes ultrasound/medical imaging/scanning without being a literal 3D stethoscope or hospital scene. Think: an abstract representation of a scan sweep, a soft-shaded volumetric form, or a subtle particle/wave field that ties to "signal → clarity" as a visual metaphor for what the AI pipeline does (noisy input → clean diagnostic signal).
- Keep it performant: export appropriately, lazy-load it, provide a static fallback image for slow connections/reduced-motion users, and don't let it block first paint.
- One Spline scene site-wide is enough. Repeating 3D elements in every section is the fastest way to make a site feel like a design-tool demo rather than a product.

---

## 4. Site Structure & Pages

This is a capstone/academic diagnostic tool demo, not a commercial product — the structure should reflect that honestly while still looking professional.

1. **Landing / Home**
   - Hero: project name, one-line positioning ("AI-assisted uterine fibroid detection from ultrasound imaging"), the Spline moment, primary CTA to the demo/upload flow
   - A short, honest "How it works" section (Upload → Preprocess → Classify → Result) — this maps directly to your locked pipeline (`Architecture.md` Section 1) and is a natural place for the Rive icon set
   - A results/credibility section: key metrics from Phase 4 (accuracy, sensitivity, AUC) presented as clean data visuals, not hidden in prose — this is genuinely your strongest asset, don't bury it
   - Methodology/transparency note — a condensed, non-technical version of the NUF definition note from `PRD.md` Section 10, since a real user (or evaluator) will ask "what does this actually classify"
   - Clear academic/non-clinical disclaimer (per `PRD.md` Section 1 — proof-of-concept, not a certified medical device) — this should be visible, not buried in a footer link
2. **Try It / Demo (the actual product surface)**
   - Upload interface wired to the `predict_image()` function from Phase 5 (`predict_uteroscan.py`) via a backend API
   - Shows: uploaded image, (optionally) the preprocessed version for transparency, classification label, confidence score, and — given your τ=0.65 optimal-threshold findings — consider surfacing confidence visually (e.g. a calibrated bar/gauge) rather than just a binary label, since you now have the data to justify this
   - Clear inline messaging for unsupported files / errors (FR-5)
3. **About / Methodology**
   - Honest project narrative: dataset, cleaning process, architecture choice, evaluation — this is a great place to actually show off the rigor of this project (the contamination-cleaning story, the threshold analysis, the V1-vs-V2 robustness check) in a readable, non-dashboard-dump way
4. **(Optional) Report/Results page**
   - A cleaner, designed presentation of your Phase 4 metrics, confusion matrix, and ROC curve — reusing the actual generated assets from `reports/`, restyled to match the site's visual language rather than pasted in as raw matplotlib output

---

## 5. Technical Stack Recommendation

- **Framework:** React-based (Next.js is a reasonable default for a project this size — static-friendly for marketing pages, with API routes available for the inference endpoint)
- **Styling:** Tailwind is fine as a utility layer, but the design tokens (colors, radii, spacing, type scale) must be fully customized per Section 1 — do not ship default Tailwind theme values
- **Backend bridge:** a thin API route wrapping `predict_image()` — reuses Phase 5's inference function directly rather than reimplementing anything
- **Animation libraries:** `lenis`, `@rive-app/react-canvas` (or equivalent), Spline's React runtime (`@splinetool/react-spline`) — all three are lightweight enough not to be a performance concern if used per Section 3's restraint guidance

---

## 6. Definition of Done for This Phase

- Palette and type system implemented as design tokens (not hardcoded per-component), reviewed against Section 1 before broader build-out
- Lenis smooth scroll live site-wide, respects `prefers-reduced-motion`
- At least one Rive interaction and one Spline scene implemented per Section 3's restraint guidance — not more without a specific reason
- Upload → preprocess → classify → result flow fully wired to the real Phase 5 model, not mocked data
- Mobile-responsive (health-related searches skew heavily mobile — this is a real usability requirement, not a nice-to-have)
- A second pair of eyes (you) reviews it against the "would this pass for a template" test in Section 0 before considering this phase complete
