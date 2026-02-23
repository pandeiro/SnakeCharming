# SnakeCharming - Technology Stack

## Overview

SnakeCharming is a **zero-dependency static web application** designed for maximum portability and ease of deployment. The technology choices prioritize simplicity, performance, and accessibility.

---

## Core Technologies

### Frontend Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Markup** | HTML5 | - | Semantic document structure |
| **Styling** | CSS3 | - | Custom properties, flexbox, grid, animations |
| **Scripting** | JavaScript | ES6+ | Interactive lesson viewer, progress tracking |
| **Content** | Markdown | - | Lesson file format |

---

## External Libraries (CDN-Hosted)

### Syntax Highlighting
- **Library:** [Highlight.js](https://highlightjs.org/)
- **Version:** 11.9.0
- **Theme:** Nord
- **Languages:** Python
- **Purpose:** Code block syntax highlighting for lesson content
- **CDN:** `https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js`

### Markdown Parsing
- **Library:** [Marked.js](https://marked.js.org/)
- **Version:** 11.1.1
- **Purpose:** Convert Markdown lesson files to HTML at runtime
- **CDN:** `https://cdnjs.cloudflare.com/ajax/libs/marked/11.1.1/marked.min.js`

### Typography (Google Fonts)
- **JetBrains Mono** - Code blocks (400, 600 weights)
- **Crimson Pro** - Body text, serif (400, 600, 700 weights)
- **Work Sans** - UI elements, headers (500, 700 weights)
- **Space Grotesk** - Display headers (300, 500, 700 weights)
- **Outfit** - Secondary UI text (300, 400, 500, 600 weights)
- **CDN:** `https://fonts.googleapis.com/css2?family=...`

---

## Architecture

### Application Type
**Static Single-Page Application (SPA)**

- Single HTML entry point (`index.html`)
- All CSS embedded in `<style>` block
- All JavaScript in external file (`app.js`)
- Lesson content loaded dynamically via `fetch()`

### Data Flow

```
┌─────────────────┐
│  index.html     │
│  (Entry Point)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  app.js         │
│  (Viewer Logic) │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌──────────┐
│ CDN   │ │ Lessons/ │
│ Libs  │ │ *.md     │
└───────┘ └──────────┘
```

### State Management
- **Client-side only:** No server communication
- **LocalStorage:** Progress persistence per lesson
  - `lessonProgress_[filename]` - Stage completion data
  - `revealedBlocks_[filename]` - Timed reveal state
  - `scrollPosition_[filename]` - Scroll position and nearest stage ID for restoration
  - `skipWelcomeModal_[filename]` - User preference to skip welcome-back modal
  - `seenLessons` - Array of lesson filenames the user has seen (for announcement system)
- **In-memory:** Current lesson state, reveal queue, observer patterns

---

## File Structure

```
SnakeCharming/
├── index.html              # Main application entry point
├── app.js                  # Lesson viewer logic
├── favicon.png             # Site icon
├── lessons/
│   ├── projectile-simulator.md
│   ├── f1-braking.md
│   └── model-training.md
├── doc/
│   ├── prompt-spec.md      # Lesson authoring specification
│   └── publishing-ideas.md # Portfolio/deployment guidance
└── conductor/              # Project management files
    ├── product.md
    ├── product-guidelines.md
    ├── tech-stack.md
    └── ...
```

---

## Browser Requirements

### Minimum Supported Browsers
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

### Required Features
- ES6+ JavaScript (arrow functions, classes, template literals, async/await)
- CSS Custom Properties (variables)
- CSS Flexbox and Grid
- Intersection Observer API
- LocalStorage API
- Fetch API

### No Support For
- Internet Explorer (any version)
- Legacy browsers without ES6 support

---

## Deployment Requirements

### Server Requirements
- **None** - Pure static files
- Any HTTP server capable of serving `.html`, `.js`, `.css`, `.md` files
- No server-side processing (PHP, Node.js, Python, etc.)

### Compatible Hosting Platforms
- GitHub Pages
- Netlify
- Vercel
- Cloudflare Pages
- Traditional web hosting (Apache, Nginx)
- Local file system (`file://` protocol works offline)

### Performance Characteristics
- **Initial load:** ~100KB (HTML + CSS + JS, excluding CDN resources)
- **Lesson files:** ~50-100KB each (Markdown)
- **CDN resources:** ~150KB (Highlight.js + Marked.js + Fonts)
- **No runtime build process** - Files served as-is

---

## Development Workflow

### Adding New Lessons
1. Create Markdown file in `lessons/` directory
2. Follow specification in `doc/prompt-spec.md`
3. Register lesson in `app.js` → `LESSONS` array
4. Test in browser

### Modifying Application Logic
1. Edit `app.js`
2. Test in browser (no build step required)
3. Commit changes

### Styling Changes
1. Edit `<style>` block in `index.html`
2. Test in browser
3. Commit changes

---

## Technical Constraints

### Current Limitations
1. **No user accounts** - Progress stored locally per browser/device
2. **No analytics** - No server-side tracking of usage
3. **No collaborative features** - Single-user, local experience
4. **CDN dependency** - Requires internet for first load (libraries, fonts)
5. **No mobile app** - Web-only, responsive but not native

### Future Enhancement Possibilities
- **Progress sync:** Backend API for cross-device progress
- **Offline mode:** Service worker for full offline capability
- **Export functionality:** Download progress reports as PDF
- **Teacher dashboard:** Track student progress across classes
- **Interactive coding:** WebAssembly-based Python execution in browser

---

## Code Style Standards

### JavaScript
- **ES6+ features:** Classes, arrow functions, template literals, async/await
- **Naming conventions:** 
  - Classes: `PascalCase` (e.g., `LessonViewer`)
  - Methods/properties: `camelCase` (e.g., `loadLesson`, `currentStage`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `LESSONS`, `REVEAL_DELAY`)
- **File organization:** Single file (`app.js`), class-based architecture
- **Comments:** JSDoc-style for methods, inline for complex logic

### CSS
- **Custom properties:** Centralized in `:root` selector
- **Naming:** BEM-inspired, descriptive class names
- **Organization:** Grouped by component/feature
- **Responsive:** Mobile-first media queries

### Markdown (Lessons)
- **Stage headings:** `## **Stage N: Title**` (required format for parser)
- **Code blocks:** Triple backticks with language specifier
- **Checkpoints:** `### **Checkpoint #N**` format
- **Partial reveals:** `<!-- PARTIAL_REVEAL -->` marker for fill-in-blank exercises

---

## Security Considerations

### Current Security Posture
- **No user input** - No forms, no data submission
- **No authentication** - No sensitive data handled
- **CDN resources** - Trusted providers (Cloudflare, Google)
- **LocalStorage** - Client-side only, no data transmission

### Potential Risks
- **XSS via lesson files** - If lesson Markdown is user-generated, sanitize before rendering
- **CDN compromise** - Mitigated by using reputable providers (Cloudflare)
- **LocalStorage manipulation** - Students could modify their own progress (not critical)

### Recommendations
- Validate/sanitize any user-generated lesson content
- Consider Subresource Integrity (SRI) hashes for CDN resources
- Document that progress data is not secure/verified

---

## Performance Optimization

### Current Optimizations
- **Minimal dependencies** - Only essential libraries
- **CDN hosting** - Leverages browser caching, global distribution
- **Lazy loading** - Lessons loaded on-demand via `fetch()`
- **Efficient rendering** - Intersection Observer for reveal triggers

### Potential Optimizations
- **Inline critical CSS** - Reduce render-blocking resources
- **Preload key fonts** - `<link rel="preload">` for above-fold fonts
- **Minify assets** - Compress JS/CSS for production
- **Service worker** - Cache lessons for offline use
- **Lazy-load Highlight.js languages** - Only load Python when needed
