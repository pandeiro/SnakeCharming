# SnakeCharming - Product Guidelines

## Brand Voice & Tone

### Primary Voice: Encouraging & Conversational

SnakeCharming communicates with students in a **friendly, supportive manner** that builds confidence and reduces the intimidation factor of learning to code.

#### Voice Characteristics
- **Warm and approachable** - Use "you" and "we" to create connection
- **Celebratory** - Acknowledge achievements ("Great work!", "You did it!")
- **Patient** - Never condescending; treat mistakes as learning opportunities
- **Clear** - Avoid jargon; explain technical terms when introduced

#### Language Patterns to Use
- ✅ "Let's build this together..."
- ✅ "You've got this! Try..."
- ✅ "Great work! Now let's..."
- ✅ "Don't worry if this seems tricky..."
- ✅ "Here's what's happening..."

#### Language Patterns to Avoid
- ❌ "Simply do X..." (dismissive of difficulty)
- ❌ "Obviously..." (assumes knowledge)
- ❌ "Just..." (minimizes complexity)
- ❌ Passive voice ("The variable should be set...")
- ❌ Overly formal academic language

---

## Visual Design Principles

### Core Aesthetic: Modern Dark Tech

SnakeCharming's interface combines **professional developer tools aesthetics** with **approachable educational design**.

#### 1. Dark Theme with High Contrast
- **Background:** Deep navy/charcoal (#0a0e16, #141922)
- **Text:** Off-white for readability (#e8eef5)
- **Accent colors:** Neon green/teal (#4af2a1, #2dd4bf)
- **Code blocks:** Darker background with syntax highlighting (Nord theme)
- **Purpose:** Reduce eye strain during extended coding sessions; make code the visual focal point

#### 2. Clean Minimalism
- **Remove visual clutter** - Every element serves a purpose
- **Generous whitespace** - Give content room to breathe
- **Single focal point** - One primary action per screen/section
- **Hidden complexity** - Advanced options collapsed by default
- **Purpose:** Students focus on learning, not navigating UI

#### 3. Modern Tech Aesthetic
- **Subtle gradients** - Background transitions, button states
- **Neon accent colors** - Success states, highlights, CTAs
- **Monospace fonts for code** - JetBrains Mono or similar
- **Smooth animations** - Transitions feel polished, not flashy
- **Purpose:** Feels like professional developer tools, not "kiddie" software

#### 4. Accessible Design
- **Minimum 16px body text** - Readable without zooming
- **High contrast ratios** - WCAG AA compliance minimum
- **Color-blind safe palette** - Never use color alone to convey meaning
- **Clear focus states** - Keyboard navigation support
- **Descriptive link text** - No "click here"
- **Purpose:** Inclusive for all learners regardless of ability

---

## Student Support Philosophy

### Gentle Guidance Approach

When students struggle, SnakeCharming provides **structured hints** that guide without giving away answers.

#### Hint Hierarchy
1. **First hint:** Point to the relevant concept or section
   - *"Check the function definition in Step 2..."*
2. **Second hint:** Identify the type of error
   - *"This looks like a variable naming issue..."*
3. **Third hint:** Show a similar example from earlier
   - *"Remember how we calculated velocity in Stage 1?"*
4. **Final resort:** Offer partial solution with blanks
   - *"The formula uses: `result = ___ * ___ / ___`"*

#### Error Messaging Guidelines
- **Specific, not generic** - "Variable `v0` is not defined" not "Error occurred"
- **Actionable** - Tell students what to check or try
- **Encouraging** - "Almost there! Check..." not "Wrong"
- **Educational** - Explain *why* something failed when possible

#### Celebration & Feedback
- **Immediate positive feedback** for completed stages
- **Visual progress indicators** - Progress bars, checkmarks
- **Milestone celebrations** - Animations for major achievements
- **Progress persistence** - Students can return where they left off

---

## Content Creation Guidelines

### For Lesson Authors

All SnakeCharming lessons must adhere to these content standards.

#### 1. Real-World Context
**Every coding concept connects to practical applications.**

- **Start with "Why"** - Explain relevance before implementation
- **Use engineering scenarios** - Physics simulations, optimization problems
- **Show tangible outcomes** - "You'll build a tool that..."
- **Connect to careers** - "This is how mechanical engineers..."

**Example:**
> *"Mechanical engineers at SpaceX use trajectory simulations like this to calculate optimal rocket launch angles. You're using the same physics!"*

#### 2. Incremental Challenges
**Break complex tasks into small, achievable steps.**

- **One concept per step** - Don't combine multiple new ideas
- **Scaffold progressively** - Each step builds on previous
- **Provide starter code** - Reduce cognitive load on syntax
- **Clear success criteria** - Students know when they're done

**Step Structure:**
```markdown
**Step N: [Action-Oriented Title]**

[1-2 sentence explanation of what this does]

```python
# Starter code with clear task
```

**What this does:**
- [Bullet explanation]
```

#### 3. Reflection Prompts
**Include questions that encourage metacognition.**

- **"What if..." questions** - Encourage experimentation
- **Prediction prompts** - "What do you think will happen if...?"
- **Connection questions** - "How does this relate to...?"
- **Extension challenges** - Optional deeper exploration

**Example Prompts:**
- *"What happens to the trajectory if you double the initial velocity? Try it and observe."*
- *"Why do you think 45° gives the maximum range? Think about the physics..."*
- *"How would this change if we added air resistance?"*

#### 4. Code Quality Standards
All example code must demonstrate best practices:

- **Meaningful variable names** - `velocity` not `v`, `angle_degrees` not `a`
- **Inline comments** - Explain *why*, not *what*
- **Consistent formatting** - Follow Python style conventions
- **Error handling** - Show defensive programming where appropriate
- **Docstrings** - All functions include documentation

---

## Scaffolding Strategy

### Balanced Approach: Progressive Release

SnakeCharming uses a **gradual release of responsibility** model across lessons.

#### Early Lessons (High Scaffolding)
- **Fill-in-the-blank code** - Key concepts highlighted
- **Partial reveals** - Show structure, hide implementation
- **Guided parameter selection** - "Try a value between 10-30"
- **Multiple choice checkpoints** - Verify understanding

#### Mid Lessons (Moderate Scaffolding)
- **Function stubs** - Provide structure, students implement
- **Hint system** - Available on demand
- **Worked examples** - Similar problems with solutions
- **Reduced hand-holding** - More independent problem-solving

#### Late Lessons (Low Scaffolding)
- **Problem statements** - Clear requirements, open implementation
- **Minimal starter code** - Just imports and structure
- **Self-directed debugging** - Students use learned strategies
- **Extension-focused** - Advanced challenges for mastery

#### Scaffolding Mechanisms

| Mechanism | Purpose | When to Use |
|-----------|---------|-------------|
| Timed reveals | Encourage thinking before copying | Key algorithm implementations |
| Fill-in-blanks | Focus attention on critical concepts | New syntax or patterns |
| Checkpoints | Verify progress at milestones | After complex multi-step tasks |
| Worked examples | Model problem-solving approach | Before similar independent tasks |
| Extension challenges | Provide depth for advanced students | After core requirements met |

---

## Accessibility & Inclusion

### Design for All Learners

SnakeCharming commits to inclusive design that serves diverse learners.

#### Cognitive Accessibility
- **Chunked content** - Short sections with clear headings
- **Multiple representations** - Code + visual + textual explanations
- **Pacing control** - Students progress at their own speed
- **Reduced cognitive load** - One concept at a time

#### Cultural Inclusivity
- **Diverse examples** - Applications from various fields and contexts
- **Universal scenarios** - Physics and math are language-independent
- **Avoid assumptions** - Don't assume prior knowledge or background
- **Gender-neutral language** - Use "they/them" in examples

#### Technical Accessibility
- **Keyboard navigation** - All features accessible without mouse
- **Screen reader compatible** - Semantic HTML, alt text for visuals
- **Responsive design** - Works on various screen sizes
- **Low bandwidth** - Minimal external dependencies

---

## Quality Standards

### Lesson Review Checklist

Before publishing, every lesson must pass:

#### Content Quality
- [ ] Clear learning objectives stated upfront
- [ ] Progressive difficulty (each stage builds on previous)
- [ ] Real-world context provided
- [ ] Checkpoints with verification code
- [ ] Extension challenges included

#### Code Quality
- [ ] All code tested and runs without errors
- [ ] Meaningful variable names
- [ ] Comments explain reasoning
- [ ] Functions include docstrings
- [ ] Expected outputs shown

#### Pedagogical Quality
- [ ] Explains "why" not just "how"
- [ ] Multiple ways to verify correctness
- [ ] Encourages experimentation
- [ ] Appropriate scaffolding for target level
- [ ] Reflection prompts included

#### Accessibility
- [ ] High contrast, readable design
- [ ] Clear, jargon-free language
- [ ] Multiple means of engagement
- [ ] Progress can be saved and resumed
