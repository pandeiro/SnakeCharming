# SnakeCharming - Product Definition

## Target Users

### Primary Audience
1. **High School Students** - Learning programming for the first time, typically with no prior coding experience
2. **University Students** - Enrolled in introductory computer science or engineering courses that incorporate Python programming

### User Characteristics
- Beginners with minimal to no programming background
- Familiar with basic high school mathematics (algebra, trigonometry)
- May have exposure to physics concepts (kinematics, mechanics)
- Need engaging, low-barrier entry points to computational thinking
- Benefit from visual feedback and progressive skill building

---

## Learning Outcomes

By completing SnakeCharming lessons, students will be able to:

### Core Competencies
1. **Python Fundamentals** - Master fundamental Python syntax and programming concepts including variables, functions, loops, and data structures
2. **Applied Computational Thinking** - Apply computational thinking to solve physics and engineering problems through numerical simulation
3. **Independent Coding** - Build confidence in reading and writing code independently without constant guidance
4. **Debugging & Problem-Solving** - Develop systematic debugging skills and algorithmic problem-solving approaches

### Secondary Outcomes
- Translate mathematical equations into working code
- Create visualizations to analyze and present results
- Understand the engineering design process through iterative testing
- Develop persistence through scaffolded challenges

---

## Pedagogical Approach

### Guided Discovery Learning

SnakeCharming employs a **guided discovery** methodology where students:

- **Explore** concepts through interactive code challenges
- **Experiment** with modified parameters to observe outcomes
- **Construct** understanding through hands-on implementation
- **Reflect** on results via checkpoint verification

### Key Principles
1. **Scaffolded Complexity** - Each lesson builds progressively from simple concepts to more advanced applications
2. **Active Learning** - Students write and modify code rather than passively reading tutorials
3. **Immediate Feedback** - Timed reveals and checkpoints provide structured guidance without giving answers too quickly
4. **Contextual Relevance** - Problems are grounded in real-world physics and engineering scenarios

---

## Interactive Features

### Core Interactions

1. **Timed Code Reveals**
   - Encourages students to think through problems before viewing solutions
   - Timer activates only when code block is in viewport
   - Promotes active engagement over passive copying

2. **Checkpoint Verification**
   - Provides expected outputs at key milestones
   - Students can verify their implementation is correct
   - Builds confidence through incremental success

### Supporting Features
- Stage-based lesson progression with unlock mechanics
- Visual progress tracking with completion percentages
- Celebration animations for milestone achievements
- Collapsible stage sections for focused learning
- Syntax-highlighted code blocks with copy functionality
- **Progress restoration** - Automatically saves and restores scroll position when returning to lessons
- **Welcome-back modal** - Shows returning users their progress summary with option to continue or start over
- **Toast notifications** - Brief, non-intrusive messages for progress restoration when modal is skipped

---

## Deployment Model

### Self-Hosted Static Site

SnakeCharming is designed as a **zero-dependency static web application**:

#### Technical Requirements
- Single HTML file with embedded CSS/JavaScript
- Markdown lesson files served from the same directory
- No server-side processing required
- No database or user authentication needed

#### Deployment Flexibility
- Host on any static file server (GitHub Pages, Netlify, personal server)
- Works completely offline once loaded
- No external API dependencies (beyond CDN for libraries)
- Portable - can be distributed as a zip file

#### Accessibility Benefits
- No login barriers for students
- Works on any modern web browser
- Low bandwidth requirements
- Compatible with school firewall restrictions

---

## Content Structure

### Lesson Format
Each lesson follows a standardized structure:
- **Project Overview** - Clear objectives and relevance
- **Stage-Based Progression** - Sequential learning modules
- **Step-by-Step Instructions** - Guided implementation
- **Checkpoints** - Verification points with expected outputs
- **Extension Challenges** - Optional advanced work

### Subject Domains
- Physics simulations (projectile motion, kinematics)
- Engineering optimization problems
- Data visualization exercises
- Numerical methods and computational modeling
