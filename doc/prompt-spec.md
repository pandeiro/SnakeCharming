# Lesson Plan Specification for Interactive Viewer

## Overview
This document provides the complete specification for creating lesson plans that work with the interactive lesson viewer. Follow this guide to ensure your lesson plan renders correctly with all interactive features.

---

## File Format Requirements

### Basic Structure
- **Format:** Markdown (.md file)
- **Encoding:** UTF-8
- **Naming Convention:** `lesson-[topic-name].md` (e.g., `lesson-data-visualization.md`, `lesson-machine-learning-basics.md`)

---

## Document Structure

### 1. Title Section (Required)
```markdown
# **Project Assignment: [Your Title]**

## **Project Overview**

**Objective:** [Clear, one-sentence description of what students will build]

**Why This Project?**
1. [Reason 1 - Connection to prior knowledge]
2. [Reason 2 - Progressive complexity]
3. [Reason 3 - Tangible outcomes]
4. [Reason 4 - Real-world relevance]

**Prerequisites:** [List required background knowledge]

**Tools:** [Software/platforms needed - e.g., Google Colab, Jupyter, VS Code]

**Estimated Time:** [Total time commitment]

---

## **Pedagogical Objectives**

By completing this project, you will learn to:
1. [Learning objective 1 - use action verbs: build, analyze, implement, visualize]
2. [Learning objective 2]
3. [Learning objective 3]
4. [Learning objective 4-6]

---
```

### 2. Stage Structure (Required)

Each lesson **must** be divided into stages using this exact heading format:

```markdown
## **Stage [N]: [Stage Title]**

### **Learning Focus: [Focus Area]**
*Core Skills: [Comma-separated list of specific skills]*

**Time Estimate:** [e.g., 2-3 hours]

**What You'll Build:** [Brief description of the deliverable for this stage]

**Assignment:** [Clear description of what the student needs to accomplish]

---
```

**Critical Requirements:**
- Stage headings MUST match pattern: `## **Stage [N]: [Title]**` (with "Stage" and number)
- Include all subsections: Learning Focus, Core Skills, Time Estimate
- The viewer will automatically parse these into collapsible, sequential sections

---

## Code Block Guidelines

### Standard Code Blocks

Use triple backticks with language specification:

````markdown
```python
import numpy as np
import matplotlib.pyplot as plt

# Your code here
x = np.linspace(0, 10, 100)
y = np.sin(x)
```
````

### Code Blocks for Timed Reveal

The viewer automatically applies timed reveals to code blocks matching these patterns:
- Import statements: `import`, `from ... import`
- Function definitions: `def function_name(`
- Class definitions: `class ClassName:`
- Loop structures: `for ... in range(`, `while ...:`

**To trigger timed reveal for specific blocks:**
- Include function definitions (viewer looks for `def` keyword)
- Include import statements at strategic learning points
- Include optimization/iteration loops

**To prevent timed reveal:**
- Use simple assignment statements
- Use small code snippets (< 5 lines)
- Use example outputs or print statements

### Partial Code Reveals (Fill-in-the-Blank)

**NEW FEATURE:** For pedagogical scenarios where you want students to complete partially-shown code, use the special marker `<!-- PARTIAL_REVEAL -->` within your code block.

**How it works:**
The content is split into **two separate code blocks**:
1. **First block (visible):** Shows immediately - this is what students see and should work with
2. **Second block (hidden):** Starts hidden behind an overlay - this is the complete solution

**Syntax:**

````markdown
```python
# This part shows immediately - the student sees this
g = 9.81  # gravity (m/s²)
v0 = ???   # What should this value be?
theta = ???  # What should this value be?
<!-- PARTIAL_REVEAL -->
# This entire block is hidden initially - the complete solution
g = 9.81  # gravity (m/s²)
v0 = 20    # initial velocity (m/s)
theta = 45  # launch angle (degrees)
```
````

**Important:** The entire solution code should be repeated below the `<!-- PARTIAL_REVEAL -->` marker. Don't try to show only the "missing parts" - show the complete, working code block.

**Visual presentation:**
1. Yellow "Try It Yourself" banner appears
2. First code block displays normally (the partial/scaffold code)
3. Second code block is hidden behind overlay with message: "Complete the code above yourself!"
4. Timer only starts when:
   - Student scrolls the hidden block into view AND
   - All previous timed reveals have been completed
5. Student can click "Skip Timer (Show Solution)" anytime
6. After timer expires or skip, overlay fades away revealing the complete solution

**Example - Good partial reveal:**

````markdown
```python
# Calculate velocity components - fill in the trig functions
theta_rad = np.radians(theta)
vx = v0 * ???(theta_rad)  # Hint: horizontal uses cosine
vy = v0 * ???(theta_rad)  # Hint: vertical uses sine
<!-- PARTIAL_REVEAL -->
# Complete solution:
theta_rad = np.radians(theta)
vx = v0 * np.cos(theta_rad)
vy = v0 * np.sin(theta_rad)
```
````

**Example - Setting up physics constants:**

````markdown
```python
# Set up the initial conditions for our projectile
g = 9.81  # gravity (m/s²) 
v0 = ???   # Try a value between 10-30 m/s
theta = ???  # Try an angle between 30-60 degrees
<!-- PARTIAL_REVEAL -->
# Recommended values:
g = 9.81
v0 = 20    # 20 m/s ≈ 45 mph
theta = 45  # Classic projectile angle
```
````

**Best practices:**
1. **Provide scaffolding:** Show structure, variable names, comments as hints
2. **Use `???` or `____`:** Make it obvious what needs to be filled in
3. **Repeat complete solution:** Always show the full working code below the marker
4. **Add helpful comments:** Guide students with hints in the visible part
5. **Keep it focused:** One clear concept per reveal (don't hide multiple unrelated things)
6. **Test the scaffold:** Make sure the visible part makes sense on its own

**What NOT to do:**
❌ Don't show only the missing snippets below the marker:
````markdown
```python
vx = v0 * ???(theta_rad)
<!-- PARTIAL_REVEAL -->
np.cos  # This alone is confusing!
```
````

✅ Do show the complete solution:
````markdown
```python
vx = v0 * ???(theta_rad)
<!-- PARTIAL_REVEAL -->
vx = v0 * np.cos(theta_rad)  # Complete line!
```
````

**Timer behavior:**
- Timer starts only when the hidden block scrolls into view
- Timer waits for all previous reveals to complete first (sequential learning)
- Students control their pace with the "Skip Timer" button
- Default timer: 120 seconds (2 minutes)

---

## Special Markdown Elements

### Checkpoints

Format checkpoints like this:

```markdown
### **Checkpoint #[N]**

**What you should see/observe:** [Expected outcome]

**Verification:**
```python
# Code to verify results
print(f"Expected value: {expected}")
```

**Expected output:**
- [Item 1]
- [Item 2]

---
```

**The viewer will automatically style these with:**
- Green accent border
- Checkmark icon
- Highlighted background

### Engineering Insights

Use blockquotes for special callouts:

```markdown
**Engineering Insight:** This "brute force" optimization approach is simple but powerful—you're literally testing every possibility. In engineering, we often use more sophisticated methods, but this demonstrates the concept.
```

Or use a special subsection:

```markdown
### **Engineering Insight**

[Your insight here connecting the concept to real-world engineering]

---
```

### Common Issues & Debugging

Include troubleshooting sections:

```markdown
### **Common Issues & Debugging**

**Problem:** "NameError: name 'np' is not defined"
- **Solution:** Check that you ran the import cell first

**Problem:** Plot looks like a flat line
- **Solution:** Verify you converted degrees to radians (`np.radians()`)

**Debugging tip:** Use `print()` statements to check your variables:
```python
print(f"Variable value: {my_variable}")
```

---
```

### Extension Challenges

Add optional challenges at the end of stages:

```markdown
### **Extension Challenge (Optional)**

Try modifying the code to:
1. [Challenge 1]
2. [Challenge 2]
3. [Challenge 3]

**Reflection Questions:**
- [Question 1]
- [Question 2]

---
```

---

## Step-by-Step Instructions

Break down complex tasks into numbered steps:

```markdown
### **Step-by-Step Instructions:**

**Step 1: [Action Title]**

[Explanation of what this step does]

```python
# Code for step 1
```

**What this does:**
- [Explanation point 1]
- [Explanation point 2]

**Key Concept:** [Important concept to understand]

---

**Step 2: [Next Action]**

[Continue pattern...]

---
```

---

## Final Sections

### Project Deliverables

```markdown
## **Project Deliverables**

Submit your [notebook/project] with:

### **1. Code (Well-Commented)**
- [Requirement 1]
- [Requirement 2]

### **2. Visualizations**
Include at least:
- [Visualization 1]
- [Visualization 2]

### **3. Written Responses**

Answer these questions in markdown cells:

**Analysis Questions:**
1. [Question 1]
2. [Question 2]

**Application Questions:**
3. [Question 3]

**Reflection:**
4. [Question 4]

---
```

### Success Criteria

Use a tiered rubric:

```markdown
## **Success Criteria**

### **Complete (Meets Expectations)**
- ✅ [Criterion 1]
- ✅ [Criterion 2]

### **Proficient (Exceeds Expectations)**
- ✅ All of above, plus:
- ✅ [Additional criterion 1]
- ✅ [Additional criterion 2]

### **Excellent (Outstanding)**
- ✅ All of above, plus:
- ✅ [Advanced criterion 1]

### **Exceptional (Goes Above and Beyond)**
- ✅ All of above, plus:
- ✅ [Creative extension criterion]

---
```

---

## Content Writing Guidelines

### Voice and Tone
- **Direct and encouraging:** "You'll build...", "Let's create..."
- **Avoid passive voice:** Use "Calculate the trajectory" not "The trajectory should be calculated"
- **Balance support with challenge:** Provide help but encourage independent problem-solving
- **Be enthusiastic but not condescending:** Treat students as capable learners

### Pedagogical Best Practices

1. **Scaffold complexity:** Each stage should build on previous knowledge
2. **Explain the "why":** Don't just give steps—explain the reasoning
3. **Connect to applications:** Show real-world engineering relevance
4. **Provide debugging support:** Anticipate common errors and address them
5. **Include verification:** Students need to know if they're on the right track
6. **Offer extensions:** Challenge advanced students without overwhelming beginners

### Code Comments

All code should include:
- **Explanatory comments:** What each section does
- **Inline comments:** Clarify complex lines
- **Docstrings:** For all functions (Google style recommended)

Example:
```python
def calculate_derivative(f, x, h=1e-5):
    """
    Calculate numerical derivative using central difference method.
    
    Parameters:
    -----------
    f : callable
        Function to differentiate
    x : float
        Point at which to evaluate derivative
    h : float, optional
        Step size for numerical differentiation (default: 1e-5)
    
    Returns:
    --------
    derivative : float
        Approximate derivative at point x
    """
    return (f(x + h) - f(x - h)) / (2 * h)
```

---

## Domain-Specific Considerations

### For Different Programming Topics

**Data Science/Analysis:**
- Include data loading and cleaning stages
- Show visualization early for motivation
- Emphasize exploratory data analysis
- Build to statistical insights or predictions

**Machine Learning:**
- Start with data understanding
- Progress through: preprocessing → training → evaluation → tuning
- Include model interpretation and validation
- Show real predictions/classifications

**Numerical Methods:**
- Connect to mathematical concepts students know
- Show analytical solution first, then numerical approximation
- Compare accuracy and performance
- Visualize convergence or error

**Web Development:**
- Show visual results immediately
- Build up: HTML structure → CSS styling → JS interactivity
- Include responsive design considerations
- Deploy or share final product

**Game Development:**
- Start with minimal playable prototype
- Add features incrementally
- Include game design principles
- Make it fun and engaging

**Data Visualization:**
- Progress from basic to advanced charts
- Emphasize storytelling with data
- Include design principles (color, layout, clarity)
- Show bad examples and how to improve them

### For Different Engineering Domains

**Mechanical Engineering:**
- Connect to statics, dynamics, thermodynamics
- Include physical units and dimensional analysis
- Show optimization of physical parameters
- Relate to real systems (vehicles, structures, machines)

**Electrical Engineering:**
- Include circuit analysis, signal processing
- Visualize waveforms and frequency spectra
- Show time-domain and frequency-domain views
- Connect to communication or control systems

**Civil Engineering:**
- Focus on structural analysis, load calculations
- Include safety factors and design codes
- Show stress/strain visualizations
- Connect to infrastructure projects

**Biomedical Engineering:**
- Use physiological data (heart rate, imaging)
- Include medical device examples
- Show biological system modeling
- Emphasize safety and regulatory considerations

**Environmental Engineering:**
- Use environmental data (pollution, climate)
- Model natural systems
- Include sustainability metrics
- Show policy or design implications

---

## Technical Requirements

### Python Libraries

Always specify and explain imports:

```python
# Core numerical computing
import numpy as np

# Data visualization
import matplotlib.pyplot as plt

# Data manipulation (if needed)
import pandas as pd

# Statistical functions (if needed)
from scipy import stats

# Machine learning (if needed)
from sklearn.model_selection import train_test_split
```

### Plot Styling

Encourage professional-looking plots:

```python
plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2, color='#4af2a1', label='Data')
plt.xlabel('X-axis Label', fontsize=12)
plt.ylabel('Y-axis Label', fontsize=12)
plt.title('Descriptive Title', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Error Handling

Teach defensive programming:

```python
try:
    # Risky operation
    result = divide(a, b)
except ZeroDivisionError:
    print("Cannot divide by zero!")
    result = None
```

---

## Example Lesson Topics by Domain

### Numerical Computing
- **Newton's Method Root Finder:** Implement iterative root-finding algorithm
- **Monte Carlo Integration:** Estimate π or definite integrals using random sampling
- **Differential Equation Solver:** Solve ODEs using Euler/RK4 methods
- **Fourier Analysis:** Decompose signals into frequency components
- **Matrix Operations:** Implement Gaussian elimination or eigenvalue solvers

### Data Science
- **Weather Pattern Analyzer:** Download and analyze historical weather data
- **Stock Market Dashboard:** Visualize financial data with trends and predictions
- **Text Sentiment Analyzer:** Process text data to determine sentiment scores
- **Movie Recommendation System:** Build collaborative filtering recommender
- **A/B Test Analyzer:** Determine statistical significance of experimental results

### Machine Learning
- **Handwritten Digit Classifier:** Train neural network on MNIST dataset
- **Housing Price Predictor:** Linear regression with feature engineering
- **Customer Churn Predictor:** Classification with decision trees/random forests
- **Image Style Transfer:** Apply artistic styles to photos using neural networks
- **Anomaly Detector:** Find outliers in sensor or transaction data

### Engineering Simulation
- **Heat Transfer Simulator:** Model 1D/2D heat conduction with finite differences
- **Traffic Flow Optimizer:** Simulate and optimize intersection timing
- **Beam Deflection Calculator:** Analyze structural loads and deflections
- **Circuit Simulator:** Solve RLC circuits in time and frequency domain
- **Fluid Flow Visualizer:** Simulate flow patterns around obstacles

### Data Visualization
- **Interactive Climate Dashboard:** Create multi-panel climate data explorer
- **Geographic Heat Map:** Visualize spatial data on maps
- **Time Series Animator:** Show evolution of data over time
- **Network Graph Visualizer:** Display relationships in graph structures
- **3D Surface Plotter:** Visualize mathematical functions or topography

---

## Checklist for New Lesson Plans

Before submitting your lesson plan, verify:

### Structure
- [ ] Title section with overview, prerequisites, time estimate
- [ ] Pedagogical objectives clearly stated
- [ ] 3-6 stages with proper heading format: `## **Stage N: Title**`
- [ ] Each stage has Learning Focus, Core Skills, Time Estimate
- [ ] Final deliverables and success criteria included

### Content Quality
- [ ] Progressive difficulty - each stage builds on previous
- [ ] Clear step-by-step instructions with explanations
- [ ] Code blocks include comments and docstrings
- [ ] Checkpoints provide verification at key milestones
- [ ] Debugging tips anticipate common student errors
- [ ] Extension challenges offer optional advanced work

### Pedagogical Elements
- [ ] Explains "why" not just "how" for key concepts
- [ ] Connects to real-world engineering applications
- [ ] Includes reflection questions
- [ ] Provides multiple ways to verify correctness
- [ ] Encourages experimentation and exploration
- [ ] Balances scaffolding with student independence

### Technical Requirements
- [ ] All code is tested and runs without errors
- [ ] Library imports are explained
- [ ] Plots are properly labeled and styled
- [ ] Variable names are clear and meaningful
- [ ] Functions include proper documentation
- [ ] Expected outputs are shown for verification

### Interactive Features
- [ ] Key implementation code will trigger timed reveals
- [ ] Checkpoints are properly formatted for styling
- [ ] Stages are sequential and unlock progressively
- [ ] Clear completion criteria for each stage
- [ ] Final celebration is earned!

---

## Registering Your Lesson

To add your lesson to the viewer, edit the `LESSONS` array in `lesson-viewer.html`:

```javascript
const LESSONS = [
    {
        name: 'Projectile Simulator',
        file: 'lesson-projectile-simulator.md'
    },
    {
        name: 'Your New Lesson Title',
        file: 'lesson-your-topic.md'
    }
    // Add more lessons here
];
```

Place your `.md` file in the same directory as `lesson-viewer.html`.

---

## Example Template

Use this template to start your lesson plan:

```markdown
# **Project Assignment: [Your Topic]**

## **Project Overview**

**Objective:** [One clear sentence]

**Why This Project?**
1. [Foundation connection]
2. [Progressive complexity]
3. [Tangible results]
4. [Real-world relevance]

**Prerequisites:** [Background needed]

**Tools:** [Platform/software]

**Estimated Time:** [X-Y hours total]

---

## **Pedagogical Objectives**

By completing this project, you will learn to:
1. [Objective 1]
2. [Objective 2]
3. [Objective 3]
4. [Objective 4]

---

## **Stage 1: [First Stage Title]**

### **Learning Focus: [Focus Area]**
*Core Skills: [Skills list]*

**Time Estimate:** [X hours]

**What You'll Build:** [Description]

**Assignment:** [Clear task]

---

### **Step-by-Step Instructions:**

**Step 1: [Action]**

[Explanation]

```python
# Code here
```

**What this does:**
- [Explanation]

---

**Step 2: [Next Action]**

[Continue...]

---

### **Checkpoint #1**

**What you should see:** [Expected result]

**Verification:**
```python
# Verification code
```

---

[Continue with more stages...]

---

## **Project Deliverables**

Submit your notebook with:

### **1. Code**
- [Requirements]

### **2. Visualizations**
- [Requirements]

### **3. Written Responses**
1. [Question 1]
2. [Question 2]

---

## **Success Criteria**

### **Complete**
- ✅ [Basics]

### **Proficient**
- ✅ All above, plus [more]

### **Excellent**
- ✅ All above, plus [advanced]

---
```

---

## Support and Questions

### Progress Tracking

The viewer automatically tracks student progress using browser localStorage:

**Features:**
- **Automatic saving:** Progress saves when students complete stages
- **Per-lesson tracking:** Each lesson has independent progress
- **Persistent across sessions:** Students can close and return later
- **Visual feedback:** Progress bar shows completion percentage
- **Restore notification:** Banner shows when returning with saved progress
- **Reset option:** Students can clear progress and start over

**Progress notifications appear when:**
1. **First load:** "Lesson Loaded: Ready to begin!"
2. **Returning student:** "Progress Restored: You've completed X stages"
3. **Switching lessons:** "New Lesson: Starting fresh"
4. **After reset:** "Progress Reset: Starting fresh"

**Data stored per lesson:**
- Current active stage
- List of completed stages
- Last update timestamp

**Student controls:**
- **Dismiss:** Hide notification banner
- **Reset Progress:** Clear all progress (with confirmation modal)

**Privacy note:** All data is stored locally in the browser. No server communication occurs.

If you have questions about creating lesson plans:
1. Review the example: `lesson-projectile-simulator.md`
2. Check that your markdown follows the structure patterns
3. Test your lesson in the viewer to ensure proper rendering
4. Iterate based on how students engage with the material

Happy lesson planning! 🚀
