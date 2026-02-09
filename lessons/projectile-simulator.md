# **Project Assignment: Projectile Simulator & Optimizer**

## **Project Overview**

**Objective:** Build a Python program that simulates projectile motion and finds the optimal launch angle for maximum distance. This project mirrors how mechanical engineers use computational tools to simulate physical systems and optimize designs—skills directly applicable to analyzing drone trajectories, vehicle jump angles, or any ballistic motion.

**Why This Project?**
1. **Clear Physics Foundation:** Starts with equations you already know (kinematics)
2. **Progressive Complexity:** Builds from simple calculations to optimization and interactivity
3. **Tangible Results:** Creates visual plots you can share and an interactive tool you can play with
4. **Engineering Relevance:** Teaches numerical simulation, parameter optimization, and visualization—core skills for modern engineering

**Prerequisites:** High school algebra and physics (kinematics). No prior coding experience needed.

**Tools:** Google Colab (colab.research.google.com)

**Estimated Time:** 8-15 hours total (can be spread over multiple sessions)

---

## **Pedagogical Objectives**

By completing this project, you will learn to:
1. **Translate** physical laws into computational models
2. **Structure** code with functions for reusability
3. **Automate** exploration using loops
4. **Visualize** results with clear, labeled plots
5. **Optimize** a design parameter computationally
6. **Create** interactive tools for parameter exploration

---

## **Stage 1: The Basic Trajectory Calculator**

### **Learning Focus: Variables, Basic Math, and Plotting**
*Core Skills: Variables, math operations, numpy arrays, matplotlib plotting*

**Time Estimate:** 2-3 hours

**What You'll Build:** A program that calculates and plots the trajectory of a projectile launched at a specific angle and velocity.

**Assignment:**
1. Create a new Colab notebook titled "Projectile_Simulator"
2. Write code that calculates and plots the trajectory of a ball launched at 45° with 20 m/s initial velocity (ignoring air resistance)

---

### **Step-by-Step Instructions:**

**Step 1: Import Required Libraries**

First, we need to import the tools we'll use:

```python
import numpy as np
import matplotlib.pyplot as plt
```

**What this does:**
- `numpy` (imported as `np`) provides mathematical functions and arrays
- `matplotlib.pyplot` (as `plt`) creates plots and visualizations

---

**Step 2: Define Physical Constants and Initial Conditions**

Now let's set up the parameters for our simulation. Think about what values make sense for these variables:

```python
# Constants for projectile motion
g = 9.81     # gravity (m/s²)

# Initial conditions - fill in reasonable values:
v0 = ???     # initial velocity (m/s) - try a value between 10-30
theta = ???  # launch angle (degrees) - try between 30-60
<!-- PARTIAL_REVEAL -->
# Correct initial values:
g = 9.81    # gravity (m/s²)
v0 = 20     # initial velocity (m/s) - about 45 mph
theta = 45  # launch angle (degrees) - classic projectile angle

# **Why these values?**
# - `g = 9.81 m/s²` is Earth's gravitational acceleration (a constant)
# - `v0 = 20 m/s` is about 45 mph - a reasonable throwing speed
# - `theta = 45°` is a common starting angle for projectile problems
```

**Key Concept:** Variables store values you'll use throughout your code. Using meaningful names like `g` for gravity makes code readable.

---

**Step 3: Convert Angle and Calculate Velocity Components**

Trigonometry time! We need to break the velocity into horizontal and vertical components. Can you fill in the trigonometric functions?

```python
theta_rad = np.radians(theta)  # Convert degrees to radians

# Fill in the correct trig functions (cos or sin?):
vx = v0 * ???(theta_rad)    # Horizontal velocity component - what function should be called on the value?
vy = v0 * ???(theta_rad)    # Vertical velocity component - what function should be called on the value?
<!-- PARTIAL_REVEAL -->
# Complete solution:
theta_rad = np.radians(theta)
vx = v0 * np.cos(theta_rad)    # Horizontal velocity component
vy = v0 * np.sin(theta_rad)    # Vertical velocity component
```

**Why this matters:** Trigonometric functions in programming use radians, not degrees. We break velocity into horizontal (vx) and vertical (vy) components because they're independent in projectile motion.

**Trigonometry reminder:**
- **Cosine** gives the adjacent side (horizontal component)
- **Sine** gives the opposite side (vertical component)
- Think: **SOH-CAH-TOA**

---

**Step 4: Calculate Time of Flight and Create Time Array**

```python
# Total time until ball returns to ground
t_flight = ??? * ??? / g

# Create 100 equally spaced parts
t = np.linspace(0, t_flight, 100)
<!-- PARTIAL_REVEAL -->
# Total time until ball returns to ground
t_flight = 2 * vy / g

# Create 100 equally spaced parts
t = np.linspace(0, t_flight, 100)
```

**Understanding the physics:** 
- Time to peak: vy / g
- Total flight time: double (vy / g) because symmetric trajectory
- numpy ftw: `linspace` creates an array of time values from 0 to t_flight

---

**Step 5: Calculate Position at Each Time Point**

```python
x = vx * t                    # Horizontal position: distance = velocity × time
y = vy * t - 0.5 * g * t**2   # Vertical position: kinematic equation
```

**The physics:** These are the kinematic equations you learned in physics class:
- x(t) = v₀ₓ·t
- y(t) = v₀ᵧ·t - ½g·t²

---

**Step 6: Create the Visualization**

```python
# Create the figure
plt.figure(figsize=(8, 4))

# These operations will all act on the figure we just created above
plt.plot(x, y, linewidth=2, color='#4af2a1')
plt.xlabel('Distance (m)', fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.title('Projectile Trajectory (45°, 20 m/s)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(0, max(y)*1.1)  # Set y-axis to start at ground level
plt.tight_layout()
plt.show()
```

**Visualization tips:**
- `figsize` controls plot dimensions
- `linewidth` and `color` make your plot more professional
- `grid` helps read values
- `ylim` ensures the ground (y=0) is visible

---

### **Checkpoint #1**

**What you should see:** A symmetric parabolic arc that:
- Starts at (0, 0)
- Reaches maximum height around x = 20m
- Returns to ground at approximately x = 40.8m

**Verification:**
```python
print(f"Time of flight: {t_flight:.2f} seconds")
print(f"Maximum height: {max(y):.2f} meters")
print(f"Range: {x[-1]:.2f} meters")
```

**Expected output:**
- Time of flight: ~2.89 seconds
- Maximum height: ~10.19 meters  
- Range: ~40.82 meters

(These values were generated by an LLM and may -- or may not -- be correct! Fun!)

---

### **Common Issues & Debugging**

**Problem:** "NameError: name 'np' is not defined"
- **Solution:** Check that you ran the import cell first

**Problem:** Plot looks like a flat line
- **Solution:** Verify you converted degrees to radians (`np.radians()`)

**Problem:** Trajectory goes below y=0
- **Solution:** Add `plt.ylim(0, max(y)*1.1)` to fix the y-axis

**Debugging tip:** Use `print()` statements to check your variables:
```python
print(f"Angle in radians: {theta_rad}")
print(f"vx: {vx}, vy: {vy}")
print(f"First few x values: {x[:5]}")
```

---

### **Extension Challenge (Optional)**

Try modifying the code to:
1. Launch at 30° instead of 45°
2. Change initial velocity to 15 m/s
3. Plot both trajectories on the same graph with different colors

**Reflection Questions:**
- What happens to the range when you change the angle?
- How does initial velocity affect maximum height?
- Why is the trajectory a parabola?

---

## **Stage 2: Creating a Reusable Function**

### **Learning Focus: Functions and Code Organization**
*Core Skills: Function definition, parameters, return values, documentation*

**Time Estimate:** 1-2 hours

**Why Functions Matter:** Right now, if you want to test different angles or velocities, you have to copy and modify code. Functions let you package code into reusable blocks—a fundamental programming concept.

**Assignment:** Encapsulate your trajectory calculation in a function that can be called with different parameters.

---

### **Step 1: Define the Trajectory Function**

```python
def calculate_trajectory(v0, theta_deg):
    """
    Calculate projectile trajectory for given initial conditions.
    
    Parameters:
    -----------
    v0 : float
        Initial velocity in m/s
    theta_deg : float
        Launch angle in degrees
    
    Returns:
    --------
    x : numpy array
        Horizontal positions (m)
    y : numpy array
        Vertical positions (m)
    t_flight : float
        Total time of flight (s)
    """
    g = 9.81
    theta_rad = np.radians(theta_deg)
    
    vx = v0 * np.cos(theta_rad)
    vy = v0 * np.sin(theta_rad)
    
    t_flight = 2 * vy / g
    t = np.linspace(0, t_flight, 100)
    
    x = vx * t
    y = vy * t - 0.5 * g * t**2
    
    return x, y, t_flight
```

**Function anatomy:**
- `def calculate_trajectory(...)`: Declares the function name and parameters
- `"""docstring"""`: Explains what the function does (good practice!)
- `return x, y, t_flight`: Sends results back to the caller

---

### **Step 2: Test Your Function**

```python
# Test with original values
x1, y1, t1 = calculate_trajectory(20, 45)

# Verify it works the same way
print(f"Range: {x1[-1]:.2f} m")
print(f"Max height: {max(y1):.2f} m")
```

---

### **Step 3: Compare Multiple Trajectories**

```python
# Calculate three different angles
x1, y1, t1 = calculate_trajectory(20, 30)
x2, y2, t2 = calculate_trajectory(20, 45)
x3, y3, t3 = calculate_trajectory(20, 60)

# Plot all three on same graph
plt.figure(figsize=(10, 5))
plt.plot(x1, y1, label='30°', linewidth=2)
plt.plot(x2, y2, label='45°', linewidth=2)
plt.plot(x3, y3, label='60°', linewidth=2)

plt.xlabel('Distance (m)', fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.title('Trajectory Comparison (v₀ = 20 m/s)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**What you'll notice:**
- 30° gives longest flight time but shorter range
- 45° gives maximum range
- 60° goes higher but lands shorter
- 30° and 60° are complementary angles with same range

---

### **Step 4: Create a Helper Function for Range**

```python
def get_range(v0, theta_deg):
    """
    Calculate the range (total horizontal distance) of a projectile.
    
    Parameters:
    -----------
    v0 : float
        Initial velocity in m/s
    theta_deg : float
        Launch angle in degrees
    
    Returns:
    --------
    range : float
        Horizontal distance traveled (m)
    """
    x, y, t_flight = calculate_trajectory(v0, theta_deg)
    return x[-1]  # Return the final x position
```

**Quick test:**
```python
range_45 = get_range(20, 45)
print(f"Range at 45°: {range_45:.2f} m")
```

---

### **Checkpoint #2**

**Success criteria:**
1. Your function produces the same results as Stage 1
2. You can easily test multiple angles
3. The 45° trajectory has the greatest range
4. You understand how to call functions and use return values

**Self-check questions:**
- Can you explain what `return` does?
- Why is it useful to have separate functions for trajectory and range?
- What would happen if you called `get_range(20, 100)`? (Try it!)

---

### **Engineering Insight**

**Functions are the building blocks of engineering software.** When Boeing simulates an aircraft wing or Tesla models battery performance, they use functions to encapsulate complex calculations. This makes code:
- **Reusable:** Write once, use many times
- **Testable:** Easier to verify correctness
- **Maintainable:** Update in one place
- **Readable:** Clear what each piece does

---

## **Stage 3: Finding the Optimal Angle**

### **Learning Focus: Loops, Optimization, and Data Analysis**
*Core Skills: For loops, lists, array operations, finding extrema*

**Time Estimate:** 2-3 hours

**The Challenge:** Instead of guessing the optimal angle, let's use code to systematically test every angle and find the one that gives maximum range.

**Assignment:** Write a program that tests angles from 0° to 90° and determines which gives the greatest distance.

---

### **Step 1: Set Up Data Collection**

```python
# Create empty lists to store results
angles = []
ranges = []
```

**Lists vs Arrays:** We use lists here because we're building them iteratively. We could convert to numpy arrays later if needed.

---

### **Step 2: Loop Through All Angles**

```python
# Test every integer angle from 0 to 90 degrees
for angle in range(0, 91):  # range(0, 91) gives 0, 1, 2, ..., 90
    r = get_range(20, angle)
    angles.append(angle)
    ranges.append(r)
    
print(f"Tested {len(angles)} different angles")
```

**How the loop works:**
- `for angle in range(0, 91)`: Iterate through 0, 1, 2, ... 90
- `get_range(20, angle)`: Calculate range for current angle
- `.append()`: Add values to our lists

---

### **Step 3: Find the Maximum**

```python
# Find the angle that gives maximum range
optimal_angle = angles[np.argmax(ranges)]  # Index of maximum value
max_range = max(ranges)  # The maximum range value

print(f"Optimal angle: {optimal_angle}°")
print(f"Maximum range: {max_range:.2f} m")
```

**Understanding argmax:**
- `max(ranges)` gives the largest value
- `np.argmax(ranges)` gives the *index* of the largest value
- `angles[index]` retrieves the angle at that index

---

### **Step 4: Visualize the Results**

```python
plt.figure(figsize=(10, 5))

# Plot all the data points
plt.plot(angles, ranges, 'b-', linewidth=2, label='Range vs Angle')

# Highlight the optimal point
plt.plot(optimal_angle, max_range, 'ro', markersize=12, 
         label=f'Optimal: {optimal_angle}°', zorder=5)

# Add annotation
plt.annotate(f'{optimal_angle}° → {max_range:.1f}m', 
             xy=(optimal_angle, max_range),
             xytext=(optimal_angle + 10, max_range - 5),
             fontsize=11,
             arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.xlabel('Launch Angle (degrees)', fontsize=12)
plt.ylabel('Range (m)', fontsize=12)
plt.title('Projectile Range vs Launch Angle (v₀ = 20 m/s)', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

### **Checkpoint #3**

**What you should observe:**
1. Optimal angle is exactly 45°
2. The curve is symmetric around 45°
3. Complementary angles (30° & 60°, 20° & 70°) give identical ranges
4. Very low angles (0-10°) and very high angles (80-90°) give poor range

**Verification:**
```python
# Test symmetry
print(f"Range at 30°: {ranges[30]:.2f} m")
print(f"Range at 60°: {ranges[60]:.2f} m")
print(f"Difference: {abs(ranges[30] - ranges[60]):.6f} m")
```

---

### **The Physics Behind It**

**Why is 45° optimal?** The range formula (without air resistance) is:
```
R = (v₀² × sin(2θ)) / g
```

- `sin(2θ)` is maximized when `2θ = 90°`, so `θ = 45°`
- This is why complementary angles give equal range: `sin(2×30°) = sin(60°)` and `sin(2×60°) = sin(120°) = sin(60°)`

**Engineering insight:** We just used a "brute force" search—testing every possibility. For 91 angles, this is fast. For millions of possibilities, engineers use smarter algorithms (gradient descent, genetic algorithms, etc.).

---

### **Extension Challenges**

1. **Finer resolution:** Test angles in 0.1° increments
```python
angles_fine = np.arange(0, 90.1, 0.1)  # 0, 0.1, 0.2, ..., 90.0
```

2. **Different velocities:** How does optimal angle change with velocity? (Hint: it shouldn't!)

3. **Create a heatmap:** Show range as a function of both angle AND velocity

---

## **Stage 4: Adding Air Resistance (Advanced)**

### **Learning Focus: Numerical Integration and Realistic Modeling**
*Core Skills: Differential equations, iterative algorithms, drag forces*

**Time Estimate:** 3-4 hours

**Why This Matters:** Real projectiles experience air resistance, which dramatically changes their trajectory. This stage teaches you to simulate systems where we can't use simple equations.

**Assignment:** Implement air resistance using numerical integration (Euler's method).

---

### **Understanding Air Resistance**

**Two types of drag:**
1. **Linear drag:** F_drag = -k·v (proportional to velocity) 
2. **Quadratic drag:** F_drag = -k·v² (proportional to velocity squared)

We'll use quadratic drag because it's more realistic for most projectiles.

**Key differences from Stage 1:**
- No analytical solution—must simulate step-by-step
- Trajectory is asymmetric (steeper descent)
- Optimal angle becomes less than 45°

---

### **Step 1: The Numerical Integration Function**

```python
def calculate_trajectory_with_drag(v0, theta_deg, k=0.1):
    """
    Calculate projectile trajectory with air resistance using Euler's method.
    
    Parameters:
    -----------
    v0 : float
        Initial velocity (m/s)
    theta_deg : float
        Launch angle (degrees)
    k : float
        Drag coefficient (default 0.1)
        Larger k = more air resistance
    
    Returns:
    --------
    x : numpy array
        Horizontal positions (m)
    y : numpy array  
        Vertical positions (m)
    """
    g = 9.81
    theta_rad = np.radians(theta_deg)
    
    # Initial conditions
    vx = v0 * np.cos(theta_rad)  # Horizontal velocity
    vy = v0 * np.sin(theta_rad)  # Vertical velocity
    x, y = 0, 0  # Starting position
    
    # Storage for trajectory
    x_vals = [x]
    y_vals = [y]
    
    # Time step (smaller = more accurate, but slower)
    dt = 0.01
    
    # Simulation loop: continue while above ground
    while y >= 0:
        # Calculate current speed (magnitude of velocity)
        speed = np.sqrt(vx**2 + vy**2)
        
        # Acceleration components
        # Drag acts opposite to velocity direction
        ax = -k * speed * vx  # Horizontal drag
        ay = -g - k * speed * vy  # Gravity + vertical drag
        
        # Update velocities (Euler's method)
        vx += ax * dt
        vy += ay * dt
        
        # Update positions
        x += vx * dt
        y += vy * dt
        
        # Store current position
        x_vals.append(x)
        y_vals.append(y)
    
    return np.array(x_vals), np.array(y_vals)
```

**Euler's method explained:**
1. Start with initial position and velocity
2. Calculate acceleration based on current velocity
3. Update velocity: v_new = v_old + a×dt
4. Update position: x_new = x_old + v×dt
5. Repeat until projectile hits ground

**The dt parameter:** Smaller timesteps give more accurate results but take longer to compute. 0.01 seconds is a good balance.

---

### **Step 2: Compare With and Without Drag**

```python
# Calculate both trajectories
x_no_drag, y_no_drag, _ = calculate_trajectory(20, 45)
x_drag, y_drag = calculate_trajectory_with_drag(20, 45, k=0.1)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(x_no_drag, y_no_drag, 'b--', linewidth=2, 
         label='No air resistance', alpha=0.7)
plt.plot(x_drag, y_drag, 'r-', linewidth=2.5, 
         label='With air resistance (k=0.1)')

plt.xlabel('Distance (m)', fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.title('Effect of Air Resistance on Trajectory', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print comparison
print(f"Range without drag: {x_no_drag[-1]:.2f} m")
print(f"Range with drag: {x_drag[-1]:.2f} m")
print(f"Reduction: {(1 - x_drag[-1]/x_no_drag[-1])*100:.1f}%")
```

---

### **Step 3: Find New Optimal Angle**

```python
# Test angles with air resistance
angles_drag = []
ranges_drag = []

for angle in range(0, 91):
    x, y = calculate_trajectory_with_drag(20, angle, k=0.1)
    angles_drag.append(angle)
    ranges_drag.append(x[-1])

# Find optimal
optimal_with_drag = angles_drag[np.argmax(ranges_drag)]
max_range_drag = max(ranges_drag)

print(f"Optimal angle with drag: {optimal_with_drag}°")
print(f"Maximum range with drag: {max_range_drag:.2f} m")
```

---

### **Step 4: Visualize the Shift**

```python
plt.figure(figsize=(12, 5))

# Plot both curves
plt.plot(angles, ranges, 'b-', linewidth=2, label='No drag', alpha=0.7)
plt.plot(angles_drag, ranges_drag, 'r-', linewidth=2, label='With drag (k=0.1)')

# Mark optimal points
plt.plot(45, ranges[45], 'bo', markersize=10)
plt.plot(optimal_with_drag, max_range_drag, 'ro', markersize=10)

plt.xlabel('Launch Angle (degrees)', fontsize=12)
plt.ylabel('Range (m)', fontsize=12)
plt.title('Optimal Angle Shifts with Air Resistance', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

### **Checkpoint #4**

**What you should see:**
1. Trajectory with drag is **shorter** (reduced range)
2. Trajectory with drag is **asymmetric** (steeper descent)
3. Optimal angle is **less than 45°** (typically 35-40° depending on k)
4. The curve is no longer symmetric

**Why the asymmetry?** The projectile slows down as it flies, so it's moving slower on the way down. Less horizontal velocity = less horizontal distance covered during descent.

---

### **Extension: Explore Drag Coefficient**

```python
# Test different drag values
k_values = [0, 0.05, 0.1, 0.2, 0.3]
colors = ['blue', 'green', 'yellow', 'orange', 'red']

plt.figure(figsize=(10, 6))

for k, color in zip(k_values, colors):
    if k == 0:
        x, y, _ = calculate_trajectory(20, 45)
    else:
        x, y = calculate_trajectory_with_drag(20, 45, k=k)
    plt.plot(x, y, linewidth=2, label=f'k = {k}', color=color)

plt.xlabel('Distance (m)', fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.title('Impact of Drag Coefficient', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## **Stage 5: Creating an Interactive Tool (Advanced)**

### **Learning Focus: Interactive Visualization**
*Core Skills: ipywidgets, real-time parameter exploration, user interfaces*

**Time Estimate:** 1-2 hours

**The Goal:** Create sliders that let you adjust parameters in real-time and see the trajectory update instantly. This is how engineers explore design spaces interactively.

---

### **Step 1: Import Widget Tools**

```python
from ipywidgets import interact, FloatSlider, IntSlider
```

---

### **Step 2: Create Interactive Function**

```python
def interactive_trajectory(v0=20.0, theta=45.0, k=0.0):
    """
    Interactive projectile trajectory plotter.
    
    Parameters adjusted via sliders:
    - v0: Initial velocity (1-50 m/s)
    - theta: Launch angle (0-90 degrees)
    - k: Drag coefficient (0-0.5)
    """
    # Calculate trajectory based on drag
    if k == 0:
        x, y, _ = calculate_trajectory(v0, theta)
    else:
        x, y = calculate_trajectory_with_drag(v0, theta, k)
    
    # Calculate range and max height
    range_val = x[-1] if len(x) > 0 else 0
    max_height = max(y) if len(y) > 0 else 0
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=3, color='#4af2a1')
    
    # Add annotations
    plt.text(0.02, 0.98, 
             f'Range: {range_val:.2f} m\nMax Height: {max_height:.2f} m', 
             transform=plt.gca().transAxes,
             fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel('Distance (m)', fontsize=12)
    plt.ylabel('Height (m)', fontsize=12)
    plt.title(f'Projectile: v₀={v0:.1f} m/s, θ={theta:.1f}°, k={k:.2f}', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(y)*1.2 if len(y) > 0 and max(y) > 0 else 20)
    plt.xlim(0, max(x)*1.1 if len(x) > 0 and max(x) > 0 else 50)
    plt.tight_layout()
    plt.show()
```

---

### **Step 3: Add Interactive Sliders**

```python
# Create interactive widget
interact(interactive_trajectory,
         v0=FloatSlider(min=1, max=50, value=20, step=0.5, 
                       description='Velocity (m/s):'),
         theta=FloatSlider(min=0, max=90, value=45, step=1, 
                          description='Angle (°):'),
         k=FloatSlider(min=0, max=0.5, value=0, step=0.01, 
                      description='Drag Coeff:'))
```

---

### **Final Checkpoint**

**You should have:**
1. Three sliders that control velocity, angle, and drag
2. Plot updates instantly when you move a slider
3. Range and max height displayed on the plot
4. Ability to explore how each parameter affects the trajectory

**Experiments to try:**
- Set k=0 and find the optimal angle (should be 45°)
- Increase k to 0.3 and find new optimal (around 35-40°)
- Max out velocity at 50 m/s and see the scale change
- Set angle to 90° and watch it go straight up

---

## **Project Deliverables**

Submit your Colab notebook with:

### **1. Code (Well-Commented)**
- All stages completed and working
- Comments explaining key steps
- Clear variable names

### **2. Visualizations**
Include at least:
- Basic trajectory plot (Stage 1)
- Multiple angle comparison (Stage 2)
- Range vs angle optimization (Stage 3)
- Drag vs no-drag comparison (Stage 4, if attempted)
- Screenshot of interactive tool (Stage 5, if attempted)

### **3. Written Responses**

Answer these questions in markdown cells:

**Analysis Questions:**
1. What happens to the optimal angle when you increase initial velocity from 10 m/s to 30 m/s (with no drag)?
2. How does air resistance affect:
   - The shape of the trajectory?
   - The optimal launch angle?
   - The symmetry of the path?
3. For the same initial velocity, which goes farther: a baseball (high drag) or a golf ball (low drag)? Why?

**Application Questions:**
4. Describe three real engineering applications that could use this type of simulation:
   - Example: "Drone delivery systems optimizing package drop angles and velocities"
5. If you were designing a catapult for a competition, how would you use this code to optimize your design?

**Reflection:**
6. What was the most challenging part of this project?
7. What did you learn about how programming can help solve engineering problems?
8. What would you add or change if you had more time?

---

## **Success Criteria**

### **Complete (Meets Expectations)**
- ✅ Stages 1-3 fully functional
- ✅ Code runs without errors
- ✅ Basic plots are clear and labeled
- ✅ Analysis questions answered

### **Proficient (Exceeds Expectations)**
- ✅ All of above, plus:
- ✅ Code is well-commented and organized
- ✅ Plots are professional-looking with proper labels
- ✅ Attempted Stage 4 with working drag simulation
- ✅ Thoughtful, detailed responses to questions

### **Excellent (Outstanding)**
- ✅ All of above, plus:
- ✅ Stage 5 interactive tool working
- ✅ Additional visualizations or analysis beyond requirements
- ✅ Creative extensions (e.g., 3D trajectories, wind effects, multiple projectiles)
- ✅ Clear documentation that could teach someone else

### **Exceptional (Goes Above and Beyond)**
- ✅ All of above, plus:
- ✅ Original extension related to your interests (e.g., "Drone Payload Delivery Optimizer")
- ✅ Exploration of advanced topics (different drag models, wind, landing predictions)
- ✅ Could be used as a teaching resource for future students

---

## **Tips for Success**

### **Before You Start**
- Read through the entire assignment
- Watch the intro video on Google Colab basics
- Set up your notebook with good section headers

### **While Working**
- Work through stages sequentially—don't skip ahead
- Test your code frequently (run cells often!)
- Use `print()` to debug when something doesn't work
- Save your work regularly (Colab auto-saves, but be safe)

### **When You Get Stuck**
1. **Read the error message** carefully—it usually tells you what's wrong
2. **Print variables** to see what values they contain
3. **Google the error**—you're not the first person to encounter it!
4. **Check your syntax**—missing parentheses, colons, or indentation?
5. **Ask for help**—that's what learning is all about

### **Best Practices**
- Comment your code as you write it
- Use meaningful variable names (`velocity` not `v1`)
- Keep your plots clean and labeled
- Write in markdown cells to explain your thinking

---

## **Extensions & Next Steps**

After completing this project, you could:

1. **3D Trajectories:** Add a third dimension for cross-wind effects
2. **Different Projectiles:** Model varying mass and drag coefficients (baseball vs golf ball)
3. **Landing Zone Prediction:** Calculate where a projectile will land given wind
4. **Optimization Algorithms:** Use scipy.optimize instead of brute force
5. **Real Data Comparison:** Find actual trajectory data and validate your model
6. **Game Integration:** Build a simple game where users try to hit targets

---

## **Additional Resources**

**Python & Colab:**
- Google Colab Tutorial: [colab.research.google.com](https://colab.research.google.com)
- Python Documentation: [docs.python.org](https://docs.python.org)

**Physics:**
- Khan Academy: Projectile Motion
- HyperPhysics: Trajectory Calculations

**Visualization:**
- Matplotlib Gallery: [matplotlib.org/gallery](https://matplotlib.org/gallery)
- Seaborn Tutorial (advanced plotting)

**Engineering Applications:**
- NASA Trajectory Simulations
- Ballistics Research Laboratory Reports

---

Remember: **The goal isn't perfection—it's progress.** Every engineer debugs code daily. Every plot goes through multiple iterations. The skills you're building here—translating physics to code, visualizing results, optimizing parameters—are exactly what you'll use in your career.

**You've got this! 🚀**
