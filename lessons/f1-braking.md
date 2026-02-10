# **Project Assignment: F1 Braking & Corner Entry Simulator**

## **Project Overview**

**Objective:** Build a Python program that models F1 car braking dynamics under traction limits and finds the optimal braking point for corner entry. This project teaches you how mechanical engineers analyze vehicle dynamics, tire grip limits, and race strategy—skills applicable to autonomous vehicle control, motorsports engineering, or any system with friction-limited motion.

**Why This Project?**
1. **New Physics Context:** Applies Newton's 2nd law to real-world traction limits (friction circles)
2. **Performance Engineering:** Introduces the "how late can you brake?" optimization problem that F1 engineers solve
3. **Tangible Results:** Creates visualizations showing braking zones and lap time improvements
4. **Career Relevance:** Teaches vehicle dynamics, deceleration analysis, and performance optimization—core skills in automotive and racing engineering

**Prerequisites:** High school algebra and physics (Newton's laws, friction). Completion of projectile simulator project helpful but not required.

**Tools:** Google Colab (colab.research.google.com)

**Estimated Time:** 8-12 hours total (can be spread over multiple sessions)

---

## **Pedagogical Objectives**

By completing this project, you will learn to:
1. **Apply** Newton's 2nd law in a friction-limited deceleration context
2. **Model** tire grip limits and the friction circle concept
3. **Calculate** time-to-stop and distance-to-stop under maximum braking
4. **Optimize** braking points for corner entry ("how late can you brake?")
5. **Visualize** braking zones and compare different strategies
6. **Analyze** the trade-offs between braking late and corner entry speed

---

## **Stage 1A: Understanding F1 Braking Physics**

### **Learning Focus: Core Deceleration Concepts**
*Core Skills: Variables, basic math operations, unit conversions, Newton's 2nd law application*

**Time Estimate:** 1-1.5 hours

**What You'll Build:** A simple calculator that computes how long it takes an F1 car to stop from a given speed.

**Assignment:**
1. Create a new Colab notebook titled "F1_Braking_Simulator"
2. Calculate stopping time and distance for an F1 car braking from 200 km/h

---

### **The Physics Behind F1 Braking**

**Key Concept: Friction Circle**

In racing, tires have a maximum grip limit. Think of it as a "friction circle"—the tire can only generate so much total force before it starts to slip. For pure braking (no cornering), all that grip goes into slowing the car.

**The relationship:**
- **Maximum braking force:** F_max = μ × m × g
  - μ (mu) = coefficient of friction (tire grip)
  - m = mass of the car
  - g = gravitational acceleration (9.81 m/s²)

- **Maximum deceleration:** a_max = F_max / m = μ × g
  - **Notice: mass cancels out!** Deceleration depends only on tire grip, not car weight
  - This is why we can define mass but won't actually need it in calculations

**F1 tire grip values:**
- Modern F1 slick tires: μ ≈ 1.8-2.2 (in optimal conditions)
- For comparison, road car tires: μ ≈ 0.7-0.9
- This means F1 cars can pull almost 2g of deceleration!

---

### **Step-by-Step Instructions:**

**Step 1: Import Required Libraries**

```python
import numpy as np
import matplotlib.pyplot as plt
```

**What this does:**
- `numpy` provides mathematical functions and arrays
- `matplotlib.pyplot` creates plots and visualizations

---

**Step 2: Define Physical Constants**

Let's start simple—just the constants we need:

```python
# Physical constants
g = 9.81  # gravitational acceleration (m/s²)

# F1 tire grip - fill in a reasonable value:
mu = ???  # coefficient of friction - F1 slick tires (hint: between 1.8 and 2.2)
<!-- PARTIAL_REVEAL -->
# Complete solution:
g = 9.81   # gravitational acceleration (m/s²)
mu = 2.0   # coefficient of friction - modern F1 slick tires

# **Why mu = 2.0?**
# F1 slick tires in optimal conditions provide 1.8-2.2g of grip
# We'll use 2.0 as a typical "good conditions" value
```

**Key Concept:** The coefficient of friction (μ) determines how much grip the tires have. Higher μ = better grip = shorter stopping distance.

---

**Step 3: Set Initial Speed**

Now define the speed we're braking from:

```python
# Initial speed - fill in the value:
v0_kmh = ???  # km/h - try a typical corner approach speed (150-250 km/h)
<!-- PARTIAL_REVEAL -->
# Complete solution:
v0_kmh = 200  # km/h - approaching a medium-speed corner

# **Why 200 km/h?**
# This is realistic for corners like:
# - Monza's first chicane
# - Barcelona's Turn 1
# - Austin's Turn 12
```

---

**Step 4: Convert Speed and Calculate Maximum Deceleration**

Physics works in m/s, not km/h, so we need to convert:

```python
# Convert km/h to m/s - fill in the conversion:
v0 = v0_kmh / ???  # Hint: divide by 3.6
<!-- PARTIAL_REVEAL -->
# Complete solution:
v0 = v0_kmh / 3.6  # Convert: 200 km/h = 55.56 m/s

# **The conversion:**
# 1 km/h = 1000m / 3600s = 1/3.6 m/s
# So to convert km/h → m/s, divide by 3.6
```

Now calculate the maximum deceleration:

```python
# Fill in the deceleration formula:
a_max = ??? * ???  # Maximum deceleration = mu × g
<!-- PARTIAL_REVEAL -->
# Complete solution:
a_max = mu * g     # Maximum deceleration = 2.0 × 9.81 = 19.62 m/s²

# **The physics:**
# From F = ma and F_max = μmg:
# a_max = F_max / m = (μmg) / m = μg
# Notice: mass (m) cancelled out!
# This means a 798kg F1 car and a 900kg F1 car stop in the same distance
# (assuming same tires and conditions)
```

**Engineering Insight:** The fact that deceleration is independent of mass is counterintuitive but true! A heavy car and light car stop in the same distance with the same tires. However, the heavy car's brakes work harder (more energy to dissipate).

---

**Step 5: Calculate Stopping Time**

Use the kinematic equation v = v₀ - at. When the car stops, v = 0:

```python
# Fill in the stopping time formula:
# Starting from: v = v0 - a*t
# When stopped: 0 = v0 - a*t
# Solve for t: t = ???
t_stop = ??? / ???
<!-- PARTIAL_REVEAL -->
# Complete solution:
t_stop = v0 / a_max  # Time to stop: t = v0 / a

# **The algebra:**
# v = v0 - at
# At stop: 0 = v0 - a*t
# Rearrange: a*t = v0
# Solve: t = v0 / a
```

---

**Step 6: Calculate Stopping Distance**

Use the kinematic equation v² = v₀² - 2ad. When the car stops, v = 0:

```python
# Fill in the stopping distance formula:
# Starting from: v² = v0² - 2*a*d
# When stopped: 0 = v0² - 2*a*d
# Solve for d: d = ???
d_stop = ???**2 / (2 * ???)
<!-- PARTIAL_REVEAL -->
# Complete solution:
d_stop = v0**2 / (2 * a_max)  # Distance to stop: d = v0² / (2a)

# **The algebra:**
# v² = v0² - 2ad
# At stop: 0 = v0² - 2ad
# Rearrange: 2ad = v0²
# Solve: d = v0² / (2a)
```

**Physics reminder:** These are the standard kinematic equations:
- **v = v₀ - at** (velocity during constant deceleration)
- **v² = v₀² - 2ad** (velocity-displacement relationship)

---

**Step 7: Display Results**

```python
# Print the results
print("=" * 50)
print("F1 BRAKING ANALYSIS")
print("=" * 50)
print(f"Initial speed: {v0_kmh} km/h ({v0:.2f} m/s)")
print(f"Tire grip (μ): {mu}")
print(f"Maximum deceleration: {a_max:.2f} m/s² ({a_max/g:.2f}g)")
print(f"Time to stop: {t_stop:.2f} seconds")
print(f"Stopping distance: {d_stop:.2f} meters")
print("=" * 50)
```

---

### **Checkpoint #1**

**What you should see:** Output showing:
- Initial speed: 200 km/h (55.56 m/s)
- Maximum deceleration: ~19.62 m/s² (2.0g)
- Time to stop: ~2.83 seconds
- Stopping distance: ~78.6 meters

**Reality Check:**
An F1 car braking from 200 km/h to 0 in under 3 seconds over less than 80 meters demonstrates the incredible grip of F1 slick tires!

**Comparison Challenge:**
```python
# How does a road car compare?
mu_road = 0.8  # Road car tires
a_road = mu_road * g
t_road = v0 / a_road
d_road = v0**2 / (2 * a_road)

print(f"\nRoad car (μ={mu_road}):")
print(f"  Time to stop: {t_road:.2f}s (+{t_road - t_stop:.2f}s)")
print(f"  Distance: {d_road:.1f}m (+{d_road - d_stop:.1f}m)")
print(f"  F1 is {d_road/d_stop:.1f}x better!")
```

**Expected:** Road car needs ~7.1 seconds and ~196 meters—2.5× longer!

---

### **Extension Challenge (Optional)**

**Experiment:**
1. Change `v0_kmh` to 100, then 300, then 400. How does stopping distance change?
2. What pattern do you notice? (Hint: d ∝ v₀²)
3. Set `mu = 1.2` (wet conditions). How much longer does braking take?

---

## **Stage 1B: Visualizing Braking Dynamics**

### **Learning Focus: Creating Clear Visualizations**
*Core Skills: matplotlib plotting, numpy arrays, professional plot formatting*

**Time Estimate:** 1-1.5 hours

**What You'll Build:** A plot showing how velocity decreases during braking.

**Assignment:** Create a professional-looking velocity vs time graph that shows the braking process.

---

### **Step 1: Create Time Array**

We'll use numpy to create an array of time points from 0 to t_stop:

```python
# Create 100 time points from start to stop
t = np.linspace(0, t_stop, 100)
```

**What this does:** `linspace` creates 100 evenly-spaced values between 0 and t_stop. This gives us smooth curves when plotting.

---

**Step 2: Calculate Velocity at Each Time Point**

Use the equation v(t) = v₀ - at:

```python
# Fill in the velocity equation:
v = ??? - ??? * t  # v(t) = v0 - a*t
<!-- PARTIAL_REVEAL -->
# Complete solution:
v = v0 - a_max * t  # Velocity decreases linearly with time
```

**What you've created:** An array of 100 velocity values, one for each time point.

---

**Step 3: Create the Plot**

Now let's visualize it! Fill in the plotting functions:

```python
plt.figure(figsize=(10, 5))

# Fill in the plotting commands (use the hints below):
plt.???(t, v * 3.6, linewidth=2, color='#e10600')  # Hint: creates line plot
plt.???('Time (s)', fontsize=12)                    # Hint: labels x-axis
plt.???('Velocity (km/h)', fontsize=12)             # Hint: labels y-axis
plt.???('F1 Braking: Velocity vs Time', fontsize=14, fontweight='bold')  # Hint: adds title
plt.???(True, alpha=0.3)                            # Hint: adds grid lines

plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)  # Ground reference
plt.tight_layout()
plt.show()
<!-- PARTIAL_REVEAL -->
# Complete solution:
plt.figure(figsize=(10, 5))
plt.plot(t, v * 3.6, linewidth=2, color='#e10600')  # F1 red color, convert to km/h
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Velocity (km/h)', fontsize=12)
plt.title('F1 Braking: Velocity vs Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

**Plotting functions you used:**
- `plt.plot()`: Creates line plots (x vs y)
- `plt.xlabel()` / `plt.ylabel()`: Add axis labels
- `plt.title()`: Adds plot title
- `plt.grid()`: Toggles grid lines (True/False, alpha controls transparency)

---

### **Checkpoint #2**

**What you should see:** A straight line declining from 200 km/h to 0 over ~2.83 seconds.

**Why is it straight?** Because deceleration is constant (maximum braking), velocity decreases linearly: v = v₀ - (constant)×t

**Verification:**
```python
# Check velocity at halfway point
halfway_time = t_stop / 2
v_halfway = v0 - a_max * halfway_time
print(f"At t={halfway_time:.2f}s (halfway), velocity is {v_halfway*3.6:.1f} km/h")
# Expected: ~100 km/h (exactly half of 200 km/h)
```

---

### **Common Issues & Debugging**

**Problem:** "NameError: name 'np' is not defined"
- **Solution:** Run the import cell first

**Problem:** Stopping distance seems unrealistic
- **Solution:** Check that you converted km/h to m/s (divide by 3.6)

**Problem:** Plot shows velocity in m/s instead of km/h
- **Solution:** Multiply by 3.6 when plotting: `v * 3.6`

**Debugging tip:** Add print statements to check your work:
```python
print(f"v0 in m/s: {v0}")
print(f"a_max: {a_max}")
print(f"Expected pattern: velocity should decrease linearly")
```

---

### **Extension Challenge (Optional)**

**Add a "G-Force Meter" annotation:**
```python
# Add text showing g-forces experienced
plt.text(0.7, 0.9, f'Driver feels: {a_max/g:.1f}g forward\n(~{a_max/g * 70:.0f} kg on 70kg driver)', 
         transform=plt.gca().transAxes,
         fontsize=10,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
```

This shows the physical force drivers experience during braking!

---

## **Stage 2: Braking Zone Visualization & Comparison**

### **Learning Focus: Distance-Velocity Relationship and Multi-Scenario Analysis**
*Core Skills: Position calculations, velocity-distance curves, function creation, comparative analysis*

**Time Estimate:** 2-3 hours

**What You'll Build:** Visualizations showing where the car is at each speed during braking, plus comparisons of different scenarios (dry vs wet, different speeds).

---

### **Step 1: Calculate Position During Braking**

So far we've looked at velocity vs *time*. Now let's find position vs *time*:

```python
# Create time array (same as before)
t = np.linspace(0, t_stop, 100)
v_t = v0 - a_max * t  # Velocity over time

# Fill in the position equation:
# Position: x(t) = v0*t - 0.5*a*t²
# (Same form as projectile motion, but horizontal!)
x_t = ??? * t - 0.5 * ??? * t**2
<!-- PARTIAL_REVEAL -->
# Complete solution:
x_t = v0 * t - 0.5 * a_max * t**2  # Position increases as car moves forward

# **Connection to projectiles:**
# Same kinematic form: x = v0*t - 0.5*a*t²
# But here 'a' is horizontal deceleration (slowing the car)
# vs projectile where 'g' was vertical acceleration (pulling down)
```

---

**Step 2: The Critical Plot - Velocity vs Distance**

Here's the plot F1 engineers actually use: "At what speed is the car at each point in the braking zone?"

This requires a different approach. We'll calculate velocity as a function of *distance*, not time.

From the kinematic equation: v² = v₀² - 2ad

Solving for v: v = √(v₀² - 2ad)

```python
# Create distance array from 0 to stopping distance
x_array = np.linspace(0, d_stop, 100)

# Fill in the velocity-distance formula:
v_x = np.sqrt(???**2 - 2 * ??? * x_array)  # v(x) = sqrt(v0² - 2ax)
<!-- PARTIAL_REVEAL -->
# Complete solution:
x_array = np.linspace(0, d_stop, 100)
v_x = np.sqrt(v0**2 - 2 * a_max * x_array)  # Velocity as function of distance
```

Now plot it:

```python
plt.figure(figsize=(10, 6))
plt.plot(x_array, v_x * 3.6, linewidth=3, color='#e10600', label='Velocity')
plt.xlabel('Distance from Braking Point (m)', fontsize=12)
plt.ylabel('Velocity (km/h)', fontsize=12)
plt.title('F1 Braking Zone: Velocity vs Distance', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add reference lines
plt.axhline(y=v0_kmh, color='green', linestyle='--', alpha=0.5, 
            label=f'Entry: {v0_kmh} km/h')
plt.axvline(x=d_stop, color='red', linestyle='--', alpha=0.5, 
            label=f'Stop: {d_stop:.1f}m')

plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
```

**Engineering Insight:** This plot answers: "If I'm 50 meters from the corner, what speed should I be at?" It's fundamental for finding optimal braking points!

---

### **Checkpoint #3**

**What you should observe:**
- The curve is NOT straight (unlike velocity vs time)
- It's steeper at the beginning (rapid deceleration)
- Flattens as it approaches the stopping point
- Total distance: ~78.6 meters

**Challenge Question:** At the halfway point (39.3m), what's the velocity?
```python
halfway_dist = d_stop / 2
v_halfway = np.sqrt(v0**2 - 2 * a_max * halfway_dist)
print(f"At {halfway_dist:.1f}m, velocity is {v_halfway*3.6:.1f} km/h")
```

**Expected:** ~141 km/h (NOT 100 km/h!) This shows velocity doesn't decrease linearly with distance—it's quadratic!

---

### **Step 3: Create a Comparison Function**

Let's write a reusable function to calculate braking for any scenario:

```python
def calculate_braking(v0_kmh, mu, label, color):
    """
    Calculate braking parameters for given conditions.
    
    Parameters:
    - v0_kmh: Initial velocity (km/h)
    - mu: Coefficient of friction
    - label: Plot label
    - color: Plot color
    
    Returns: x_array, v_array (in km/h), d_stop, t_stop
    """
    # Fill in the calculations:
    v0 = v0_kmh / ???
    a_max = ??? * g
    t_stop = ??? / ???
    d_stop = ???**2 / (2 * ???)
    
    x_array = np.linspace(0, d_stop, 100)
    v_array = np.sqrt(v0**2 - 2 * a_max * x_array) * 3.6  # Convert to km/h
    
    return x_array, v_array, d_stop, t_stop
<!-- PARTIAL_REVEAL -->
# Complete solution:
def calculate_braking(v0_kmh, mu, label, color):
    """
    Calculate braking parameters for given conditions.
    
    Parameters:
    - v0_kmh: Initial velocity (km/h)
    - mu: Coefficient of friction
    - label: Plot label
    - color: Plot color
    
    Returns: x_array, v_array (in km/h), d_stop, t_stop
    """
    v0 = v0_kmh / 3.6
    a_max = mu * g
    t_stop = v0 / a_max
    d_stop = v0**2 / (2 * a_max)
    
    x_array = np.linspace(0, d_stop, 100)
    v_array = np.sqrt(v0**2 - 2 * a_max * x_array) * 3.6
    
    return x_array, v_array, d_stop, t_stop
```

---

**Step 4: Compare Dry vs Wet Conditions**

Now use the function to compare scenarios:

```python
# Define scenarios to compare
scenarios = [
    (200, 2.0, 'Dry (μ=2.0)', '#e10600'),      # Standard F1 grip
    (200, 1.2, 'Wet (μ=1.2)', '#0066cc'),      # Wet weather
    (200, 0.8, 'Very Wet (μ=0.8)', '#00cc66')  # Heavy rain
]

plt.figure(figsize=(12, 6))

for v0_kmh, mu, label, color in scenarios:
    x, v, d_stop, t_stop = calculate_braking(v0_kmh, mu, label, color)
    plt.plot(x, v, linewidth=2.5, label=f'{label} - Stop: {d_stop:.1f}m', color=color)

plt.xlabel('Distance from Braking Point (m)', fontsize=12)
plt.ylabel('Velocity (km/h)', fontsize=12)
plt.title('Braking Zone Comparison: Dry vs Wet Conditions', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
```

**What you should observe:**
- Dry conditions (μ=2.0): ~78m stopping distance
- Wet conditions (μ=1.2): ~131m (67% longer!)
- Very wet (μ=0.8): ~196m (2.5× longer!)

**Real-world insight:** This is why F1 drivers brake MUCH earlier in the rain—tire grip drops dramatically.

---

### **Extension Challenge (Optional)**

**Compare different entry speeds:**
```python
speeds = [150, 200, 250, 300]  # Different corner types
colors = ['#00cc66', '#e10600', '#ff9900', '#cc00cc']

plt.figure(figsize=(12, 6))
for speed, color in zip(speeds, colors):
    x, v, d_stop, t_stop = calculate_braking(speed, 2.0, f'{speed} km/h', color)
    plt.plot(x, v, linewidth=2.5, label=f'{speed} km/h → {d_stop:.1f}m', color=color)

plt.xlabel('Distance (m)', fontsize=12)
plt.ylabel('Velocity (km/h)', fontsize=12)
plt.title('How Speed Affects Braking Distance (μ=2.0)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
```

**Analysis:** Notice how doubling speed (150→300) quadruples stopping distance (45m→353m). That's the v² term in action!

---

## **Stage 3: The "Late Braking" Challenge**

### **Learning Focus: Optimization and Strategic Decision-Making**
*Core Skills: Functions, optimization logic, trade-off analysis, competitive thinking*

**Time Estimate:** 2-3 hours

**The Challenge:** In F1, you don't brake to a stop—you brake to a **target corner entry speed**. The question is: **How late can you brake and still hit that speed?**

This is where races are won and lost. Braking 5 meters later than your opponent = carrying more speed down the straight = overtaking opportunity!

---

### **Step 1: The Late Braking Problem**

**Scenario:** Lewis Hamilton vs Max Verstappen approaching Turn 1 at Monza:
- **Straight speed:** Both at 280 km/h
- **Target corner entry:** 120 km/h (any faster = run wide)
- **Question:** If Max brakes 10m later than Lewis, what advantage does he gain?

**The physics:** We need to find the distance required to slow from v₁ (straight) to v₂ (corner), not to zero.

From v² = v₁² - 2ad, solving for d: **d = (v₁² - v₂²) / (2a)**

---

**Step 2: Create the Braking Distance Function**

```python
def braking_distance(v_entry_kmh, v_corner_kmh, mu):
    """
    Calculate minimum braking distance from entry speed to corner speed.
    
    Parameters:
    - v_entry_kmh: Speed when braking starts (km/h)
    - v_corner_kmh: Target corner entry speed (km/h)
    - mu: Coefficient of friction
    
    Returns:
    - distance: Required braking distance (meters)
    - time: Time spent braking (seconds)
    """
    # Fill in the conversions and calculations:
    v_entry = ??? / 3.6
    v_corner = ??? / 3.6
    a_max = ??? * g
    
    # Distance: d = (v1² - v2²) / (2a)
    distance = (???**2 - ???**2) / (2 * ???)
    
    # Time: t = (v1 - v2) / a
    time = (??? - ???) / ???
    
    return distance, time
<!-- PARTIAL_REVEAL -->
# Complete solution:
def braking_distance(v_entry_kmh, v_corner_kmh, mu):
    """
    Calculate minimum braking distance from entry speed to corner speed.
    
    Parameters:
    - v_entry_kmh: Speed when braking starts (km/h)
    - v_corner_kmh: Target corner entry speed (km/h)
    - mu: Coefficient of friction
    
    Returns:
    - distance: Required braking distance (meters)
    - time: Time spent braking (seconds)
    """
    v_entry = v_entry_kmh / 3.6
    v_corner = v_corner_kmh / 3.6
    a_max = mu * g
    
    distance = (v_entry**2 - v_corner**2) / (2 * a_max)
    time = (v_entry - v_corner) / a_max
    
    return distance, time
```

---

**Step 3: Test the Monza Scenario**

```python
# Monza Turn 1 braking
v_straight = 280  # km/h
v_corner = 120    # km/h
mu_dry = 2.0

d_brake, t_brake = braking_distance(v_straight, v_corner, mu_dry)

print("=" * 60)
print("MONZA TURN 1 - BRAKING ANALYSIS")
print("=" * 60)
print(f"Approach speed:     {v_straight} km/h")
print(f"Corner entry speed: {v_corner} km/h")
print(f"Speed to scrub:     {v_straight - v_corner} km/h")
print("-" * 60)
print(f"BRAKING POINT:      {d_brake:.1f} meters before turn-in")
print(f"BRAKING TIME:       {t_brake:.2f} seconds")
print("=" * 60)
```

**Expected output:**
- Braking distance: ~211.6 meters
- Braking time: ~2.27 seconds

**Reality check:** At 280 km/h, you're covering 77.8 m/s. Hitting a braking point within ±1 meter requires ±0.013 second reaction time precision!

---

### **Step 4: THE BATTLE - Lewis vs Max**

Now let's simulate the late braking advantage:

```python
print("\n" + "=" * 70)
print("LATE BRAKING BATTLE: HAMILTON vs VERSTAPPEN")
print("=" * 70)

# Optimal braking point
d_optimal = d_brake

# Lewis: Brakes 10m early (conservative approach)
lewis_brake_point = d_optimal + 10

# Calculate Lewis's speed at the optimal braking point
# He's already been braking for 10m
v_lewis_at_optimal = np.sqrt((v_straight/3.6)**2 - 2 * (mu_dry * g) * 10) * 3.6

print(f"\nLEWIS (conservative):")
print(f"  Brakes at:        {lewis_brake_point:.1f}m mark")
print(f"  Speed at optimal: {v_lewis_at_optimal:.1f} km/h")

print(f"\nMAX (aggressive):")
print(f"  Brakes at:        {d_optimal:.1f}m mark")  
print(f"  Speed at optimal: {v_straight} km/h")

print(f"\n" + "=" * 70)
print(f"MAX'S ADVANTAGE:")
print(f"  Speed difference: {v_straight - v_lewis_at_optimal:.1f} km/h")
print(f"  Time advantage:   ~{10 / (v_straight/3.6):.3f} seconds")
print(f"  Position gain:    ~{10:.1f} meters")
print("=" * 70)

print("\nINSIGHT: Braking just 10m later means carrying ~26 km/h more speed")
print("at the optimal point. That's the difference between P1 and P2!")
```

**What this shows:** Late braking isn't just about bravery—it's physics. Every meter counts!

---

### **Checkpoint #4**

**Gamified Challenge:** Can you find the braking point for these famous corners?

```python
corners = {
    "Monaco Hairpin": (180, 50),      # Slowest corner in F1
    "Silverstone Copse": (310, 200),  # High-speed corner
    "Spa Bus Stop": (270, 90),        # Heavy braking chicane
}

print("\n" + "=" * 60)
print("F1 CORNER BRAKING CHALLENGE")
print("=" * 60)

for corner_name, (v_in, v_out) in corners.items():
    d, t = braking_distance(v_in, v_out, 2.0)
    print(f"\n{corner_name}:")
    print(f"  {v_in} → {v_out} km/h")
    print(f"  Braking zone: {d:.1f} meters")
    print(f"  Braking time: {t:.2f} seconds")

print("=" * 60)
```

**Challenge:** Which corner has the longest braking zone? Can you explain why based on the (v₁² - v₂²) term?

---

### **Step 5: Visualize Corner Entry (Simplified Version)**

Let's create a simple visualization first, then build complexity:

```python
# Simple braking zone visualization
def plot_simple_braking_zone(v_straight, v_corner, mu):
    """Simple visualization of braking zone."""
    
    # Calculate parameters
    d, t = braking_distance(v_straight, v_corner, mu)
    v1 = v_straight / 3.6
    v2 = v_corner / 3.6
    a_max = mu * g
    
    # Create arrays
    x = np.linspace(0, d, 200)
    v_x = np.sqrt(v1**2 - 2 * a_max * x) * 3.6
    
    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(x, v_x, linewidth=3, color='#e10600')
    plt.fill_between(x, 0, v_x, alpha=0.2, color='red', label='Braking zone')
    
    # Mark key points
    plt.axvline(x=0, color='green', linewidth=2, label='Braking point (latest)')
    plt.axvline(x=d, color='blue', linewidth=2, label='Turn-in point')
    
    plt.xlabel('Distance to Corner (m)', fontsize=12)
    plt.ylabel('Velocity (km/h)', fontsize=12)
    plt.title(f'Braking Zone: {v_straight}→{v_corner} km/h | {d:.1f}m zone', 
              fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Test it!
plot_simple_braking_zone(280, 120, 2.0)
```

**Extension:** Try visualizing all three corners from the challenge above!

---

## **Stage 4: Weather Impact Analysis**

### **Learning Focus: Multi-Scenario Comparison and Decision-Making**
*Core Skills: Data analysis, comparative visualization, strategic thinking*

**Time Estimate:** 2-3 hours

**The Goal:** Build a comparison tool showing how weather dramatically changes braking strategy—the kind of analysis F1 teams do when rain starts mid-race.

---

### **Step 1: Define Track Corners**

Create a simplified F1 track with realistic braking zones:

```python
# Simplified track with 5 key braking zones
track_corners = {
    "Turn 1": (300, 80),    # Heavy braking from high speed
    "Turn 3": (260, 140),   # Medium-speed corner
    "Turn 6": (280, 100),   # Chicane
    "Turn 9": (310, 220),   # Fast corner (less braking)
    "Turn 12": (200, 60),   # Hairpin
}

# Weather conditions
conditions = {
    "Dry": 2.0,
    "Damp": 1.5,
    "Wet": 1.2,
    "Very Wet": 0.8
}
```

---

### **Step 2: Calculate and Compare (Simple Bar Chart First)**

Let's start with a clear, simple visualization before getting fancy:

```python
# Calculate dry vs wet for all corners
corner_names = list(track_corners.keys())
dry_distances = []
wet_distances = []

for corner_name, (v_in, v_out) in track_corners.items():
    d_dry, _ = braking_distance(v_in, v_out, 2.0)
    d_wet, _ = braking_distance(v_in, v_out, 1.2)
    dry_distances.append(d_dry)
    wet_distances.append(d_wet)

# Create comparison bar chart
x_pos = np.arange(len(corner_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x_pos - width/2, dry_distances, width, 
               label='Dry (μ=2.0)', color='#e10600', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, wet_distances, width, 
               label='Wet (μ=1.2)', color='#0066cc', alpha=0.8)

# Add values on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.0f}m', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Corner', fontsize=12)
ax.set_ylabel('Braking Distance (m)', fontsize=12)
ax.set_title('Dry vs Wet: Braking Distance Comparison', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(corner_names)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Print percentage increases
print("\n" + "=" * 60)
print("BRAKING DISTANCE INCREASE IN WET CONDITIONS")
print("=" * 60)
for i, corner in enumerate(corner_names):
    increase = wet_distances[i] - dry_distances[i]
    percent = (increase / dry_distances[i]) * 100
    print(f"{corner:15s} +{increase:5.1f}m  (+{percent:4.1f}%)")
print("=" * 60)
```

**What you should see:** Every corner needs ~65-67% more distance in wet conditions. Heavy braking zones show the largest absolute increases.

---

### **Step 3: Lap Time Impact**

Calculate the time penalty for one lap:

```python
# Calculate total braking time per lap
total_time_dry = sum([braking_distance(speeds[0], speeds[1], 2.0)[1] 
                      for speeds in track_corners.values()])
total_time_wet = sum([braking_distance(speeds[0], speeds[1], 1.2)[1] 
                      for speeds in track_corners.values()])

penalty_per_lap = total_time_wet - total_time_dry

print("\n" + "=" * 60)
print("LAP TIME IMPACT ANALYSIS")
print("=" * 60)
print(f"Total braking time (dry): {total_time_dry:.2f}s per lap")
print(f"Total braking time (wet): {total_time_wet:.2f}s per lap")
print(f"Time penalty in wet:      +{penalty_per_lap:.2f}s per lap")
print(f"\nOver 50-lap race:")
print(f"  Total penalty: +{penalty_per_lap * 50:.1f}s")
print(f"  That's {(penalty_per_lap * 50)/60:.1f} minutes slower!")
print("=" * 60)
```

**Engineering Insight:** This is why wet weather completely changes race strategy—braking alone can cost 2-3 minutes over a race distance!

---

### **Step 4: Advanced Track Visualization (Optional)**

Now that we understand the basics, here's the full overhead track view:

```python
def plot_track_overhead():
    """
    Overhead track view showing dry vs wet braking zones.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Simplified track positions
    corner_positions = [100, 300, 500, 750, 950]
    corner_labels = list(track_corners.keys())
    
    for ax, condition, mu, color, title_color in [
        (ax1, 'DRY', 2.0, '#e10600', 'black'),
        (ax2, 'WET', 1.2, '#0066cc', '#0066cc')
    ]:
        ax.set_title(f'{condition} CONDITIONS (μ={mu})', 
                     fontsize=14, fontweight='bold', color=title_color)
        ax.set_xlim(-50, 1050)
        ax.set_ylim(-20, 100)
        ax.set_xlabel('Track Position (m)', fontsize=12)
        ax.axhline(y=0, color='black', linewidth=3)
        
        for i, (corner_name, speeds) in enumerate(track_corners.items()):
            d, _ = braking_distance(speeds[0], speeds[1], mu)
            corner_pos = corner_positions[i]
            
            # Draw braking zone
            ax.add_patch(plt.Rectangle((corner_pos - d, -10), d, 20, 
                                       color=color, alpha=0.3))
            ax.plot([corner_pos - d, corner_pos], [0, 0], 
                    color=color, linewidth=4)
            ax.scatter([corner_pos], [0], color='blue', s=200, zorder=5)
            ax.text(corner_pos, 15, corner_labels[i], 
                    ha='center', fontsize=11, fontweight='bold')
        
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()

plot_track_overhead()
```

**What this shows:** How braking zones nearly double in the wet, completely changing the track's rhythm and braking markers.

---

### **Checkpoint #5**

**Challenge:** If it starts raining mid-race, which corners become the most dangerous (largest absolute distance increase)? Which corners are relatively "safer" (smallest change)?

**Extension:** Add intermediate conditions (damp, μ=1.5) and see how drivers must adjust as the track dries.

---

## **Stage 5: Interactive Braking Calculator**

### **Learning Focus: Interactive Engineering Tools**
*Core Skills: ipywidgets, real-time visualization, parameter exploration*

**Time Estimate:** 1-2 hours

**The Goal:** Create an interactive tool that updates in real-time as you adjust sliders—just like F1 simulation software.

---

### **Step 1: Import Widget Tools**

```python
from ipywidgets import interact, FloatSlider, IntSlider
```

---

### **Step 2: Create Interactive Function**

```python
def interactive_braking(v_entry_kmh=280, v_corner_kmh=120, mu=2.0):
    """
    Interactive braking calculator with real-time updates.
    """
    # Calculate parameters
    d_brake, t_brake = braking_distance(v_entry_kmh, v_corner_kmh, mu)
    a_max = mu * g
    
    # Create visualization
    v1 = v_entry_kmh / 3.6
    v2 = v_corner_kmh / 3.6
    x = np.linspace(0, d_brake, 200)
    v_x = np.sqrt(v1**2 - 2 * a_max * x) * 3.6
    
    # Two-panel plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # LEFT: Braking zone visualization
    ax1.plot(x, v_x, linewidth=3, color='#e10600')
    ax1.fill_between(x, 0, v_x, alpha=0.2, color='red')
    ax1.axvline(x=0, color='green', linewidth=2, label='Braking point')
    ax1.axvline(x=d_brake, color='blue', linewidth=2, label='Turn-in')
    ax1.set_xlabel('Distance to Corner (m)', fontsize=12)
    ax1.set_ylabel('Velocity (km/h)', fontsize=12)
    ax1.set_title(f'Braking Zone: {v_entry_kmh}→{v_corner_kmh} km/h', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, d_brake * 1.1)
    
    # RIGHT: Performance metrics
    ax2.axis('off')
    
    metrics_text = f"""
    ═══════════════════════════════════════
    F1 BRAKING CALCULATOR
    ═══════════════════════════════════════
    
    SPEEDS
    ───────────────────────────────────────
    Entry speed:        {v_entry_kmh} km/h
    Corner speed:       {v_corner_kmh} km/h
    Speed reduction:    {v_entry_kmh - v_corner_kmh} km/h
    
    TIRE PERFORMANCE
    ───────────────────────────────────────
    Grip coefficient:   μ = {mu:.2f}
    Max deceleration:   {a_max:.2f} m/s²
    G-force:            {a_max/g:.2f}g
    
    BRAKING ZONE
    ───────────────────────────────────────
    Distance required:  {d_brake:.1f} meters
    Time in braking:    {t_brake:.2f} seconds
    
    DRIVER EXPERIENCE
    ───────────────────────────────────────
    Force on 70kg driver: ~{a_max/g * 70:.0f} kg forward
    Precision needed:     ±0.5m braking point
    
    ═══════════════════════════════════════
    """
    
    ax2.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Comparison
    mu_road = 0.8
    d_road, _ = braking_distance(v_entry_kmh, v_corner_kmh, mu_road)
    advantage = ((d_road - d_brake) / d_road) * 100
    print(f"\n🏎️  F1 advantage over road car: {advantage:.1f}% shorter braking!")

# Create interactive widget
interact(interactive_braking,
         v_entry_kmh=IntSlider(min=100, max=350, value=280, step=10,
                               description='Entry Speed (km/h):',
                               style={'description_width': '160px'}),
         v_corner_kmh=IntSlider(min=50, max=250, value=120, step=10,
                                description='Corner Speed (km/h):',
                                style={'description_width': '160px'}),
         mu=FloatSlider(min=0.5, max=2.5, value=2.0, step=0.1,
                       description='Tire Grip (μ):',
                       style={'description_width': '160px'}))
```

---

### **Final Checkpoint**

**You should have:**
- Interactive sliders that update plots in real-time
- Performance metrics display
- Comparison to road car performance

**Challenge Scenarios to Try:**
1. **Monaco Hairpin:** v_entry=180, v_corner=50, μ=2.0
2. **Wet Monza T1:** v_entry=280, v_corner=120, μ=1.2
3. **Degraded Tires:** v_entry=280, v_corner=120, μ=1.6

**Can you find:** The minimum grip (μ) needed to brake from 300→100 km/h in under 150 meters?

---

## **Project Deliverables**

Submit your Colab notebook with:

### **1. Code (Well-Commented)**
- All stages completed and working
- Clear physics explanations in comments
- Meaningful variable names
- Function docstrings

### **2. Visualizations**
- Velocity vs time plot (Stage 1B)
- Velocity vs distance braking zone (Stage 2)
- Dry vs wet comparison (Stage 2 & 4)
- Corner entry visualization (Stage 3)
- Track overhead view (Stage 4, if attempted)
- Interactive tool screenshot (Stage 5, if attempted)

### **3. Written Responses**

**Physics Questions:**
1. Explain why a_max = μg is independent of car mass. What does this mean for F1 vs GT car braking?
2. The stopping distance formula contains v². If you double your speed from 150→300 km/h, how much does braking distance increase? Calculate specific values to support your answer.
3. Compare braking from 280→120 km/h in dry (μ=2.0) vs wet (μ=1.2). Calculate both distances and explain the physical reason for the difference.

**Strategic Analysis:**
4. **The Late Braking Battle:** Driver A brakes at 220m before Turn 1 (280→120 km/h, μ=2.0). Driver B brakes at the optimal point. Calculate:
   - The optimal braking distance
   - Driver A's speed at the optimal braking point
   - Driver B's advantage in km/h and meters
5. Using the 5-corner track from Stage 4, calculate total time spent braking per lap in:
   - Dry conditions (μ=2.0)
   - Wet conditions (μ=1.2)
   What's the lap time penalty from braking alone?

**Real-World Application:**
6. An F1 driver overshoots their braking point by 15 meters. They were approaching at 300 km/h for a corner requiring 100 km/h entry (μ=2.0). What speed will they carry into the corner? Will they make it?

**Reflection:**
7. What surprised you most about F1 braking physics?
8. How does this project change your understanding of what drivers must do lap after lap?
9. If you were designing a driver training program, how would you use this simulator?

---

## **Success Criteria**

### **Complete (Meets Expectations)**
- ✅ Stages 1A, 1B, 2, and 3 fully functional
- ✅ Code runs without errors
- ✅ Plots are clear with proper labels and units
- ✅ Analysis questions answered with calculations

### **Proficient (Exceeds Expectations)**
- ✅ All of above, plus:
- ✅ Code well-commented with physics explanations
- ✅ Professional visualizations with proper formatting
- ✅ Stage 4 comparison analysis completed
- ✅ Thoughtful responses showing physics understanding

### **Excellent (Outstanding)**
- ✅ All of above, plus:
- ✅ Stage 5 interactive tool working smoothly
- ✅ Additional visualizations or analysis
- ✅ Extensions attempted (tire degradation, etc.)
- ✅ Clear documentation demonstrating deep understanding

### **Exceptional (Goes Above and Beyond)**
- ✅ All of above, plus:
- ✅ Original extension (combined braking+cornering, downforce, etc.)
- ✅ Real F1 data comparison or validation
- ✅ Could serve as teaching resource

---

## **Tips for Success**

### **Before You Start**
- Review kinematic equations (v = v₀ - at, v² = v₀² - 2ad)
- Understand that μ represents tire grip
- Watch F1 onboard footage to see braking zones

### **While Working**
- Work sequentially through stages
- Test each function before moving forward
- Use print() to verify calculations
- Check units constantly (km/h ↔ m/s)

### **When Stuck**
1. Check unit conversions (km/h needs /3.6 to become m/s)
2. Verify equation signs (deceleration slows the car)
3. Print intermediate values to debug
4. Compare results to expected ranges

### **Best Practices**
- Comment physics formulas in code
- Use descriptive names (`v_entry` not `v1`)
- Label all plots with units
- Include reality checks (compare to road cars)

---

## **Extensions & Next Steps**

After completing this project, explore:

1. **Friction Circle:** Model combined braking + cornering
2. **Downforce:** Add speed-dependent grip (μ increases with v²)
3. **Brake Temperature:** Model heat buildup and fade
4. **Trail Braking:** Optimize brake release curve into corners
5. **Full Lap Sim:** String corners together with acceleration
6. **Tire Degradation:** Model grip loss over stint
7. **Race Strategy:** Optimize fuel vs lap time trade-offs

**You've got this! 🏎️💨**
