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

## **Stage 1: Basic Braking Calculator**

### **Learning Focus: Deceleration Dynamics and Friction Limits**
*Core Skills: Newton's 2nd law, friction coefficients, velocity-time calculations, basic plotting*

**Time Estimate:** 2-3 hours

**What You'll Build:** A program that calculates how far an F1 car travels while braking from a given speed to a standstill, considering tire grip limits.

**Assignment:**
1. Create a new Colab notebook titled "F1_Braking_Simulator"
2. Write code that calculates the stopping distance and time for an F1 car braking from 200 km/h (a typical corner entry speed)

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
  - Notice: mass cancels out! Deceleration depends only on tire grip, not car weight

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

**Step 2: Define Physical Constants and Initial Conditions**

Set up the parameters for an F1 car braking scenario. Think about what values make sense:

```python
# Physical constants
g = 9.81  # gravitational acceleration (m/s²)

# F1 car parameters - fill in reasonable values:
mu = ???  # coefficient of friction - F1 slick tires (try 1.8-2.2)
m = ???   # car mass (kg) - F1 car minimum weight is 798 kg
v0_kmh = ???  # initial speed (km/h) - typical corner approach is 150-250 km/h
<!-- PARTIAL_REVEAL -->
# Complete solution:
g = 9.81   # gravitational acceleration (m/s²)
mu = 2.0   # coefficient of friction - modern F1 slick tires
m = 798    # car mass (kg) - 2024 F1 minimum weight
v0_kmh = 200  # initial speed (km/h) - approaching a medium-speed corner

# **Why these values?**
# - `mu = 2.0`: F1 slick tires in optimal conditions provide 1.8-2.2g of grip
# - `m = 798 kg`: FIA minimum weight (car + driver) as of 2024
# - `v0_kmh = 200 km/h`: Realistic approach speed for turns like Monza's first chicane
```

**Key Concept:** Unlike the projectile problem where we tracked position over time, here we're focused on **deceleration**—how quickly the car can slow down within tire grip limits.

---

**Step 3: Convert Speed and Calculate Maximum Deceleration**

Convert km/h to m/s (standard physics units) and calculate the maximum deceleration:

```python
# Convert speed to m/s (physics standard unit)
v0 = v0_kmh / 3.6  # Divide by 3.6 to convert km/h → m/s

# Fill in the maximum deceleration formula:
a_max = ??? * ???  # Maximum deceleration = mu × g
<!-- PARTIAL_REVEAL -->
# Complete solution:
v0 = v0_kmh / 3.6  # Convert: 200 km/h = 55.56 m/s
a_max = mu * g     # Maximum deceleration = 2.0 × 9.81 = 19.62 m/s²

# **The physics:**
# - We use a_max (not a) because the driver is at the tire grip limit
# - This is about 2.0g—F1 drivers experience twice Earth's gravity pushing them forward!
# - Notice: mass (m) canceled out in the formula a_max = (μmg)/m = μg
```

**Engineering Insight:** The fact that deceleration is independent of mass means a heavy F1 car and a light one stop in the same distance (assuming same tires). However, heavier cars have more kinetic energy, so their brakes and tires work harder!

---

**Step 4: Calculate Stopping Time and Distance**

Now use kinematic equations to find how long it takes to stop and how far the car travels:

```python
# Fill in the kinematic formulas:
t_stop = ??? / ???           # Time to stop: v = v0 - at, solve for t when v=0
d_stop = ??? / (2 * ???)     # Distance to stop: v² = v0² - 2ad, solve for d when v=0
<!-- PARTIAL_REVEAL -->
# Complete solution:
t_stop = v0 / a_max              # Time to stop: t = v0 / a
d_stop = v0**2 / (2 * a_max)     # Distance to stop: d = v0² / (2a)

# **The kinematic equations:**
# - Starting from v = v0 - at, when the car stops (v=0): t = v0/a
# - Starting from v² = v0² - 2ad, when the car stops (v=0): d = v0²/(2a)
# - These assume constant deceleration (maximum braking throughout)
```

**Physics reminder:**
- **v = v₀ - at** (velocity during constant deceleration)
- **v² = v₀² - 2ad** (velocity-displacement relationship)
- We use the **negative** sign because deceleration opposes motion

---

**Step 5: Print Results**

```python
# Display the results with proper units and formatting
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
An F1 car braking from 200 km/h to 0 in under 3 seconds over less than 80 meters demonstrates the incredible grip of F1 slick tires. For comparison, a typical road car would need about 130-150 meters!

---

**Step 6: Visualize Velocity Over Time**

Create a plot showing how velocity decreases during braking:

```python
# Create time array from start to stop
t = np.linspace(0, t_stop, 100)

# Fill in the velocity equation:
v = ??? - ??? * t  # v(t) = v0 - at
<!-- PARTIAL_REVEAL -->
# Complete solution:
t = np.linspace(0, t_stop, 100)
v = v0 - a_max * t  # Velocity as a function of time
```

Now plot it:

```python
plt.figure(figsize=(10, 5))

# Fill in the plotting commands:
plt.???(t, v * 3.6, linewidth=2, color='#e10600')  # Plot time vs velocity (convert to km/h)
plt.???('Time (s)', fontsize=12)
plt.???('Velocity (km/h)', fontsize=12)
plt.???('F1 Braking: Velocity vs Time', fontsize=14, fontweight='bold')
plt.???(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)  # Ground reference
plt.tight_layout()
plt.show()
<!-- PARTIAL_REVEAL -->
# Complete solution:
plt.figure(figsize=(10, 5))
plt.plot(t, v * 3.6, linewidth=2, color='#e10600')  # F1 red color
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Velocity (km/h)', fontsize=12)
plt.title('F1 Braking: Velocity vs Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

**What you should see:** A straight line declining from 200 km/h to 0—this represents constant maximum braking (constant deceleration = linear velocity decrease).

---

### **Common Issues & Debugging**

**Problem:** "NameError: name 'np' is not defined"
- **Solution:** Run the import cell first

**Problem:** Stopping distance seems too short or too long
- **Solution:** Check that you converted km/h to m/s (divide by 3.6, not 3.0)

**Problem:** Plot shows velocity in m/s instead of km/h
- **Solution:** Multiply velocity by 3.6 when plotting: `v * 3.6`

**Debugging tip:** Add print statements to check intermediate values:
```python
print(f"v0 in m/s: {v0}")
print(f"a_max: {a_max}")
```

---

### **Extension Challenge (Optional)**

**Experiment with different scenarios:**
1. Change `v0_kmh` to 300 km/h (Monaco tunnel exit) and see how stopping distance changes
2. Reduce `mu` to 0.8 (wet conditions) and observe the impact
3. Calculate the percentage increase in stopping distance from dry to wet conditions

**Reflection Questions:**
- Why does an F1 car stop in much less distance than a road car at the same speed?
- How does doubling the initial speed affect stopping distance? (Hint: it's not 2x!)
- What happens to braking performance if tire temperature drops and grip (μ) decreases?

---

## **Stage 2: Braking Zone Visualization**

### **Learning Focus: Distance-Velocity Relationship and Visual Analysis**
*Core Skills: Position-velocity curves, numpy arrays, multi-plot visualization, engineering analysis*

**Time Estimate:** 2-3 hours

**What You'll Build:** A program that plots the "braking zone"—showing both how velocity and position change during maximum braking, plus comparing multiple scenarios.

**Assignment:** Create visualizations that show:
1. Velocity vs distance (showing where the car is at each speed)
2. Comparison of braking from different initial speeds
3. Side-by-side dry vs wet conditions

---

### **Step-by-Step Instructions:**

**Step 1: Calculate Position During Braking**

We need position as a function of time to see where the car is throughout the braking zone:

```python
# Using our time array from Stage 1
t = np.linspace(0, t_stop, 100)
v_t = v0 - a_max * t  # Velocity over time

# Fill in the position equation:
# Position: x(t) = v0*t - 0.5*a*t²  (same form as projectile vertical motion!)
x_t = ??? * t - 0.5 * ??? * t**2
<!-- PARTIAL_REVEAL -->
# Complete solution:
t = np.linspace(0, t_stop, 100)
v_t = v0 - a_max * t
x_t = v0 * t - 0.5 * a_max * t**2  # Position increases as car travels forward

# **Connection to projectiles:**
# Same kinematic form: x = v0*t - 0.5*a*t²
# But here 'a' is braking deceleration (slowing horizontal motion)
# vs projectile where 'g' was pulling downward
```

---

**Step 2: Create Two-Panel Visualization**

Create a professional multi-panel plot showing velocity and position:

```python
# Create figure with 2 subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# LEFT PANEL: Velocity vs Time
ax1.plot(t, v_t * 3.6, linewidth=2, color='#e10600', label='Velocity')
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Velocity (km/h)', fontsize=12)
ax1.set_title('Velocity During Braking', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax1.legend()

# RIGHT PANEL: Position vs Time
ax2.plot(t, x_t, linewidth=2, color='#0600e1', label='Position')
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Distance (m)', fontsize=12)
ax2.set_title('Distance Covered During Braking', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=d_stop, color='red', linestyle='--', alpha=0.5, label=f'Stop: {d_stop:.1f}m')
ax2.legend()

plt.tight_layout()
plt.show()
```

**What you should see:** 
- Left: Linear velocity decrease (straight line down)
- Right: Curved position increase (parabola leveling off at stopping distance)

---

**Step 3: The Critical Plot - Velocity vs Distance**

This is THE plot F1 engineers use—it shows "at what speed is the car at each point in the braking zone?"

```python
# Calculate velocity as a function of distance (not time)
# We can derive this from v² = v0² - 2*a*x, so v = sqrt(v0² - 2*a*x)

# Create distance array
x_array = np.linspace(0, d_stop, 100)

# Fill in the velocity-distance equation:
v_x = np.sqrt(???**2 - 2 * ??? * x_array)  # v(x) = sqrt(v0² - 2ax)
<!-- PARTIAL_REVEAL -->
# Complete solution:
x_array = np.linspace(0, d_stop, 100)
v_x = np.sqrt(v0**2 - 2 * a_max * x_array)

# **The physics:**
# Starting from v² = v0² - 2ad
# Solve for v: v = sqrt(v0² - 2ad)
# This gives velocity as a function of distance traveled
```

Now plot it:

```python
plt.figure(figsize=(10, 6))
plt.plot(x_array, v_x * 3.6, linewidth=3, color='#e10600', label='Velocity')
plt.xlabel('Distance from Braking Point (m)', fontsize=12)
plt.ylabel('Velocity (km/h)', fontsize=12)
plt.title('F1 Braking Zone: Velocity vs Distance', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add annotations
plt.axhline(y=v0_kmh, color='green', linestyle='--', alpha=0.5, label=f'Entry: {v0_kmh} km/h')
plt.axvline(x=d_stop, color='red', linestyle='--', alpha=0.5, label=f'Stop: {d_stop:.1f}m')

plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
```

**Engineering Insight:** This plot answers "If I'm 50 meters from the corner, what speed should I be at?" It's the foundation for finding optimal braking points!

---

### **Checkpoint #2**

**What you should see:**
- A curved line (not straight!) decreasing from 200 km/h to 0
- The curve is steeper at first (rapid deceleration) then flattens
- X-axis shows the 78.6m braking zone
- Annotations marking entry speed and stopping point

**Verification:**
```python
# Check velocity at halfway point
halfway = d_stop / 2
v_halfway = np.sqrt(v0**2 - 2 * a_max * halfway)
print(f"At {halfway:.1f}m (halfway), velocity is {v_halfway*3.6:.1f} km/h")

# Expected: At 39.3m, velocity should be ~141 km/h (not 100 km/h!)
# This shows velocity doesn't decrease linearly with distance
```

---

**Step 4: Compare Multiple Braking Scenarios**

Create a function to calculate braking and compare different cases:

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
    v0 = v0_kmh / 3.6
    a_max = mu * g
    t_stop = v0 / a_max
    d_stop = v0**2 / (2 * a_max)
    
    x_array = np.linspace(0, d_stop, 100)
    v_array = np.sqrt(v0**2 - 2 * a_max * x_array) * 3.6  # Convert to km/h
    
    return x_array, v_array, d_stop, t_stop
```

Now compare scenarios:

```python
# Define scenarios
scenarios = [
    (200, 2.0, 'Dry (μ=2.0)', '#e10600'),     # Standard F1 grip
    (200, 1.2, 'Wet (μ=1.2)', '#0066cc'),     # Wet weather
    (200, 0.8, 'Very Wet (μ=0.8)', '#00cc66') # Heavy rain
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
- Dry conditions: ~78m stopping distance
- Wet conditions: ~130m stopping distance (65% longer!)
- Very wet: ~196m stopping distance (2.5x longer!)

**Real-world application:** This is why F1 drivers brake much earlier in the rain—tire grip (μ) drops dramatically, requiring longer braking zones.

---

### **Extension Challenge (Optional)**

**Compare different entry speeds:**
```python
speeds = [150, 200, 250, 300]  # km/h - different corner types
colors = ['#00cc66', '#e10600', '#ff9900', '#cc00cc']

plt.figure(figsize=(12, 6))

for speed, color in zip(speeds, colors):
    x, v, d_stop, t_stop = calculate_braking(speed, 2.0, f'{speed} km/h', color)
    plt.plot(x, v, linewidth=2.5, label=f'{speed} km/h → {d_stop:.1f}m', color=color)

plt.xlabel('Distance (m)', fontsize=12)
plt.ylabel('Velocity (km/h)', fontsize=12)
plt.title('Braking Distance vs Entry Speed (μ=2.0)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
```

**Analysis Question:** How does stopping distance scale with initial velocity? Try v0 = 100, 200, 300 km/h and note the pattern. (Hint: d ∝ v₀²)

---

## **Stage 3: Optimal Braking Point Calculator**

### **Learning Focus: Optimization and Corner Entry Strategy**
*Core Skills: Functions, optimization logic, trade-off analysis, strategic thinking*

**Time Estimate:** 2-3 hours

**The Challenge:** In F1, you don't just brake to a stop—you brake to a **target corner entry speed**. The question is: **How late can you brake and still hit that speed?**

**What You'll Build:** A calculator that finds the optimal braking point given:
- Current speed (e.g., 280 km/h on straight)
- Target corner entry speed (e.g., 120 km/h for a medium corner)
- Available grip (tire condition)

This is the fundamental "late braking" problem in racing!

---

### **Step-by-Step Instructions:**

**Step 1: Understand the Problem**

**Scenario:** You're approaching Turn 1 at Monza:
- **Straight speed:** 280 km/h
- **Corner entry speed:** 120 km/h (any faster and you'll run wide)
- **Question:** How far before the corner do you need to start braking?

**The physics:** We need to find the distance required to decelerate from v₁ to v₂.

From the kinematic equation: **v₂² = v₁² - 2ad**

Solving for distance: **d = (v₁² - v₂²) / (2a)**

---

**Step 2: Create Braking Distance Function**

```python
def braking_distance(v_entry_kmh, v_corner_kmh, mu):
    """
    Calculate minimum braking distance from entry speed to corner speed.
    
    Parameters:
    - v_entry_kmh: Speed at braking point (km/h)
    - v_corner_kmh: Target corner entry speed (km/h)
    - mu: Coefficient of friction
    
    Returns:
    - distance: Required braking distance (meters)
    - time: Time spent braking (seconds)
    """
    # Fill in the calculations:
    v_entry = ??? / 3.6       # Convert to m/s
    v_corner = ??? / 3.6      # Convert to m/s
    a_max = ??? * g           # Maximum deceleration
    
    # Distance required: d = (v1² - v2²) / (2a)
    distance = (???**2 - ???**2) / (2 * ???)
    
    # Time required: t = (v1 - v2) / a
    time = (??? - ???) / ???
    
    return distance, time
<!-- PARTIAL_REVEAL -->
# Complete solution:
def braking_distance(v_entry_kmh, v_corner_kmh, mu):
    """
    Calculate minimum braking distance from entry speed to corner speed.
    
    Parameters:
    - v_entry_kmh: Speed at braking point (km/h)
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

**Step 3: Test the Function**

```python
# Monza Turn 1 scenario
v_straight = 280  # km/h - top speed on main straight
v_corner = 120    # km/h - maximum safe corner entry speed
mu_dry = 2.0      # dry conditions

d_brake, t_brake = braking_distance(v_straight, v_corner, mu_dry)

print("=" * 60)
print("MONZA TURN 1 BRAKING ANALYSIS")
print("=" * 60)
print(f"Straight speed: {v_straight} km/h")
print(f"Corner entry speed: {v_corner} km/h")
print(f"Speed reduction: {v_straight - v_corner} km/h")
print(f"Tire grip: μ = {mu_dry}")
print("-" * 60)
print(f"BRAKING DISTANCE: {d_brake:.1f} meters")
print(f"BRAKING TIME: {t_brake:.2f} seconds")
print("=" * 60)
```

**Expected output:**
- Braking distance: ~211.6 meters
- Braking time: ~2.27 seconds

**Engineering Insight:** F1 drivers must hit this braking point within ±1 meter and ±0.01 seconds lap after lap. That's the precision required at 280 km/h!

---

**Step 4: Visualize the Corner Entry Scenario**

Create a plot showing the complete braking zone from straight to corner:

```python
def plot_corner_entry(v_straight, v_corner, mu, corner_name="Turn 1"):
    """
    Visualize complete braking zone for corner entry.
    """
    # Calculate braking parameters
    v1 = v_straight / 3.6
    v2 = v_corner / 3.6
    a_max = mu * g
    d_brake = (v1**2 - v2**2) / (2 * a_max)
    
    # Create distance array
    x = np.linspace(0, d_brake, 200)
    
    # Velocity at each point: v² = v1² - 2ax, so v = sqrt(v1² - 2ax)
    # But we need to account for final velocity v2, not zero
    # So: v² = v1² - 2ax, rearranging from v2² = v1² - 2a*d_brake
    v_x = np.sqrt(v1**2 - 2 * a_max * x) * 3.6  # Convert to km/h
    
    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(x, v_x, linewidth=3, color='#e10600', label='Velocity during braking')
    
    # Mark key points
    plt.axhline(y=v_straight, color='green', linestyle='--', alpha=0.6, 
                label=f'Straight speed: {v_straight} km/h')
    plt.axhline(y=v_corner, color='blue', linestyle='--', alpha=0.6, 
                label=f'Corner entry: {v_corner} km/h')
    plt.axvline(x=0, color='green', linestyle='-', linewidth=2, alpha=0.6, 
                label='Braking point (latest possible)')
    plt.axvline(x=d_brake, color='blue', linestyle='-', linewidth=2, alpha=0.6, 
                label='Corner turn-in point')
    
    # Add shaded braking zone
    plt.fill_between(x, 0, v_x, alpha=0.2, color='red', label='Braking zone')
    
    plt.xlabel('Distance to Corner (m)', fontsize=13)
    plt.ylabel('Velocity (km/h)', fontsize=13)
    plt.title(f'{corner_name} Braking Zone | {v_straight}→{v_corner} km/h | μ={mu}', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper right')
    
    # Add text annotation
    mid_point = d_brake / 2
    mid_velocity = np.sqrt(v1**2 - 2 * a_max * mid_point) * 3.6
    plt.annotate(f'{d_brake:.1f}m braking zone', 
                 xy=(mid_point, mid_velocity), 
                 xytext=(mid_point, mid_velocity + 30),
                 fontsize=11, ha='center',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    plt.tight_layout()
    plt.show()

# Test it!
plot_corner_entry(280, 120, 2.0, "Monza Turn 1")
```

**What you should see:**
- Red shaded area shows the braking zone
- Green vertical line marks where braking starts (latest possible point)
- Blue vertical line marks corner entry
- Velocity curve from 280 → 120 km/h

---

### **Checkpoint #3**

**What you should understand:**
1. The braking point is determined by: target corner speed, current speed, and grip level
2. Braking "late" means starting as close to the corner as physics allows
3. If you brake too late, you'll enter the corner too fast and run wide (or crash!)

**Verification:**
```python
# Test with different scenarios
scenarios = [
    ("Monza T1", 280, 120, 2.0),
    ("Monaco Hairpin", 180, 50, 2.0),
    ("Silverstone Copse", 310, 200, 2.0)  # High-speed corner
]

for name, v_in, v_out, mu in scenarios:
    d, t = braking_distance(v_in, v_out, mu)
    print(f"{name}: {v_in}→{v_out} km/h | Distance: {d:.1f}m | Time: {t:.2f}s")
```

**Expected patterns:**
- Larger speed reduction = longer braking zone
- High-speed corners need longer distances even with smaller speed reduction (quadratic relationship!)

---

**Step 5: The Late Braking Advantage**

Compare two drivers: one who brakes early (safe) vs one who brakes late (optimal):

```python
# Scenario: Both drivers reach same corner speed, but start braking at different points
v_straight = 280
v_corner = 120
mu = 2.0

# Calculate optimal (latest) braking point
d_optimal, t_optimal = braking_distance(v_straight, v_corner, mu)

# Driver A: Brakes 20m early (conservative)
d_early = d_optimal + 20

# Calculate what speed Driver A has at the optimal braking point
# They've already been braking for 20m at that point
# v² = v_straight² - 2*a*20
a_max = mu * g
v_early_at_optimal = np.sqrt((v_straight/3.6)**2 - 2 * a_max * 20) * 3.6

print("=" * 60)
print("LATE BRAKING ADVANTAGE ANALYSIS")
print("=" * 60)
print(f"Driver B (optimal): Brakes at {d_optimal:.1f}m mark")
print(f"Driver A (conservative): Brakes at {d_early:.1f}m mark")
print("-" * 60)
print(f"At the optimal braking point ({d_optimal:.1f}m):")
print(f"  Driver B: {v_straight} km/h (full speed)")
print(f"  Driver A: {v_early_at_optimal:.1f} km/h (already braking)")
print(f"  Speed advantage to B: {v_straight - v_early_at_optimal:.1f} km/h")
print("-" * 60)
print(f"Time advantage for Driver B: ~{20 / (v_straight/3.6):.3f} seconds")
print("=" * 60)
```

**Engineering Insight:** Braking just 20m earlier means carrying ~26 km/h less speed at the optimal point. Over a lap with 10 braking zones, this adds up to significant lap time!

---

### **Extension Challenge (Optional)**

**Create a braking point calculator that considers safety margin:**
```python
def braking_point_with_margin(v_straight, v_corner, mu, safety_margin_m=5):
    """
    Calculate braking point with safety margin.
    
    safety_margin_m: Extra distance added for safety (meters)
    """
    d_optimal, t_optimal = braking_distance(v_straight, v_corner, mu)
    d_safe = d_optimal + safety_margin_m
    
    print(f"Optimal braking: {d_optimal:.1f}m before corner")
    print(f"With {safety_margin_m}m safety margin: {d_safe:.1f}m before corner")
    print(f"Time cost of safety margin: {safety_margin_m / (v_straight/3.6):.3f}s")
    
    return d_safe

# Test different safety margins
for margin in [0, 5, 10, 20]:
    print(f"\n--- {margin}m margin ---")
    braking_point_with_margin(280, 120, 2.0, margin)
```

**Reflection Questions:**
- Why do F1 drivers practice hitting the exact same braking point thousands of times?
- How does wet weather change the optimal braking point? (Try μ = 1.2 vs μ = 2.0)
- What happens if you brake too late and miss the corner entry speed?

---

## **Stage 4: Comparison Tool - Dry vs Wet Racing**

### **Learning Focus: Multi-Scenario Analysis and Performance Trade-offs**
*Core Skills: Comparative analysis, data presentation, engineering decision-making*

**Time Estimate:** 2-3 hours

**The Goal:** Build a comparison tool that shows how weather conditions dramatically change braking strategy. This mirrors real F1 decision-making when rain starts during a race.

---

### **Step 1: Define Track Scenarios**

Create realistic corner data for a simplified track:

```python
# Simplified F1 track with 5 key braking zones
track_corners = {
    "Turn 1 (Heavy Braking)": {"v_entry": 300, "v_corner": 80},
    "Turn 3 (Medium)": {"v_entry": 260, "v_corner": 140},
    "Turn 6 (Chicane)": {"v_entry": 280, "v_corner": 100},
    "Turn 9 (Fast Corner)": {"v_entry": 310, "v_corner": 220},
    "Turn 12 (Hairpin)": {"v_entry": 200, "v_corner": 60}
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

**Step 2: Calculate All Braking Zones**

```python
import pandas as pd  # For nice table display

# Calculate braking distances for all combinations
results = []

for corner_name, speeds in track_corners.items():
    for condition_name, mu in conditions.items():
        d, t = braking_distance(speeds["v_entry"], speeds["v_corner"], mu)
        results.append({
            "Corner": corner_name,
            "Condition": condition_name,
            "Entry Speed": speeds["v_entry"],
            "Corner Speed": speeds["v_corner"],
            "μ": mu,
            "Braking Distance (m)": round(d, 1),
            "Braking Time (s)": round(t, 2)
        })

# Create DataFrame for easy viewing
df = pd.DataFrame(results)

# Display results for Turn 1
print("=" * 80)
print("TURN 1 BRAKING ANALYSIS ACROSS CONDITIONS")
print("=" * 80)
turn1_data = df[df["Corner"] == "Turn 1 (Heavy Braking)"]
print(turn1_data.to_string(index=False))
print("=" * 80)
```

---

**Step 3: Visualize Braking Distance Increase**

```python
# Compare dry vs wet for all corners
corner_names = list(track_corners.keys())
dry_distances = []
wet_distances = []

for corner_name, speeds in track_corners.items():
    d_dry, _ = braking_distance(speeds["v_entry"], speeds["v_corner"], 2.0)
    d_wet, _ = braking_distance(speeds["v_entry"], speeds["v_corner"], 1.2)
    dry_distances.append(d_dry)
    wet_distances.append(d_wet)

# Create bar chart comparison
x = np.arange(len(corner_names))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))
bars1 = ax.bar(x - width/2, dry_distances, width, label='Dry (μ=2.0)', color='#e10600', alpha=0.8)
bars2 = ax.bar(x + width/2, wet_distances, width, label='Wet (μ=1.2)', color='#0066cc', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}m',
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Corner', fontsize=12)
ax.set_ylabel('Braking Distance (m)', fontsize=12)
ax.set_title('Braking Distance Comparison: Dry vs Wet Conditions', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([name.split('(')[0].strip() for name in corner_names], rotation=15, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Calculate percentage increases
print("\n" + "=" * 60)
print("BRAKING DISTANCE INCREASE IN WET CONDITIONS")
print("=" * 60)
for i, corner in enumerate(corner_names):
    increase = wet_distances[i] - dry_distances[i]
    percent = (increase / dry_distances[i]) * 100
    print(f"{corner:30s} +{increase:5.1f}m  (+{percent:4.1f}%)")
print("=" * 60)
```

**What you should observe:**
- All corners require ~65-67% more braking distance in wet conditions
- Heavy braking zones show the largest absolute distance increase
- Even "fast corners" need significantly earlier braking

---

**Step 4: Track Map Visualization with Braking Zones**

Create a simplified overhead view showing braking zones:

```python
def plot_track_with_braking_zones():
    """
    Create simplified overhead track view showing dry vs wet braking zones.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Simplified track coordinates (just for visualization)
    # Each corner gets a position on a simplified circuit
    corner_positions = [100, 300, 500, 750, 950]  # Distance along track
    
    corner_names_short = ["T1", "T3", "T6", "T9", "T12"]
    
    # DRY CONDITIONS
    ax1.set_title("DRY CONDITIONS (μ=2.0)", fontsize=14, fontweight='bold')
    ax1.set_xlim(-50, 1050)
    ax1.set_ylim(-20, 100)
    ax1.set_xlabel("Track Position (m)", fontsize=12)
    ax1.axhline(y=0, color='black', linewidth=3, label='Track centerline')
    
    for i, (corner_name, speeds) in enumerate(track_corners.items()):
        d_dry, _ = braking_distance(speeds["v_entry"], speeds["v_corner"], 2.0)
        corner_pos = corner_positions[i]
        
        # Draw braking zone
        ax1.add_patch(plt.Rectangle((corner_pos - d_dry, -10), d_dry, 20, 
                                     color='#e10600', alpha=0.3))
        ax1.plot([corner_pos - d_dry, corner_pos], [0, 0], 
                 color='#e10600', linewidth=4, label=f'{corner_names_short[i]}: {d_dry:.0f}m' if i < 3 else '')
        ax1.scatter([corner_pos], [0], color='blue', s=200, zorder=5)
        ax1.text(corner_pos, 15, corner_names_short[i], ha='center', fontsize=11, fontweight='bold')
    
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_yticks([])
    ax1.grid(True, alpha=0.3, axis='x')
    
    # WET CONDITIONS
    ax2.set_title("WET CONDITIONS (μ=1.2)", fontsize=14, fontweight='bold', color='#0066cc')
    ax2.set_xlim(-50, 1050)
    ax2.set_ylim(-20, 100)
    ax2.set_xlabel("Track Position (m)", fontsize=12)
    ax2.axhline(y=0, color='black', linewidth=3)
    
    for i, (corner_name, speeds) in enumerate(track_corners.items()):
        d_wet, _ = braking_distance(speeds["v_entry"], speeds["v_corner"], 1.2)
        corner_pos = corner_positions[i]
        
        # Draw braking zone
        ax2.add_patch(plt.Rectangle((corner_pos - d_wet, -10), d_wet, 20, 
                                     color='#0066cc', alpha=0.3))
        ax2.plot([corner_pos - d_wet, corner_pos], [0, 0], 
                 color='#0066cc', linewidth=4, label=f'{corner_names_short[i]}: {d_wet:.0f}m' if i < 3 else '')
        ax2.scatter([corner_pos], [0], color='blue', s=200, zorder=5)
        ax2.text(corner_pos, 15, corner_names_short[i], ha='center', fontsize=11, fontweight='bold')
    
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()

plot_track_with_braking_zones()
```

**Engineering Insight:** This overhead view shows why wet weather racing is so challenging—braking zones nearly double, changing the entire rhythm of the track. Braking markers that work in the dry are useless in the wet!

---

### **Checkpoint #4**

**What you should have:**
1. Table showing braking distances for all corners in all conditions
2. Bar chart comparing dry vs wet braking distances
3. Track visualization showing how braking zones expand

**Analysis Exercise:**
Calculate the total time difference for one lap:
```python
# Sum all braking times for dry vs wet
total_time_dry = sum([braking_distance(speeds["v_entry"], speeds["v_corner"], 2.0)[1] 
                      for speeds in track_corners.values()])
total_time_wet = sum([braking_distance(speeds["v_entry"], speeds["v_corner"], 1.2)[1] 
                      for speeds in track_corners.values()])

time_penalty = total_time_wet - total_time_dry

print(f"Total braking time (dry): {total_time_dry:.2f}s")
print(f"Total braking time (wet): {total_time_wet:.2f}s")
print(f"Time penalty in wet: +{time_penalty:.2f}s per lap")
print(f"Over 50-lap race: +{time_penalty * 50:.1f}s total ({(time_penalty * 50)/60:.1f} minutes)")
```

---

### **Extension Challenge (Optional)**

**Tire degradation model:**
As tires wear, grip decreases. Model how braking distances increase over a race stint:

```python
def tire_degradation_analysis():
    """
    Show how braking performance degrades as tires wear.
    """
    # Grip coefficient over tire life (laps)
    laps = np.arange(0, 31, 5)  # 0 to 30 laps, check every 5 laps
    mu_values = 2.0 - 0.012 * laps  # Grip decreases ~0.012 per lap
    
    # Test corner: Turn 1
    v_entry = 300
    v_corner = 80
    
    distances = []
    for mu in mu_values:
        d, _ = braking_distance(v_entry, v_corner, mu)
        distances.append(d)
    
    plt.figure(figsize=(10, 6))
    plt.plot(laps, distances, linewidth=3, marker='o', markersize=8, color='#e10600')
    plt.xlabel('Lap Number', fontsize=12)
    plt.ylabel('Braking Distance (m)', fontsize=12)
    plt.title('Turn 1 Braking Distance vs Tire Degradation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Annotate fresh vs worn
    plt.annotate(f'Fresh tires\n{distances[0]:.1f}m', 
                 xy=(laps[0], distances[0]), xytext=(5, distances[0]-10),
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', color='black'))
    plt.annotate(f'Worn tires\n{distances[-1]:.1f}m', 
                 xy=(laps[-1], distances[-1]), xytext=(laps[-1]-5, distances[-1]+10),
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    plt.show()
    
    print(f"Braking distance increase over 30 laps: +{distances[-1] - distances[0]:.1f}m")

tire_degradation_analysis()
```

---

## **Stage 5: Creating an Interactive Braking Calculator (Advanced)**

### **Learning Focus: Interactive Tools for Engineering Analysis**
*Core Skills: ipywidgets, user interfaces, parameter exploration, real-time feedback*

**Time Estimate:** 1-2 hours

**The Goal:** Create an interactive tool with sliders that lets you explore braking scenarios in real-time—just like F1 engineers use simulation tools to analyze different strategies.

---

### **Step 1: Import Widget Tools**

```python
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown
```

---

### **Step 2: Create Interactive Braking Function**

```python
def interactive_braking(v_entry_kmh=280, v_corner_kmh=120, mu=2.0, show_zones=True):
    """
    Interactive braking zone calculator and visualizer.
    
    Adjustable parameters:
    - v_entry_kmh: Entry speed (100-350 km/h)
    - v_corner_kmh: Target corner speed (50-250 km/h)
    - mu: Tire grip coefficient (0.5-2.5)
    - show_zones: Toggle braking zone visualization
    """
    # Calculate braking parameters
    d_brake, t_brake = braking_distance(v_entry_kmh, v_corner_kmh, mu)
    
    # Calculate deceleration
    a_max = mu * g
    
    # Create visualization
    v1 = v_entry_kmh / 3.6
    v2 = v_corner_kmh / 3.6
    
    x = np.linspace(0, d_brake, 200)
    v_x = np.sqrt(v1**2 - 2 * a_max * x) * 3.6
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # LEFT: Velocity vs Distance
    ax1.plot(x, v_x, linewidth=3, color='#e10600', label='Velocity profile')
    ax1.axhline(y=v_entry_kmh, color='green', linestyle='--', alpha=0.6, label=f'Entry: {v_entry_kmh} km/h')
    ax1.axhline(y=v_corner_kmh, color='blue', linestyle='--', alpha=0.6, label=f'Corner: {v_corner_kmh} km/h')
    
    if show_zones:
        ax1.fill_between(x, 0, v_x, alpha=0.2, color='red', label='Braking zone')
    
    ax1.set_xlabel('Distance to Corner (m)', fontsize=12)
    ax1.set_ylabel('Velocity (km/h)', fontsize=12)
    ax1.set_title(f'Braking Zone: {v_entry_kmh}→{v_corner_kmh} km/h', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, d_brake * 1.1)
    
    # RIGHT: Performance metrics
    ax2.axis('off')
    
    # Create info box
    info_text = f"""
    BRAKING ANALYSIS
    ═══════════════════════════════════
    
    Entry Speed:        {v_entry_kmh} km/h ({v1:.1f} m/s)
    Corner Speed:       {v_corner_kmh} km/h ({v2:.1f} m/s)
    Speed Reduction:    {v_entry_kmh - v_corner_kmh} km/h
    
    Tire Grip (μ):      {mu:.2f}
    Max Deceleration:   {a_max:.2f} m/s² ({a_max/g:.2f}g)
    
    ───────────────────────────────────
    
    BRAKING DISTANCE:   {d_brake:.1f} meters
    BRAKING TIME:       {t_brake:.2f} seconds
    
    ───────────────────────────────────
    
    G-Force:            {a_max/g:.2f}g forward
    Driver feels:       ~{a_max/g * 70:.0f} kg on chest
                        (for 70kg driver)
    
    ═══════════════════════════════════
    """
    
    ax2.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison to road car
    road_car_mu = 0.8
    d_road, t_road = braking_distance(v_entry_kmh, v_corner_kmh, road_car_mu)
    print(f"\nComparison to road car (μ={road_car_mu}):")
    print(f"  Road car needs: {d_road:.1f}m ({d_road - d_brake:.1f}m longer)")
    print(f"  F1 advantage: {((d_road - d_brake) / d_road * 100):.1f}% shorter braking distance")
```

---

### **Step 3: Add Interactive Sliders**

```python
# Create interactive widget with sliders
interact(interactive_braking,
         v_entry_kmh=IntSlider(min=100, max=350, value=280, step=10, 
                               description='Entry Speed (km/h):',
                               style={'description_width': '160px'}),
         v_corner_kmh=IntSlider(min=50, max=250, value=120, step=10, 
                                description='Corner Speed (km/h):',
                                style={'description_width': '160px'}),
         mu=FloatSlider(min=0.5, max=2.5, value=2.0, step=0.1, 
                       description='Tire Grip (μ):',
                       style={'description_width': '160px'}),
         show_zones=True)
```

---

### **Final Checkpoint**

**You should have:**
1. Interactive sliders controlling entry speed, corner speed, and grip
2. Real-time updating plot showing braking zone
3. Performance metrics display showing distance, time, and g-forces
4. Comparison to road car performance

**Experiments to try:**
1. **Wet conditions:** Set μ = 1.2 and watch braking distance nearly double
2. **High-speed corner:** Try v_entry = 310, v_corner = 220 (Silverstone Copse style)
3. **Hairpin:** Try v_entry = 180, v_corner = 50 (Monaco Loews style)
4. **Extreme grip:** Set μ = 2.5 (brand new softs, optimal temperature)
5. **Poor grip:** Set μ = 0.6 (very wet or cold tires)

**Real-world scenarios:**
- **Qualifying (low fuel, fresh tires):** μ = 2.2, maximum attack
- **Race start (full fuel, worn formation lap tires):** μ = 1.8, more conservative
- **End of stint (degraded tires):** μ = 1.6, earlier braking needed

---

## **Project Deliverables**

Submit your Colab notebook with:

### **1. Code (Well-Commented)**
- All stages completed and working
- Clear comments explaining physics and formulas
- Meaningful variable names
- Function documentation (docstrings)

### **2. Visualizations**
Include at least:
- Basic braking velocity plot (Stage 1)
- Velocity vs distance braking zone plot (Stage 2)
- Multi-scenario comparison (Stage 2)
- Corner entry visualization (Stage 3)
- Dry vs wet comparison bar chart (Stage 4)
- Track overhead view with braking zones (Stage 4)
- Screenshot of interactive tool (Stage 5, if attempted)

### **3. Written Responses**

Answer these questions in markdown cells:

**Physics & Engineering Questions:**
1. Explain why maximum deceleration (a_max = μg) is independent of car mass. What does this mean for F1 car design?
2. How does the v² term in the stopping distance formula (d = v²/2a) affect braking at high speeds? If you double your speed, how much longer does braking take?
3. Why do F1 drivers brake earlier in wet conditions? Use your calculations to support your answer.

**Application Questions:**
4. Compare braking from 300 km/h to 100 km/h in dry (μ=2.0) vs wet (μ=1.2) conditions. Calculate:
   - Stopping distances
   - Time difference
   - Percentage increase in wet
5. An F1 driver overshoots the braking point by 10 meters. If they were approaching at 280 km/h for a corner requiring 120 km/h entry, what speed will they carry into the corner? (Hint: Use v² = v₀² - 2ad)

**Strategic Analysis:**
6. Using the track in Stage 4, calculate the total time spent braking in one lap for:
   - Dry conditions (μ=2.0)
   - Wet conditions (μ=1.2)
   How much lap time is lost just from longer braking in the wet?

**Reflection:**
7. What was the most surprising result from your simulations?
8. How does this project change your understanding of what F1 drivers must do on every lap?
9. If you were coaching a racing driver, how would you use this tool to help them improve?

---

## **Success Criteria**

### **Complete (Meets Expectations)**
- ✅ Stages 1-3 fully functional
- ✅ Code runs without errors
- ✅ Basic plots are clear and properly labeled
- ✅ Analysis questions answered with calculations

### **Proficient (Exceeds Expectations)**
- ✅ All of above, plus:
- ✅ Code is well-commented with physics explanations
- ✅ Professional-looking plots with proper formatting
- ✅ Stage 4 comparison analysis completed
- ✅ Thoughtful, detailed responses showing understanding of physics

### **Excellent (Outstanding)**
- ✅ All of above, plus:
- ✅ Stage 5 interactive tool working smoothly
- ✅ Additional visualizations beyond requirements
- ✅ Extensions attempted (tire degradation, different tracks, etc.)
- ✅ Clear documentation that demonstrates deep understanding

### **Exceptional (Goes Above and Beyond)**
- ✅ All of above, plus:
- ✅ Original extension related to racing or vehicle dynamics
- ✅ Exploration of advanced concepts (combined braking+turning, downforce effects, brake temperature)
- ✅ Could serve as teaching resource for other students

---

## **Tips for Success**

### **Before You Start**
- Review Newton's 2nd law and kinematic equations
- Understand the friction circle concept
- Watch an F1 onboard lap to see braking zones in action

### **While Working**
- Work through stages sequentially
- Test each function before moving forward
- Use print statements to verify calculations
- Compare your results to real F1 data when possible

### **When You Get Stuck**
1. Check your unit conversions (km/h ↔ m/s)
2. Verify signs in equations (deceleration is slowing, not speeding up)
3. Print intermediate values to debug
4. Review the physics formulas—does your code match the math?

### **Best Practices**
- Comment physics formulas in your code
- Use descriptive variable names (`v_entry` not `v1`)
- Label plots with units (m, m/s, km/h, seconds)
- Include comparison to road cars for perspective

---

## **Extensions & Next Steps**

After completing this project, you could explore:

1. **Combined Braking + Cornering:** Model the friction circle with simultaneous braking and turning
2. **Downforce Effects:** Add aerodynamic grip that increases with speed (μ_total = μ_mechanical + μ_aero(v))
3. **Brake Temperature:** Model how repeated braking heats brakes and affects performance
4. **Trail Braking:** Calculate optimal brake release as you turn into corners
5. **Full Lap Simulation:** String together all corners with acceleration zones between
6. **Race Strategy:** Optimize fuel load vs lap time trade-offs
7. **Telemetry Analysis:** Compare your model to real F1 GPS/telemetry data

**You've got this! 🏎️💨**
