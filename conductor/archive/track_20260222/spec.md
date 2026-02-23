# Track Specification: Enhanced Progress Restoration

## Overview

This track implements a more intuitive and user-friendly progress restoration system for SnakeCharming. Currently, the application saves stage completion data but doesn't restore the user's reading position or provide a welcoming return experience.

## Current State

The application currently:
- Saves completed stages per lesson in localStorage
- Saves revealed code blocks per lesson
- Shows a generic progress notification banner when returning to a lesson
- Does NOT save or restore scroll position
- Does NOT provide a detailed welcome-back experience

## Desired State

After implementation, the application will:
- Save scroll position (or nearest section header) when user leaves a lesson
- Animate scroll restoration to the saved position when user returns
- Display a welcome-back modal with:
  - Summary of completed stages
  - Current position indicator
  - "Don't show again for this lesson" option
- Persist modal dismissal preference per lesson
- Maintain all existing progress tracking functionality

---

## Functional Requirements

### FR1: Scroll Position Persistence
- **FR1.1:** When a user navigates away from or closes a lesson, save the current scroll position (Y offset)
- **FR1.2:** If the exact scroll position doesn't correspond to a visible section, find the nearest section header above that position
- **FR1.3:** Store scroll position in localStorage with key format: `scrollPosition_[lessonFilename]`
- **FR1.4:** Scroll position should be updated periodically (e.g., every 2 seconds) and on page unload

### FR2: Animated Scroll Restoration
- **FR2.1:** When a lesson loads and has saved scroll position, animate scroll to that position
- **FR2.2:** Scroll animation should be smooth (CSS `scroll-behavior: smooth` or JavaScript animation)
- **FR2.3:** If exact position is not available, scroll to the nearest stage header above the saved position
- **FR2.4:** Scroll restoration should occur after lesson content is fully rendered

### FR3: Welcome-Back Modal
- **FR3.1:** Display a modal when user returns to a lesson with saved progress
- **FR3.2:** Modal should NOT show on first visit (no saved progress)
- **FR3.3:** Modal content must include:
  - Welcome message ("Welcome back!")
  - Number of completed stages (e.g., "You've completed 2 of 4 stages")
  - List of completed stage names/titles
  - Current position indicator (e.g., "Last viewed: Stage 3")
  - "Continue where you left off" button (closes modal, triggers scroll)
  - "Start from beginning" button (clears progress for this lesson, reloads)
  - "Don't show this again for this lesson" checkbox
- **FR3.4:** Modal should be dismissible by clicking outside or pressing Escape
- **FR3.5:** Modal should have a semi-transparent backdrop overlay

### FR4: Modal Dismissal Preference
- **FR4.1:** If user checks "Don't show again", store preference in localStorage
- **FR4.2:** Key format: `skipWelcomeModal_[lessonFilename]` = `true`
- **FR4.3:** When preference is set, skip modal display on subsequent visits
- **FR4.4:** User can reset preference by clearing browser localStorage

### FR5: Progress Notification Enhancement
- **FR5.1:** Keep existing progress notification banner as a fallback
- **FR5.2:** Modal takes precedence over banner (if modal shown, suppress banner)
- **FR5.3:** If modal dismissed with "Don't show again", show brief toast notification instead

---

## Technical Requirements

### TR1: localStorage Schema
```javascript
{
  // Existing keys (unchanged)
  `lessonProgress_[filename]`: {
    currentStage: number,
    completedStages: number[],
    lastUpdated: ISODateString
  }
  `revealedBlocks_[filename]`: string[]  // Array of revealed block IDs

  // New keys
  `scrollPosition_[filename]`: number  // Y offset in pixels
  `skipWelcomeModal_[filename]`: boolean  // User preference
}
```

### TR2: Scroll Position Calculation
- Calculate scroll position relative to `window.scrollY`
- On save, also capture the ID of the nearest visible stage header
- On restore, prioritize stage header ID over pixel position for accuracy

### TR3: Modal Component
- Reuse existing modal styling from `confirmResetProgress()` method
- Create new method: `showWelcomeBackModal()`
- Modal HTML structure should follow existing patterns
- CSS classes should follow project naming conventions

### TR4: Animation
- Use CSS transitions for smooth scroll: `scroll-behavior: smooth`
- Modal fade-in animation: 0.3s ease
- Backdrop fade-in: 0.3s ease

### TR5: Browser Compatibility
- Must work on Chrome, Firefox, Safari, Edge (latest versions)
- Graceful degradation: If scroll restoration fails, show console warning but don't break
- localStorage must be available (check and handle if disabled)

---

## UI/UX Requirements

### UR1: Modal Design
- Follow existing dark theme aesthetic
- Modal background: `var(--bg-secondary)`
- Border: `2px solid var(--accent-primary)`
- Text: `var(--text-primary)` for headings, `var(--text-secondary)` for body
- Buttons: Follow existing button styles (primary/secondary variants)

### UR2: Modal Content Layout
```
┌─────────────────────────────────────────┐
│  🎉 Welcome Back!                       │
│                                         │
│  You've completed 2 of 4 stages in     │
│  this lesson. Great work!              │
│                                         │
│  ✅ Stage 1: The Basic Trajectory      │
│  ✅ Stage 2: Creating a Reusable       │
│  📍 Last viewed: Stage 3               │
│                                         │
│  [ ] Don't show this again for this    │
│      lesson                            │
│                                         │
│  [Continue Where You Left Off]         │
│  [Start from Beginning]                │
└─────────────────────────────────────────┘
```

### UR3: Scroll Animation
- Duration: 0.5-1.0 seconds
- Easing: ease-in-out
- Should not cause motion sickness (respect `prefers-reduced-motion`)

### UR4: Responsive Behavior
- Modal must be readable on mobile devices
- Max-width: 500px, but shrink for small screens
- Buttons should be touch-friendly (min 44px height)

---

## Accessibility Requirements

### AR1: Keyboard Navigation
- Tab key cycles through modal elements
- Escape key closes modal
- Enter/Space activates focused button

### AR2: Screen Reader Support
- Modal has `role="dialog"` and `aria-modal="true"`
- Modal has descriptive `aria-labelledby` pointing to title
- Focus is trapped within modal while open
- Focus returns to trigger element when modal closes

### AR3: Motion Sensitivity
- Respect `prefers-reduced-motion` media query
- If set, disable scroll animation and modal fade-in

### AR4: Color Contrast
- All text meets WCAG AA contrast requirements
- Buttons have clear focus indicators

---

## Acceptance Criteria

### AC1: First Visit (No Saved Progress)
- [ ] No modal appears
- [ ] Lesson loads at top (scrollY = 0)
- [ ] Progress bar shows 0%
- [ ] No scroll restoration occurs

### AC2: Return Visit with Saved Progress
- [ ] Modal appears after lesson loads
- [ ] Modal shows correct number of completed stages
- [ ] Modal lists completed stage titles correctly
- [ ] Modal shows last viewed section
- [ ] Scroll position is restored after modal closes

### AC3: "Continue Where You Left Off"
- [ ] Modal closes
- [ ] Page animates scroll to saved position
- [ ] Stage at that position is highlighted/active

### AC4: "Start from Beginning"
- [ ] Confirmation dialog appears ("Are you sure?")
- [ ] If confirmed: clears all progress for this lesson
- [ ] Page reloads or resets to stage 1
- [ ] Scroll position is 0

### AC5: "Don't Show Again" Checked
- [ ] Preference saved to localStorage
- [ ] Modal does not appear on next visit
- [ ] Brief toast notification shows instead
- [ ] Scroll restoration still occurs

### AC6: Modal Dismissal
- [ ] Clicking backdrop closes modal
- [ ] Pressing Escape closes modal
- [ ] Scroll restoration still occurs after dismissal

### AC7: Edge Cases
- [ ] Works correctly if lesson structure changed (missing stages)
- [ ] Handles very long stage titles (text wrapping)
- [ ] Works on mobile devices
- [ ] Handles localStorage being unavailable

---

## Out of Scope

- Cross-device progress sync (requires backend)
- Progress export/import functionality
- Social sharing of progress
- Time-based reminders to return
- Email notifications

---

## Success Metrics

After implementation:
- Users return to lessons more frequently (measurable via future analytics)
- Reduced confusion about "where was I?"
- Improved user satisfaction (qualitative feedback)
- No regression in existing progress tracking functionality
