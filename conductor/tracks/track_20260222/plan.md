# Implementation Plan: Enhanced Progress Restoration

## Phase 1: Scroll Position Persistence

- [x] **Task: Analyze existing progress tracking code** `a1b2c3d`
  - Read `app.js` and understand current `saveProgress()`, `loadProgress()` methods
  - Identify where to hook scroll position saving
  - Document findings

- [x] **Task: Create scroll position save function** `75153a3`
  - Add `saveScrollPosition(filename)` method to `LessonViewer` class
  - Save `window.scrollY` to localStorage with key `scrollPosition_[filename]`
  - Also save nearest stage header ID for more accurate restoration
  - Call this method:
    - On `beforeunload` event
    - Periodically (every 2 seconds) via `setInterval`
    - When user completes a stage

- [x] **Task: Create scroll position load function** `75153a3`
  - Add `loadScrollPosition(filename)` method to `LessonViewer` class
  - Read from localStorage and return `{ scrollY: number, nearestStageId: string }`
  - Handle missing data gracefully (return null)

- [x] **Task: Implement animated scroll restoration** `75153a3`
  - Add `restoreScrollPosition()` method
  - Use `window.scrollTo({ top: y, behavior: 'smooth' })` for smooth animation
  - If stage ID available, use `element.scrollIntoView({ behavior: 'smooth' })`
  - Respect `prefers-reduced-motion` media query
  - Call after lesson content is fully rendered

- [x] **Task: Test scroll persistence** `3b1d619`
  - Load a lesson, scroll to middle, close tab
  - Reopen lesson, verify scroll position restored
  - Test with different lessons
  - Verify no console errors
  - Note: Manual browser testing required for full verification

---

## Phase 2: Welcome-Back Modal

- [x] **Task: Create modal HTML structure** `modal-html`
  - Add modal HTML to `index.html` (can reuse existing modal patterns)
  - Include:
    - Modal container with `role="dialog"`
    - Title: "Welcome Back!"
    - Progress summary text
    - Completed stages list container
    - Last viewed section indicator
    - Checkbox: "Don't show again"
    - Two buttons: "Continue" and "Start Over"
  - Add backdrop overlay element

- [x] **Task: Add modal CSS styles** `modal-html`
  - Follow existing modal styling from `confirmResetProgress()`
  - Dark theme colors
  - Responsive sizing (max-width 500px)
  - Fade-in animation
  - Ensure touch-friendly buttons (min 44px height)
  - Add `prefers-reduced-motion` support

- [x] **Task: Implement `showWelcomeBackModal()` method** `modal-html`
  - Check if lesson has saved progress (skip if first visit)
  - Check if user disabled modal for this lesson
  - Populate modal with:
    - Count of completed stages
    - List of completed stage titles (from `stages` array)
    - Last viewed section name
  - Display modal with fade-in animation
  - Trap focus within modal
  - Return Promise that resolves when modal closes

- [x] **Task: Implement modal action handlers** `modal-html`
  - **"Continue" button:** Close modal, trigger scroll restoration
  - **"Start Over" button:** Show confirmation, clear progress, reload lesson
  - **Checkbox:** Save preference to `skipWelcomeModal_[filename]`
  - **Backdrop click:** Close modal, still restore scroll
  - **Escape key:** Close modal, still restore scroll

- [x] **Task: Integrate modal into lesson load flow** `modal-html`
  - Modify `loadLesson()` method
  - After progress is loaded and content rendered:
    - Call `showWelcomeBackModal()` if applicable
    - Await user action
    - Then restore scroll position
  - Ensure existing progress notification is suppressed when modal shown

- [x] **Task: Add accessibility features** `modal-html`
  - Focus trap implementation
  - Keyboard navigation (Tab, Escape, Enter)
  - Screen reader announcements
  - ARIA labels and roles
  - Focus restoration after modal closes

- [x] **Task: Test modal functionality** `3d100f0`
  - First visit: no modal appears
  - Return visit: modal appears with correct content
  - "Continue" works and scrolls to position
  - "Start Over" clears progress correctly
  - "Don't show again" persists and works
  - Keyboard navigation works
  - Screen reader testing (if possible)
  - Mobile testing

---

## Phase 3: Polish and Edge Cases

- [ ] **Task: Handle edge cases**
  - Lesson structure changed (stage no longer exists): scroll to nearest valid stage
  - Very long stage titles: CSS text wrapping
  - localStorage unavailable: graceful degradation, no errors
  - Very old saved progress: still works correctly

- [x] **Task: Add toast notification for skipped modal** `toast-notif`
  - Create brief notification when modal is skipped
  - Shows "Progress restored. Scroll to last position..."
  - Auto-dismiss after 3 seconds
  - Less intrusive than full modal

- [x] **Task: Optimize scroll position saving** `262fbf3`
  - Debounce scroll saves (don't save on every scroll event)
  - Save on stage completion
  - Save on visibility change (tab switch)

- [x] **Task: Code cleanup and documentation** `cleanup-docs`
  - Add JSDoc comments to new methods
  - Ensure consistent naming conventions
  - Remove any debug console.log statements
  - Verify code follows `javascript.md` style guide

- [x] **Task: Final testing pass** `8ab161a`
  - Test all acceptance criteria from spec.md
  - Cross-browser testing (Chrome, Firefox, Safari, Edge)
  - Mobile testing (iOS Safari, Android Chrome)
  - Verify no regressions in existing functionality
  - Test with multiple lessons simultaneously

---

## Phase Completion Verification and Checkpointing Protocol

- [ ] **Task: Conductor - User Manual Verification 'Phase 1: Scroll Position Persistence' (Protocol in workflow.md)**

- [ ] **Task: Conductor - User Manual Verification 'Phase 2: Welcome-Back Modal' (Protocol in workflow.md)**

- [ ] **Task: Conductor - User Manual Verification 'Phase 3: Polish and Edge Cases' (Protocol in workflow.md)**
