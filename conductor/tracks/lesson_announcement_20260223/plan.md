# Implementation Plan: New Lesson Announcement Feature

## Phase 1: Core Announcement Logic

- [ ] **Task: Create lesson digest content renderer**
  - [ ] Add `createDigestContent(unseenLessons)` method to LessonViewer class
  - [ ] Render multiple lesson preview cards stacked vertically
  - [ ] Add section title based on count (singular vs plural)
  - [ ] Return DOM element for modal content
  - [ ] Add JSDoc comments

- [ ] **Task: Implement checkAndAnnounceNewLessons method**
  - [ ] Add `checkAndAnnounceNewLessons()` method to LessonViewer class
  - [ ] Get unseen lessons using `getUnseenLessons(LESSONS.map(l => l.file))`
  - [ ] If unseen lessons exist, dispatch `modal:show` event
  - [ ] Build modal payload with digest content and "Got it!" action
  - [ ] Handle empty unseen list gracefully (no modal)
  - [ ] Add JSDoc comments

- [ ] **Task: Integrate announcement into initialization flow**
  - [ ] Call `checkAndAnnounceNewLessons()` in `init()` method
  - [ ] Ensure it runs after modal event listeners are set up
  - [ ] Add slight delay (e.g., 500ms) to not block initial render
  - [ ] Test that modal appears on page load

- [ ] **Task: Test core announcement functionality**
  - [ ] Clear LocalStorage, reload page, verify modal appears
  - [ ] Verify all lessons are listed in digest
  - [ ] Verify alphabetical ordering
  - [ ] Click "Got it!", verify modal closes
  - [ ] Reload page, verify modal does NOT appear again

---

## Phase 2: UI/UX Polish

- [ ] **Task: Add digest modal CSS styles**
  - [ ] Add `.lesson-digest-container` class for stacking cards
  - [ ] Add gap between cards for visual separation
  - [ ] Ensure cards are scrollable if many lessons
  - [ ] Add max-height with overflow for long lists
  - [ ] Mobile-responsive adjustments

- [ ] **Task: Refine modal messaging**
  - [ ] Dynamic title: "New Lesson Available!" (singular) vs "New Lessons Available!" (plural)
  - [ ] Add introductory text: "We've added {count} new lesson{plural} to help you learn..."
  - [ ] Ensure tone matches product guidelines (encouraging, conversational)

- [ ] **Task: Test digest modal UI**
  - [ ] Test with 1 lesson (verify singular messaging)
  - [ ] Test with 5+ lessons (verify scrolling behavior)
  - [ ] Test on mobile viewport (verify responsive layout)
  - [ ] Verify all lesson info displays correctly

---

## Phase 3: Edge Cases & Integration

- [ ] **Task: Handle edge cases**
  - [ ] All lessons unseen (first-time user): show modal normally
  - [ ] LocalStorage unavailable: skip announcement, no errors
  - [ ] LESSONS array empty: no modal, no errors
  - [ ] Lesson metadata missing (description, etc.): show fallback content

- [ ] **Task: Verify backward compatibility**
  - [ ] Existing welcome-back modal still works
  - [ ] Existing reset progress modal still works
  - [ ] Lesson loading functionality unchanged
  - [ ] No console errors from existing code

- [ ] **Task: Code cleanup and documentation**
  - [ ] Add JSDoc comments to all new public methods
  - [ ] Ensure consistent naming conventions (camelCase)
  - [ ] Remove any debug console.log statements
  - [ ] Verify code follows `javascript.md` style guide

- [ ] **Task: Final testing pass**
  - [ ] Test all acceptance criteria from spec.md
  - [ ] Cross-browser testing (Chrome, Firefox, Safari, Edge)
  - [ ] Mobile testing (iOS Safari, Android Chrome)
  - [ ] Verify no regressions in existing functionality

---

## Phase Completion Verification and Checkpointing Protocol

- [ ] **Task: Conductor - User Manual Verification 'Phase 1: Core Announcement Logic' (Protocol in workflow.md)**

- [ ] **Task: Conductor - User Manual Verification 'Phase 2: UI/UX Polish' (Protocol in workflow.md)**

- [ ] **Task: Conductor - User Manual Verification 'Phase 3: Edge Cases & Integration' (Protocol in workflow.md)**
