# Implementation Plan: Event-Driven Modal System Foundation

## Phase 1: Core Event-Driven Modal System

- [ ] **Task: Set up modal event listener infrastructure**
  - [ ] Add `setupModalEventListeners()` method to `LessonViewer` class
  - [ ] Register listeners for `modal:show` and `modal:hide` events in `init()`
  - [ ] Create `handleModalShow(payload)` method that renders modal from payload
  - [ ] Create `handleModalHide()` method that closes modal
  - [ ] Add JSDoc comments to all new methods

- [ ] **Task: Create generic modal HTML structure**
  - [ ] Add generic modal HTML to `index.html` (reusable container)
  - [ ] Include: modal container, header (title), content area, actions area
  - [ ] Add `role="dialog"`, `aria-modal="true"`, `aria-labelledby` attributes
  - [ ] Add backdrop overlay element
  - [ ] Modal hidden by default, shown via CSS class

- [ ] **Task: Add generic modal CSS styles**
  - [ ] Add modal styles to `index.html` `<style>` block
  - [ ] Follow existing modal naming convention (`.modal`, `.modal-content`, etc.)
  - [ ] Dark theme colors matching existing design
  - [ ] Responsive sizing (max-width 500px, shrink for mobile)
  - [ ] Fade-in animation (0.3s ease)
  - [ ] Touch-friendly buttons (min 44px height)
  - [ ] `prefers-reduced-motion` support

- [ ] **Task: Implement modal rendering logic**
  - [ ] Parse payload and populate modal title
  - [ ] Support string content (innerHTML)
  - [ ] Support DOM element content (appendChild)
  - [ ] Render action buttons dynamically from payload
  - [ ] Wire up button click handlers (call onClick, close modal)
  - [ ] Add backdrop click handler (close modal)
  - [ ] Add Escape key handler (close modal)

- [ ] **Task: Implement focus restoration**
  - [ ] Store `previouslyFocused` element before showing modal
  - [ ] Restore focus on modal close
  - [ ] Handle edge case: element no longer in DOM

- [ ] **Task: Test event-driven modal manually**
  - [ ] Dispatch `modal:show` event with info modal payload
  - [ ] Verify modal appears with correct content
  - [ ] Test all close mechanisms (Escape, backdrop, button click)
  - [ ] Verify focus restoration
  - [ ] Test with different payload types

---

## Phase 2: Lesson Preview Card Component

- [ ] **Task: Create lesson preview card renderer**
  - [ ] Add `createLessonPreviewCard(lessonData)` method
  - [ ] Card structure: header with emoji, description, metadata grid, actions
  - [ ] Metadata fields: difficulty, time, topic
  - [ ] Return DOM element (not HTML string)
  - [ ] Follow existing CSS naming conventions

- [ ] **Task: Add lesson preview card CSS styles**
  - [ ] Card container with border and background
  - [ ] Metadata grid layout (CSS Grid or Flexbox)
  - [ ] Badge styling for difficulty level
  - [ ] Responsive layout for mobile
  - [ ] Hover states for interactive elements

- [ ] **Task: Test lesson preview card rendering**
  - [ ] Dispatch `modal:show` with lesson preview card content
  - [ ] Verify card renders correctly
  - [ ] Test on mobile viewport
  - [ ] Verify button touch targets are adequate

---

## Phase 3: State Management Utilities

- [ ] **Task: Implement seen lessons LocalStorage utilities**
  - [ ] Add `getSeenLessons()` static method - reads from LocalStorage
  - [ ] Add `markLessonSeen(filename)` method - adds to list, saves
  - [ ] Add `hasLessonBeenSeen(filename)` method - returns boolean
  - [ ] Add `getUnseenLessons(availableLessons)` method - filters list
  - [ ] Handle LocalStorage unavailable gracefully (try/catch)
  - [ ] Add JSDoc comments to all methods

- [ ] **Task: Integrate state utilities with LessonViewer**
  - [ ] Add `seenLessons` property to LessonViewer class
  - [ ] Load seen lessons in `init()` or `loadLesson()`
  - [ ] Expose utilities via `window.lessonViewer.seenLessons` for debugging

- [ ] **Task: Test state management utilities**
  - [ ] Call `markLessonSeen('test.md')`, verify LocalStorage updated
  - [ ] Call `hasLessonBeenSeen('test.md')`, verify returns true
  - [ ] Call `getUnseenLessons([...])`, verify correct filtering
  - [ ] Test with LocalStorage disabled (verify no errors)

---

## Phase 4: Integration & Backward Compatibility

- [ ] **Task: Verify backward compatibility**
  - [ ] Test `showWelcomeBackModal()` still works
  - [ ] Test `confirmResetProgress()` still works
  - [ ] Test `confirmStartOver()` still works
  - [ ] Verify no console errors from existing code

- [ ] **Task: Add example usage documentation**
  - [ ] Add code comments showing how to dispatch `modal:show` events
  - [ ] Document payload structure in JSDoc
  - [ ] Add example in `doc/` directory or as code comment

- [ ] **Task: Code cleanup and documentation**
  - [ ] Add JSDoc comments to all new public methods
  - [ ] Ensure consistent naming conventions
  - [ ] Remove any debug console.log statements
  - [ ] Verify code follows `javascript.md` style guide

- [ ] **Task: Final testing pass**
  - [ ] Test all acceptance criteria from spec.md
  - [ ] Cross-browser testing (Chrome, Firefox, Safari, Edge)
  - [ ] Mobile testing (iOS Safari, Android Chrome)
  - [ ] Verify no regressions in existing functionality

---

## Phase Completion Verification and Checkpointing Protocol

- [ ] **Task: Conductor - User Manual Verification 'Phase 1: Core Event-Driven Modal System' (Protocol in workflow.md)**

- [ ] **Task: Conductor - User Manual Verification 'Phase 2: Lesson Preview Card Component' (Protocol in workflow.md)**

- [ ] **Task: Conductor - User Manual Verification 'Phase 3: State Management Utilities' (Protocol in workflow.md)**

- [ ] **Task: Conductor - User Manual Verification 'Phase 4: Integration & Backward Compatibility' (Protocol in workflow.md)**
