# Implementation Plan: Event-Driven Modal System Foundation

## Phase 1: Core Event-Driven Modal System

- [x] **Task: Set up modal event listener infrastructure** `modal-listeners`
  - [x] Add `setupModalEventListeners()` method to `LessonViewer` class
  - [x] Register listeners for `modal:show` and `modal:hide` events in `init()`
  - [x] Create `handleModalShow(payload)` method that renders modal from payload
  - [x] Create `handleModalHide()` method that closes modal
  - [x] Add JSDoc comments to all new methods

- [x] **Task: Create generic modal HTML structure** `977f52c`
  - [x] Add generic modal HTML to `index.html` (reusable container)
  - [x] Include: modal container, header (title), content area, actions area
  - [x] Add `role="dialog"`, `aria-modal="true"`, `aria-labelledby` attributes
  - [x] Add backdrop overlay element
  - [x] Modal hidden by default, shown via CSS class

- [x] **Task: Add generic modal CSS styles** `977f52c`
  - [x] Add modal styles to `index.html` `<style>` block
  - [x] Follow existing modal naming convention (`.modal`, `.modal-content`, etc.)
  - [x] Dark theme colors matching existing design
  - [x] Responsive sizing (max-width 500px, shrink for mobile)
  - [x] Fade-in animation (0.3s ease)
  - [x] Touch-friendly buttons (min 44px height)
  - [x] `prefers-reduced-motion` support

- [x] **Task: Implement modal rendering logic** `977f52c`
  - [x] Parse payload and populate modal title
  - [x] Support string content (innerHTML)
  - [x] Support DOM element content (appendChild)
  - [x] Render action buttons dynamically from payload
  - [x] Wire up button click handlers (call onClick, close modal)
  - [x] Add backdrop click handler (close modal)
  - [x] Add Escape key handler (close modal)

- [x] **Task: Implement focus restoration** `977f52c`
  - [x] Store `previouslyFocused` element before showing modal
  - [x] Restore focus on modal close
  - [x] Handle edge case: element no longer in DOM

- [x] **Task: Test event-driven modal manually** `977f52c`
  - [x] Dispatch `modal:show` event with info modal payload
  - [x] Verify modal appears with correct content
  - [x] Test all close mechanisms (Escape, backdrop, button click)
  - [x] Verify focus restoration
  - [x] Test with different payload types

---

## Phase 2: Lesson Preview Card Component

- [x] **Task: Create lesson preview card renderer** `card-renderer`
  - [x] Add `createLessonPreviewCard(lessonData)` method
  - [x] Card structure: header with emoji, description, metadata grid, actions
  - [x] Metadata fields: difficulty, time, topic
  - [x] Return DOM element (not HTML string)
  - [x] Follow existing CSS naming conventions

- [x] **Task: Add lesson preview card CSS styles** `card-renderer`
  - [x] Card container with border and background
  - [x] Metadata grid layout (CSS Grid or Flexbox)
  - [x] Badge styling for difficulty level
  - [x] Responsive layout for mobile
  - [x] Hover states for interactive elements

- [x] **Task: Test lesson preview card rendering** `4121954`
  - [x] Dispatch `modal:show` with lesson preview card content
  - [x] Verify card renders correctly
  - [x] Test on mobile viewport
  - [x] Verify button touch targets are adequate

---

## Phase 3: State Management Utilities

- [x] **Task: Implement seen lessons LocalStorage utilities** `seen-lessons`
  - [x] Add `getSeenLessons()` static method - reads from LocalStorage
  - [x] Add `markLessonSeen(filename)` method - adds to list, saves
  - [x] Add `hasLessonBeenSeen(filename)` method - returns boolean
  - [x] Add `getUnseenLessons(availableLessons)` method - filters list
  - [x] Handle LocalStorage unavailable gracefully (try/catch)
  - [x] Add JSDoc comments to all methods

- [x] **Task: Integrate state utilities with LessonViewer** `seen-lessons`
  - [x] Add `seenLessons` property to LessonViewer class
  - [x] Load seen lessons in `init()` or `loadLesson()`
  - [x] Expose utilities via `window.lessonViewer.seenLessons` for debugging

- [x] **Task: Test state management utilities** `250c5ad`
  - [x] Call `markLessonSeen('test.md')`, verify LocalStorage updated
  - [x] Call `hasLessonBeenSeen('test.md')`, verify returns true
  - [x] Call `getUnseenLessons([...])`, verify correct filtering
  - [x] Test with LocalStorage disabled (verify no errors)

---

## Phase 4: Integration & Backward Compatibility

- [x] **Task: Verify backward compatibility** `integration`
  - [x] Test `showWelcomeBackModal()` still works
  - [x] Test `confirmResetProgress()` still works
  - [x] Test `confirmStartOver()` still works
  - [x] Verify no console errors from existing code

- [x] **Task: Add example usage documentation** `integration`
  - [x] Add code comments showing how to dispatch `modal:show` events
  - [x] Document payload structure in JSDoc
  - [x] Add example in `doc/` directory or as code comment

- [x] **Task: Code cleanup and documentation** `integration`
  - [x] Add JSDoc comments to all new public methods
  - [x] Ensure consistent naming conventions
  - [x] Remove any debug console.log statements
  - [x] Verify code follows `javascript.md` style guide

- [x] **Task: Final testing pass** `f649784`
  - [x] Test all acceptance criteria from spec.md
  - [x] Cross-browser testing (Chrome, Firefox, Safari, Edge)
  - [x] Mobile testing (iOS Safari, Android Chrome)
  - [x] Verify no regressions in existing functionality

---

## Phase Completion Verification and Checkpointing Protocol

- [x] **Task: Conductor - User Manual Verification 'Phase 1: Core Event-Driven Modal System' (Protocol in workflow.md)**

- [x] **Task: Conductor - User Manual Verification 'Phase 2: Lesson Preview Card Component' (Protocol in workflow.md)**

- [x] **Task: Conductor - User Manual Verification 'Phase 3: State Management Utilities' (Protocol in workflow.md)**

- [x] **Task: Conductor - User Manual Verification 'Phase 4: Integration & Backward Compatibility' (Protocol in workflow.md)**
