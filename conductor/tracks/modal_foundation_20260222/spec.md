# Track Specification: Event-Driven Modal System Foundation

## Overview

This track builds a reusable, event-driven modal system for SnakeCharming that decouples modal triggers from modal rendering. The system provides a foundation for displaying various modal types (confirmations, announcements, info dialogs) throughout the application.

**Note:** This is a **foundation track**. The actual "New Lesson Announcement" feature will be implemented in a subsequent track using this modal system.

## Current State

The application currently has:
- A welcome-back modal for returning users (hardcoded in `showWelcomeBackModal()`)
- A reset progress confirmation modal (hardcoded in `confirmResetProgress()`)
- A resources overlay modal
- Each modal is implemented independently with duplicated HTML/CSS/JS patterns
- No centralized modal management or reusability

## Desired State

After implementation:
- A generic event-driven modal system using `modal:show` and `modal:hide` custom events
- Modal content is configured via payload objects passed with events
- Support for lesson preview cards (title, description, metadata, action buttons)
- LocalStorage-based state tracking for "seen lessons" (prepared for future track)
- Core accessibility features (Escape to close, ARIA labels)
- All existing modal functionality preserved

---

## Functional Requirements

### FR1: Event-Driven Modal API

- **FR1.1:** The system MUST listen for `modal:show` events dispatched on the `window` object
- **FR1.2:** The `modal:show` event MUST carry a `detail.payload` object with the following structure:
  ```javascript
  {
    type: 'info' | 'confirm' | 'announcement' | 'custom',
    title: string,
    content: string | HTMLElement,
    actions?: Array<{
      label: string,
      onClick: () => void,
      variant?: 'primary' | 'secondary' | 'danger'
    }>
  }
  ```
- **FR1.3:** The system MUST listen for `modal:hide` events to programmatically close modals
- **FR1.4:** The modal MUST close when:
  - The Escape key is pressed
  - The user clicks outside the modal content (on the backdrop)
  - A modal action button is clicked

### FR2: Modal Content Rendering

- **FR2.1:** Support plain HTML string content
- **FR2.2:** Support DOM element content (for complex layouts like lesson preview cards)
- **FR2.3:** Lesson preview card layout MUST support:
  - Title (e.g., "🎉 New Lesson Available!")
  - Lesson name and brief description
  - Metadata: difficulty level, estimated time, topic
  - Primary action button (e.g., "Try it now")
  - Secondary action button (e.g., "Maybe later")

### FR3: State Management (Foundation)

- **FR3.1:** Provide a utility method `getSeenLessons()` that reads from LocalStorage key `seenLessons`
- **FR3.2:** Provide a utility method `markLessonSeen(filename)` that adds a lesson to the seen list
- **FR3.3:** Provide a utility method `hasLessonBeenSeen(filename)` that returns a boolean
- **FR3.4:** Provide a utility method `getUnseenLessons(availableLessons)` that filters and returns lessons not in the seen list
- **FR3.5:** LocalStorage data format:
  ```json
  {
    "seenLessons": ["projectile-simulator.md", "f1-braking.md"]
  }
  ```

### FR4: Accessibility

- **FR4.1:** Modal MUST have `role="dialog"`, `aria-modal="true"`, and `aria-labelledby` pointing to the title
- **FR4.2:** Pressing Escape MUST close the modal
- **FR4.3:** Focus MUST be restored to the triggering element when the modal closes
- **FR4.4:** Modal MUST trap focus internally (optional, configurable per modal type)

### FR5: Backward Compatibility

- **FR5.1:** Existing `showWelcomeBackModal()` MUST continue to work unchanged
- **FR5.2:** Existing `confirmResetProgress()` MUST continue to work unchanged
- **FR5.3:** Existing `confirmStartOver()` MUST continue to work unchanged

---

## Non-Functional Requirements

### NFR1: Code Organization

- Modal event listeners registered in `LessonViewer.init()`
- Modal rendering logic in new methods: `handleModalShow(payload)`, `handleModalHide()`
- State management utilities as static methods or separate module
- All styles added to existing `<style>` block in `index.html`

### NFR2: Performance

- Modal HTML injected only when first shown (lazy rendering)
- Event listeners properly cleaned up if needed
- No memory leaks from event listener accumulation

### NFR3: Maintainability

- JSDoc comments on all new public methods
- Follow existing JavaScript naming conventions (camelCase methods, PascalCase classes)
- Consistent with existing code style in `app.js`

---

## Acceptance Criteria

### AC1: Event-Driven Modal Display

- [ ] Dispatch `window.dispatchEvent(new CustomEvent('modal:show', { detail: { payload: {...} } }))` with valid payload
- [ ] Modal appears with correct title, content, and action buttons
- [ ] Clicking action buttons triggers the correct callbacks and closes modal
- [ ] Pressing Escape closes the modal
- [ ] Clicking backdrop closes the modal

### AC2: Lesson Preview Card Rendering

- [ ] Pass a lesson preview card DOM element as content
- [ ] Card renders with proper styling (title, description, metadata, buttons)
- [ ] Buttons are touch-friendly (min 44px height)
- [ ] Card is readable on mobile devices

### AC3: State Management Utilities

- [ ] `hasLessonBeenSeen('projectile-simulator.md')` returns `true` after marking
- [ ] `getUnseenLessons([...])` correctly filters out seen lessons
- [ ] Data persists in LocalStorage across page reloads

### AC4: Accessibility

- [ ] Modal has correct ARIA attributes
- [ ] Escape key closes modal
- [ ] Focus restored to trigger element on close
- [ ] Screen reader announces modal title when opened

### AC5: Backward Compatibility

- [ ] Welcome-back modal still appears for returning users
- [ ] Reset progress confirmation still works
- [ ] No console errors from existing code

---

## Out of Scope

- Actual "New Lesson Announcement" feature (will use this foundation)
- Modal animations beyond existing fade-in
- Server-side lesson tracking or sync
- User preferences for modal frequency
- Analytics or tracking of modal interactions
- Focus trap implementation (optional, can be added per-modal)

---

## Technical Notes

### Event Payload Structure Example

```javascript
// Dispatching a lesson announcement
window.dispatchEvent(new CustomEvent('modal:show', {
  detail: {
    payload: {
      type: 'announcement',
      title: '🎉 New Lesson Available!',
      content: createLessonPreviewCard({
        name: 'F1 Braking Analysis',
        description: 'Analyze braking performance using real telemetry data.',
        difficulty: 'Intermediate',
        time: '15-20 min',
        topic: 'Kinematics'
      }),
      actions: [
        { label: 'Try it now', onClick: () => loadLesson('f1-braking.md'), variant: 'primary' },
        { label: 'Maybe later', onClick: () => {}, variant: 'secondary' }
      ]
    }
  }
}));
```

### LocalStorage Schema

```javascript
// Key: seenLessons
// Value: JSON array of lesson filenames
["projectile-simulator.md", "f1-braking.md"]
```

---

## Success Metrics

After implementation:
- Modal system is reusable and decoupled from specific use cases
- Code duplication reduced (no more hardcoded modal HTML in multiple methods)
- Future modal features can be added by dispatching events
- State management utilities ready for lesson announcement feature
