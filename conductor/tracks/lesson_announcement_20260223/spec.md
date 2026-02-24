# Track Specification: New Lesson Announcement Feature

## Overview

This track implements the user-facing "New Lesson Announcement" feature that uses the event-driven modal system foundation to inform users about newly added lessons when they first open the application.

## Current State

After the foundation track (`modal_foundation_20260222`), the application has:
- Event-driven modal system with `modal:show` / `modal:hide` events
- Lesson preview card component (`createLessonPreviewCard()`)
- LocalStorage utilities for tracking seen lessons (`getSeenLessons()`, `markLessonSeen()`, etc.)

The application does NOT yet:
- Automatically detect new lessons
- Show announcements to users
- Integrate the modal system with the lesson loading flow

## Desired State

After implementation:
- On page load, the app checks for unseen lessons
- If unseen lessons exist, a digest modal displays all of them (alphabetically ordered)
- Modal is informational only - no "Try it now" CTA
- Dismissing the modal marks all shown lessons as seen
- Users can discover new lessons naturally via the dropdown

---

## Functional Requirements

### FR1: Automatic Detection on Page Load

- **FR1.1:** When the app loads (LessonViewer initializes), check for unseen lessons
- **FR1.2:** Compare available lessons (from `LESSONS` array) against seen lessons (from LocalStorage)
- **FR1.3:** If unseen lessons exist, trigger the announcement modal
- **FR1.4:** If no unseen lessons, do nothing (silent operation)

### FR2: Digest Modal Display

- **FR2.1:** Show a single modal containing all unseen lessons
- **FR2.2:** Modal title: "🎉 New Lessons Available!" (or singular if only one)
- **FR2.3:** Display each unseen lesson as a preview card (using `createLessonPreviewCard()`)
- **FR2.4:** Order lessons alphabetically by name
- **FR2.5:** Modal has a single "Got it!" dismiss button

### FR3: Dismissal Behavior

- **FR3.1:** When user clicks "Got it!", mark all displayed lessons as seen
- **FR3.2:** Close the modal
- **FR3.3:** No navigation or lesson loading occurs
- **FR3.4:** User continues using the app normally

### FR4: Edge Cases

- **FR4.1:** If only one unseen lesson, still show digest format (consistent UX)
- **FR4.2:** If all lessons are unseen (first-time user), show the modal
- **FR4.3:** Handle LocalStorage unavailable gracefully (skip announcement, no errors)

---

## Non-Functional Requirements

### NFR1: Code Organization

- New method `checkAndAnnounceNewLessons()` in LessonViewer class
- Called from `init()` after lesson data is available
- Uses existing `getUnseenLessons()` utility from foundation track

### NFR2: User Experience

- Non-intrusive - informational only, no pressure to act
- Clear, friendly messaging consistent with product guidelines
- Modal appears after initial page load (not blocking)

### NFR3: Maintainability

- JSDoc comments on all new methods
- Follow existing JavaScript naming conventions
- Consistent with foundation track patterns

---

## Acceptance Criteria

### AC1: Page Load Detection

- [ ] Open app with unseen lessons in LocalStorage
- [ ] Verify modal appears within 1 second of page load
- [ ] Verify modal shows correct list of unseen lessons

### AC2: Digest Modal Content

- [ ] Modal title is "🎉 New Lessons Available!" (or singular variant)
- [ ] Each lesson shows: name, description, difficulty, time, topic
- [ ] Lessons are ordered alphabetically by name
- [ ] Single "Got it!" button is present

### AC3: Dismissal Behavior

- [ ] Clicking "Got it!" closes the modal
- [ ] All displayed lessons are marked as seen in LocalStorage
- [ ] Reloading the page does NOT show the modal again

### AC4: No Unseen Lessons

- [ ] Open app with all lessons marked as seen
- [ ] Verify no modal appears
- [ ] Verify no console errors

### AC5: Backward Compatibility

- [ ] Existing lesson loading functionality unchanged
- [ ] Existing modal system (welcome-back, reset confirm) still works
- [ ] No regressions in existing functionality

---

## Out of Scope

- Analytics or tracking of modal interactions
- User preferences for announcement frequency
- "Snooze" or "Remind me later" functionality
- Per-lesson dismissal (all-or-nothing)

---

## Technical Notes

### Integration Point

```javascript
// In LessonViewer.init(), after setup:
init() {
  // ... existing initialization ...
  this.setupModalEventListeners();
  this.checkAndAnnounceNewLessons(); // NEW: Check for new lessons
}
```

### Modal Payload Structure

```javascript
window.dispatchEvent(new CustomEvent('modal:show', {
  detail: {
    payload: {
      type: 'announcement',
      title: '🎉 New Lessons Available!',
      content: createDigestContent(unseenLessons), // DOM element with cards
      actions: [
        { label: 'Got it!', onClick: () => markAllAsSeen(unseenLessons), variant: 'primary' }
      ]
    }
  }
}));
```

---

## Success Metrics

After implementation:
- Users are informed about new lessons without being intrusive
- No increase in bounce rate (non-annoying UX)
- Clean integration with existing modal foundation
