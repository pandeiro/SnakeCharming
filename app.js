// Configuration for available lessons
const LESSONS = [
  {
    name: 'Projectile Simulator',
    file: 'projectile-simulator.md'
  }
  // Add more lessons here as { name: 'Lesson Name', file: 'filename.md' }
];

// Only code blocks with <!-- PARTIAL_REVEAL --> marker get timed reveals
const REVEAL_DELAY = 75; // 1.25 min in seconds
const PARTIAL_REVEAL_MARKER = '<!-- PARTIAL_REVEAL -->';

class LessonViewer {
  constructor() {
    this.stages = [];
    this.currentStage = 0;
    this.completedStages = new Set();
    this.currentLessonFile = null;
    
    // Reveal queue state management
    this.revealQueue = [];
    this.revealedBlockIds = new Set();
    this.currentActiveIndex = -1;
    
    // Intersection observer
    this.observer = null;
    this.observedElements = new Set();
    
    this.init();
  }

  init() {
    this.populateLessonSelector();
    this.loadProgress();
    document.getElementById('lesson-select').addEventListener('change', (e) => {
      if (e.target.value) {
        this.loadLesson(e.target.value);
      }
    });
    this.setupIntersectionObserver();
    this.setupScrollListener();
    this.renderResources();
    this.setupScrollPositionSaving();
  }

  /**
   * Set up automatic scroll position saving
   */
  setupScrollPositionSaving() {
    // Save on beforeunload (page close/navigation)
    window.addEventListener('beforeunload', () => {
      this.saveScrollPosition(this.currentLessonFile);
    });

    // Save periodically (every 2 seconds)
    setInterval(() => {
      if (this.currentLessonFile) {
        this.saveScrollPosition(this.currentLessonFile);
      }
    }, 2000);

    // Save on visibility change (tab switch)
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && this.currentLessonFile) {
        this.saveScrollPosition(this.currentLessonFile);
      }
    });
  }

  renderResources() {
    const md = document.getElementById('resources-markdown').textContent;
    document.getElementById('resources-content').innerHTML = marked.parse(md);
    
    // Ensure links in resources open in new tab
    document.querySelectorAll('#resources-content a').forEach(a => {
      a.setAttribute('target', '_blank');
    });
  }

  setupScrollListener() {
    const header = document.getElementById('main-header');
    window.addEventListener('scroll', () => {
      if (window.scrollY > 50) {
        header.classList.add('shrunk');
      } else {
        header.classList.remove('shrunk');
      }
    });
  }

  setupIntersectionObserver() {
    this.observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        const revealId = entry.target.dataset.revealId;
        if (revealId) {
          const queueItem = this.revealQueue.find(item => item.id === revealId);
          if (queueItem) {
            queueItem.isInViewport = entry.isIntersecting;
            if (entry.isIntersecting) {
              this.processRevealQueue();
            }
          }
        }
      });
    }, {
      threshold: 0.5,
      rootMargin: '0px'
    });
  }

  populateLessonSelector() {
    const select = document.getElementById('lesson-select');
    LESSONS.forEach(lesson => {
      const option = document.createElement('option');
      option.value = lesson.file;
      option.textContent = lesson.name;
      select.appendChild(option);
    });

    if (LESSONS.length > 0) {
      select.value = LESSONS[0].file;
      this.loadLesson(LESSONS[0].file);
    }
  }

  async loadLesson(filename) {
    try {
      const response = await fetch(`lessons/${filename}`);
      if (!response.ok) throw new Error('Failed to load lesson');
      const markdown = await response.text();

      const switchingLesson = this.currentLessonFile && this.currentLessonFile !== filename;
      this.currentLessonFile = filename;

      this.loadProgress();
      this.loadRevealedBlocks();

      this.parseAndRender(markdown);

      // Check if we should show welcome back modal (has saved progress)
      const savedProgress = localStorage.getItem(`lessonProgress_${filename}`);
      const hasProgress = savedProgress && JSON.parse(savedProgress).completedStages?.length > 0;

      if (hasProgress) {
        // Show welcome back modal, then restore scroll based on user action
        await this.showWelcomeBackModal();
        this.restoreScrollPosition();
      } else {
        // No saved progress, just restore scroll (will be at top)
        this.restoreScrollPosition();

        if (this.completedStages.size > 0) {
          this.showProgressNotification(
            `<strong>Progress Restored:</strong> You've completed ${this.completedStages.size} stage${this.completedStages.size === 1 ? '' : 's'} in this lesson.`
          );
        } else if (switchingLesson) {
          this.showProgressNotification(
            `<strong>New Lesson:</strong> Starting fresh - no saved progress for this lesson yet.`
          );
        } else {
          this.showProgressNotification(
            `<strong>Lesson Loaded:</strong> Ready to begin! Complete stages to track your progress.`
          );
        }
      }
    } catch (error) {
      console.error('Error loading lesson:', error);
      document.getElementById('lesson-content').innerHTML =
        '<p style="color: var(--reveal-timer); text-align: center; padding: 2rem;">Failed to load lesson. Make sure the markdown file is in the same directory.</p>';
    }
  }

  parseAndRender(markdown) {
    const stageRegex = /##\s+\*\*Stage\s+(\d+):[^*]+\*\*/gi;
    const matches = [...markdown.matchAll(stageRegex)];
    
    this.stages = [];
    
    for (let i = 0; i < matches.length; i++) {
      const start = matches[i].index;
      const end = i < matches.length - 1 ? matches[i + 1].index : markdown.length;
      const stageContent = markdown.substring(start, end);
      this.stages.push(stageContent);
    }

    if (this.stages.length === 0) {
      this.stages.push(markdown);
    }

    this.renderStages();
    this.updateProgress();
  }

  toggleResources() {
    const overlay = document.getElementById('resources-overlay');
    overlay.classList.toggle('hidden');
  }

  renderStages() {
    const container = document.getElementById('lesson-content');
    container.innerHTML = '';

    // Clear previous observers and queue
    this.observedElements.clear();
    this.revealQueue = [];
    this.currentActiveIndex = -1;

    this.stages.forEach((stageMarkdown, index) => {
      const stageEl = this.createStageElement(stageMarkdown, index);
      container.appendChild(stageEl);
    });

    // Setup observers after rendering
    setTimeout(() => {
      this.setupRevealObservers();
      this.processRevealQueue();
    }, 100);
  }

  createStageElement(markdown, index) {
    const stage = document.createElement('div');
    stage.className = 'stage';
    stage.id = `stage-${index}`;

    const isCompleted = this.completedStages.has(index);
    const isActive = index === this.currentStage;
    const isLocked = index > this.currentStage;

    if (isCompleted) stage.classList.add('completed');
    if (isLocked) stage.classList.add('locked');

    const titleMatch = markdown.match(/##\s+\*\*Stage\s+\d+:\s*([^*]+)\*\*/i) || 
                       markdown.match(/##\s+([^\n]+)/);
    const title = titleMatch ? titleMatch[1].trim() : `Stage ${index + 1}`;

    const header = document.createElement('div');
    header.className = 'stage-header';
    if (isCompleted) header.classList.add('collapsed');
    
    header.innerHTML = `
      <h2>${title}</h2>
      <div class="stage-status">
        <span class="stage-badge ${isCompleted ? 'completed' : isActive ? 'active' : 'locked'}">
          ${isCompleted ? 'Completed' : isActive ? 'Active' : 'Locked'}
        </span>
        <svg class="collapse-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>
      </div>
    `;

    const content = document.createElement('div');
    content.className = 'stage-content';
    if (isCompleted) content.classList.add('collapsed');

    let html = marked.parse(markdown);
    html = this.processCodeBlocks(html, index);
    html = html.replace(/<code>([^<]+)<\/code>/g, '<code class="inline-code">$1</code>');
    html = html.replace(/<p><strong>Checkpoint[^:]*:<\/strong>/gi, '<div class="checkpoint"><p><strong>Checkpoint:</strong>');
    html = html.replace(/(<div class="checkpoint">[\s\S]*?)<\/p>/i, '$1</p></div>');

    content.innerHTML = html;

    if (isActive && !isCompleted) {
      const completeBtn = document.createElement('button');
      completeBtn.className = 'complete-stage-btn';
      completeBtn.textContent = 'Complete This Stage';
      completeBtn.onclick = () => this.completeStage(index);
      content.appendChild(completeBtn);
    }

    header.onclick = () => {
      if (isCompleted || isActive) {
        header.classList.toggle('collapsed');
        content.classList.toggle('collapsed');
      }
    };

    stage.appendChild(header);
    stage.appendChild(content);

    setTimeout(() => {
      stage.querySelectorAll('pre code').forEach(block => {
        hljs.highlightElement(block);
      });
    }, 0);

    return stage;
  }

  processCodeBlocks(html, stageIndex) {
    let blockCounter = 0;
    
    return html.replace(/<pre><code([^>]*)>([\s\S]*?)<\/code><\/pre>/g, (match, attrs, code) => {
      blockCounter++;
      const blockId = `code-${stageIndex}-${blockCounter}`;
      
      // Check for both raw and HTML-encoded versions of the marker
      const rawMarker = PARTIAL_REVEAL_MARKER;
      const encodedMarker = rawMarker.replace(/</g, '&lt;').replace(/>/g, '&gt;');
      
      const hasRawMarker = code.includes(rawMarker);
      const hasEncodedMarker = code.includes(encodedMarker);
      const hasPartialReveal = hasRawMarker || hasEncodedMarker;
      
      // Copy button HTML
      const copyBtnHtml = `
        <div class="code-toolbar">
          <button class="copy-btn" onclick="window.lessonViewer.copyCodeBlock('${blockId}')" title="Copy to clipboard">
            <svg class="copy-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            <span class="copy-text">Copy</span>
          </button>
        </div>
      `;
      
      if (hasPartialReveal) {
        // Use whichever marker is present
        const marker = hasRawMarker ? rawMarker : encodedMarker;
        const parts = code.split(marker);
        const visiblePart = parts[0].trim();
        const hiddenPart = parts[1] ? parts[1].trim() : '';
        
        const revealId = `reveal-${stageIndex}-${blockCounter}`;
        const isAlreadyRevealed = this.revealedBlockIds.has(revealId);
        
        // Add to queue
        const queueItem = {
          id: revealId,
          visiblePart: visiblePart,
          hiddenPart: hiddenPart,
          state: isAlreadyRevealed ? 'revealed' : 'waiting',
          remainingTime: REVEAL_DELAY,
          timerId: null,
          isInViewport: false,
          stageIndex: stageIndex,
          blockCounter: blockCounter
        };
        
        this.revealQueue.push(queueItem);
        
        // If already revealed, show only the solution (hidden part)
        if (isAlreadyRevealed) {
          return `
            <div class="code-block-wrapper partial-reveal-wrapper revealed" data-reveal-id="${revealId}" id="${blockId}">
              <div class="partial-reveal-header">
                <span class="partial-reveal-label">Solution</span>
                <button class="reveal-btn revealed" disabled>
                  <svg class="timer-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20 6 9 17 4 12"></polyline>
                  </svg>
                  Revealed
                </button>
              </div>
              <pre><code${attrs}>${hiddenPart}</code></pre>
              ${copyBtnHtml}
            </div>
          `;
        }
        
        // Show partial content with reveal button
        return `
          <div class="code-block-wrapper partial-reveal-wrapper" data-reveal-id="${revealId}" id="${blockId}">
            <div class="partial-reveal-header">
              <span class="partial-reveal-label">Fill in the blanks</span>
              <button class="reveal-btn" id="reveal-btn-${revealId}" onclick="window.lessonViewer.handleRevealClick('${revealId}')" disabled>
                <svg class="timer-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="12" cy="12" r="10"></circle>
                  <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                <span class="reveal-btn-text">Reveal</span>
              </button>
            </div>
            <pre><code${attrs}>${visiblePart}</code></pre>
            <div class="hidden-code" style="display: none;">${hiddenPart}</div>
            ${copyBtnHtml}
          </div>
        `;
      }

      // Regular code block - wrap with copy toolbar
      return `
        <div class="code-block-wrapper" id="${blockId}">
          <pre><code${attrs}>${code}</code></pre>
          ${copyBtnHtml}
        </div>
      `;
    });
  }

  setupRevealObservers() {
    this.revealQueue.forEach(queueItem => {
      if (queueItem.state !== 'revealed') {
        const element = document.querySelector(`[data-reveal-id="${queueItem.id}"]`);
        if (element && !this.observedElements.has(queueItem.id)) {
          this.observer.observe(element);
          this.observedElements.add(queueItem.id);
        }
      }
    });
  }

  processRevealQueue() {
    // Find the first unrevealed item
    const firstUnrevealedIndex = this.revealQueue.findIndex(item => item.state !== 'revealed');
    
    if (firstUnrevealedIndex === -1) return;
    
    const activeItem = this.revealQueue[firstUnrevealedIndex];
    
    // Only start timer if this is the first unrevealed and it's in viewport
    if (activeItem.state === 'waiting' && activeItem.isInViewport) {
      this.startRevealTimer(activeItem);
    }
    
    // Update all button states
    this.updateRevealButtonStates();
  }

  startRevealTimer(queueItem) {
    queueItem.state = 'counting';
    
    const updateButton = () => {
      const btn = document.getElementById(`reveal-btn-${queueItem.id}`);
      if (!btn) return;
      
      // Update to show timer is active (slightly different visual state)
      btn.classList.add('timer-active');
    };
    
    updateButton();
    
    queueItem.timerId = setInterval(() => {
      queueItem.remainingTime--;
      
      if (queueItem.remainingTime <= 0) {
        clearInterval(queueItem.timerId);
        queueItem.timerId = null;
        queueItem.state = 'ready';
        this.updateRevealButtonStates();
      }
    }, 1000);
  }

  updateRevealButtonStates() {
    this.revealQueue.forEach((item, index) => {
      if (item.state === 'revealed') return;
      
      const btn = document.getElementById(`reveal-btn-${item.id}`);
      if (!btn) return;
      
      const btnText = btn.querySelector('.reveal-btn-text');
      const firstUnrevealedIndex = this.revealQueue.findIndex(i => i.state !== 'revealed');
      const isFirstUnrevealed = index === firstUnrevealedIndex;
      
      if (item.state === 'waiting') {
        // Not the active block yet
        btn.disabled = true;
        btn.classList.remove('timer-active');
        if (btnText) btnText.textContent = 'Reveal';
      } else if (item.state === 'counting') {
        // Timer is running
        btn.disabled = true;
        btn.classList.add('timer-active');
        if (btnText) btnText.textContent = 'Reveal';
      } else if (item.state === 'ready') {
        // Timer complete, ready to reveal
        btn.disabled = false;
        btn.classList.remove('timer-active');
        if (btnText) btnText.textContent = 'Reveal';
      }
    });
  }

  handleRevealClick(revealId) {
    const queueItem = this.revealQueue.find(item => item.id === revealId);
    if (!queueItem || queueItem.state !== 'ready') return;
    
    // Mark as revealed
    queueItem.state = 'revealed';
    this.revealedBlockIds.add(revealId);
    this.saveRevealedBlocks();
    
    // Update UI
    const wrapper = document.querySelector(`[data-reveal-id="${revealId}"]`);
    if (wrapper) {
      const pre = wrapper.querySelector('pre');
      const hiddenCode = wrapper.querySelector('.hidden-code');
      const btn = document.getElementById(`reveal-btn-${revealId}`);
      const label = wrapper.querySelector('.partial-reveal-label');
      
      if (pre && hiddenCode) {
        const code = pre.querySelector('code');
        if (code) {
          // Replace visible code with hidden (solution) code
          code.innerHTML = hiddenCode.innerHTML;
          hljs.highlightElement(code);
        }
      }
      
      if (btn) {
        btn.disabled = true;
        btn.classList.remove('timer-active');
        btn.innerHTML = `
          <svg class="timer-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
          Revealed
        `;
      }
      
      if (label) {
        label.textContent = 'Solution';
      }
      
      wrapper.classList.add('revealed');
    }
    
    // Process queue to start timer for next block
    this.processRevealQueue();
  }

  async copyCodeBlock(blockId) {
    const wrapper = document.getElementById(blockId);
    if (!wrapper) return;
    
    // Get the code element - for partial reveal blocks, it's in the pre tag
    const codeElement = wrapper.querySelector('pre code');
    if (!codeElement) return;
    
    // Get text content (decodes HTML entities)
    const codeText = codeElement.textContent || codeElement.innerText;
    
    try {
      await navigator.clipboard.writeText(codeText);
      
      // Show "Copied!" feedback
      const copyBtn = wrapper.querySelector('.copy-btn');
      const copyText = wrapper.querySelector('.copy-text');
      
      if (copyBtn && copyText) {
        const originalText = copyText.textContent;
        copyBtn.classList.add('copied');
        copyText.textContent = 'Copied!';
        
        setTimeout(() => {
          copyBtn.classList.remove('copied');
          copyText.textContent = originalText;
        }, 2000);
      }
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }

  saveRevealedBlocks() {
    if (!this.currentLessonFile) return;
    
    const revealedArray = Array.from(this.revealedBlockIds);
    localStorage.setItem(
      `revealedBlocks_${this.currentLessonFile}`,
      JSON.stringify(revealedArray)
    );
  }

  loadRevealedBlocks() {
    if (!this.currentLessonFile) return;
    
    const saved = localStorage.getItem(`revealedBlocks_${this.currentLessonFile}`);
    if (saved) {
      try {
        const revealedArray = JSON.parse(saved);
        this.revealedBlockIds = new Set(revealedArray);
      } catch (e) {
        console.error('Error loading revealed blocks:', e);
        this.revealedBlockIds = new Set();
      }
    } else {
      this.revealedBlockIds = new Set();
    }
  }

  completeStage(index) {
    this.completedStages.add(index);
    this.currentStage = Math.min(index + 1, this.stages.length - 1);
    this.saveProgress();
    this.saveScrollPosition(this.currentLessonFile);
    this.showCelebration(index);

    setTimeout(() => {
      this.renderStages();
      this.updateProgress();

      if (index + 1 < this.stages.length) {
        document.getElementById(`stage-${index + 1}`).scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    }, 2000);
  }

  showCelebration(stageIndex) {
    const overlay = document.createElement('div');
    overlay.className = 'overlay';
    
    const celebration = document.createElement('div');
    celebration.className = 'celebration';
    celebration.innerHTML = `
      <h3>🎉 Stage ${stageIndex + 1} Complete!</h3>
      <p>Great work! Keep going!</p>
    `;

    document.body.appendChild(overlay);
    document.body.appendChild(celebration);

    for (let i = 0; i < 50; i++) {
      setTimeout(() => {
        const confetti = document.createElement('div');
        confetti.className = 'confetti';
        confetti.style.left = Math.random() * 100 + 'vw';
        confetti.style.background = ['#4af2a1', '#2dd4bf', '#f59e0b'][Math.floor(Math.random() * 3)];
        confetti.style.animationDelay = Math.random() * 0.5 + 's';
        document.body.appendChild(confetti);
        setTimeout(() => confetti.remove(), 3000);
      }, i * 20);
    }

    setTimeout(() => {
      overlay.remove();
      celebration.remove();
    }, 2000);
  }

  updateProgress() {
    const completed = this.completedStages.size;
    const total = this.stages.length;
    const percentage = total > 0 ? (completed / total) * 100 : 0;

    document.getElementById('progress-fill').style.width = percentage + '%';
    document.getElementById('progress-label').textContent = `${Math.round(percentage)}%`;
  }

  saveProgress() {
    if (!this.currentLessonFile) return;
    
    const progressData = {
      currentStage: this.currentStage,
      completedStages: Array.from(this.completedStages),
      lastUpdated: new Date().toISOString()
    };
    
    localStorage.setItem(
      `lessonProgress_${this.currentLessonFile}`,
      JSON.stringify(progressData)
    );
  }

  loadProgress() {
    if (!this.currentLessonFile) return;

    const saved = localStorage.getItem(`lessonProgress_${this.currentLessonFile}`);
    if (saved) {
      try {
        const data = JSON.parse(saved);
        this.currentStage = data.currentStage || 0;
        this.completedStages = new Set(data.completedStages || []);
      } catch (e) {
        console.error('Error loading progress:', e);
        this.currentStage = 0;
        this.completedStages = new Set();
      }
    } else {
      this.currentStage = 0;
      this.completedStages = new Set();
    }
  }

  /**
   * Save current scroll position to localStorage
   * @param {string} filename - The current lesson filename
   */
  saveScrollPosition(filename) {
    if (!filename) return;

    const scrollData = {
      scrollY: window.scrollY,
      timestamp: new Date().toISOString()
    };

    // Find nearest stage header above current scroll position
    const stages = document.querySelectorAll('.stage');
    let nearestStageId = null;
    
    stages.forEach((stage, index) => {
      const stageTop = stage.offsetTop;
      if (stageTop <= scrollY + 100) {  // 100px buffer for header
        nearestStageId = `stage-${index}`;
        scrollData.nearestStageIndex = index;
      }
    });

    scrollData.nearestStageId = nearestStageId;

    try {
      localStorage.setItem(
        `scrollPosition_${filename}`,
        JSON.stringify(scrollData)
      );
    } catch (e) {
      console.warn('Failed to save scroll position:', e);
    }
  }

  /**
   * Load saved scroll position from localStorage
   * @param {string} filename - The lesson filename
   * @returns {{scrollY: number, nearestStageId: string, nearestStageIndex: number}|null}
   */
  loadScrollPosition(filename) {
    if (!filename) return null;

    const saved = localStorage.getItem(`scrollPosition_${filename}`);
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        console.warn('Failed to parse scroll position:', e);
        return null;
      }
    }
    return null;
  }

  /**
   * Restore scroll position with smooth animation
   */
  restoreScrollPosition() {
    const scrollData = this.loadScrollPosition(this.currentLessonFile);
    if (!scrollData) return;

    // Check for reduced motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Prioritize stage ID over pixel position for accuracy
    if (scrollData.nearestStageId) {
      const stageElement = document.getElementById(scrollData.nearestStageId);
      if (stageElement) {
        stageElement.scrollIntoView({
          behavior: prefersReducedMotion ? 'auto' : 'smooth',
          block: 'start'
        });
        return;
      }
    }

    // Fallback to pixel position
    const targetY = scrollData.scrollY || 0;
    if (targetY > 0) {
      window.scrollTo({
        top: targetY,
        behavior: prefersReducedMotion ? 'auto' : 'smooth'
      });
    }
  }

  showProgressNotification(message) {
    const notification = document.getElementById('progress-notification');
    const messageEl = document.getElementById('progress-message');
    
    messageEl.innerHTML = message;
    notification.classList.remove('hidden');
    
    setTimeout(() => {
      if (!notification.classList.contains('hidden')) {
        notification.classList.add('hidden');
      }
    }, 10000);
  }

  dismissNotification() {
    document.getElementById('progress-notification').classList.add('hidden');
  }

  confirmResetProgress() {
    const overlay = document.createElement('div');
    overlay.className = 'overlay';
    
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
      <h3>⚠️ Reset Progress?</h3>
      <p>This will clear all your completed stages, revealed solutions, and start this lesson from the beginning. This action cannot be undone.</p>
      <div class="modal-actions">
        <button class="modal-btn modal-btn-cancel" onclick="window.lessonViewer.closeModal()">
          Cancel
        </button>
        <button class="modal-btn modal-btn-confirm" onclick="window.lessonViewer.resetProgress()">
          Reset Progress
        </button>
      </div>
    `;
    
    document.body.appendChild(overlay);
    document.body.appendChild(modal);
    
    this.currentModal = { overlay, modal };
  }

  closeModal() {
    if (this.currentModal) {
      this.currentModal.overlay.remove();
      this.currentModal.modal.remove();
      this.currentModal = null;
    }
  }

  resetProgress() {
    if (!this.currentLessonFile) return;

    localStorage.removeItem(`lessonProgress_${this.currentLessonFile}`);
    localStorage.removeItem(`revealedBlocks_${this.currentLessonFile}`);
    localStorage.removeItem(`scrollPosition_${this.currentLessonFile}`);
    localStorage.removeItem(`skipWelcomeModal_${this.currentLessonFile}`);

    this.currentStage = 0;
    this.completedStages = new Set();
    this.revealedBlockIds = new Set();

    this.closeModal();
    this.loadLesson(this.currentLessonFile);

    this.showProgressNotification(
      `<strong>Progress Reset:</strong> Starting fresh - all progress has been cleared.`
    );
  }

  /**
   * Show welcome back modal for returning users
   * @returns {Promise<'continue'|'startOver'>} User's choice
   */
  showWelcomeBackModal() {
    return new Promise((resolve) => {
      // Store the currently focused element for focus restoration
      const previouslyFocused = document.activeElement;

      // Check if should skip modal (show toast instead)
      const skipPreference = localStorage.getItem(`skipWelcomeModal_${this.currentLessonFile}`);
      if (skipPreference === 'true') {
        this.showToastNotification('Progress restored. Scroll to last position...');
        resolve('continue');
        return;
      }

      // Check if there's saved progress
      const savedProgress = localStorage.getItem(`lessonProgress_${this.currentLessonFile}`);
      if (!savedProgress) {
        resolve('continue');
        return;
      }

      try {
        const progressData = JSON.parse(savedProgress);
        const completedCount = progressData.completedStages ? progressData.completedStages.length : 0;

        if (completedCount === 0) {
          resolve('continue');
          return;
        }

        // Populate modal content
        const summaryEl = document.getElementById('welcome-back-summary');
        const stagesEl = document.getElementById('welcome-back-stages');
        const positionEl = document.getElementById('welcome-back-position');
        const checkboxEl = document.getElementById('welcome-back-skip-checkbox');

        // Summary text
        summaryEl.textContent = `You've completed ${completedCount} of ${this.stages.length} stage${completedCount === 1 ? '' : 's'} in this lesson. Great work!`;

        // List completed stages
        stagesEl.innerHTML = '';
        progressData.completedStages.forEach(stageIndex => {
          if (stageIndex < this.stages.length) {
            const stageTitle = this.extractStageTitle(this.stages[stageIndex]);
            const stageItem = document.createElement('div');
            stageItem.className = 'welcome-back-stage-item';
            stageItem.innerHTML = `
              <svg class="welcome-back-stage-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"></polyline>
              </svg>
              <span class="welcome-back-stage-title">${stageTitle}</span>
            `;
            stagesEl.appendChild(stageItem);
          }
        });

        // Last viewed position - defensive check for empty array
        const lastStageIndex = progressData.completedStages.length > 0
          ? Math.max(...progressData.completedStages)
          : 0;
        const lastStageTitle = this.extractStageTitle(this.stages[lastStageIndex]);
        positionEl.innerHTML = `<strong>📍 Last viewed:</strong> ${lastStageTitle}`;

        // Reset checkbox
        checkboxEl.checked = false;

        // Show modal
        const overlay = document.getElementById('welcome-back-overlay');
        overlay.classList.remove('hidden');

        // Setup button handlers
        const continueBtn = document.getElementById('welcome-back-continue-btn');
        const startOverBtn = document.getElementById('welcome-back-start-over-btn');

        // Clone buttons to remove old event listeners
        const newContinueBtn = continueBtn.cloneNode(true);
        const newStartOverBtn = startOverBtn.cloneNode(true);
        continueBtn.parentNode.replaceChild(newContinueBtn, continueBtn);
        startOverBtn.parentNode.replaceChild(newStartOverBtn, startOverBtn);

        // Continue button handler
        newContinueBtn.addEventListener('click', () => {
          const skipChecked = checkboxEl.checked;
          if (skipChecked) {
            localStorage.setItem(`skipWelcomeModal_${this.currentLessonFile}`, 'true');
          }
          this.closeWelcomeBackModal(previouslyFocused);
          resolve('continue');
        });

        // Start Over button handler
        newStartOverBtn.addEventListener('click', () => {
          this.closeWelcomeBackModal();
          this.confirmStartOver().then(confirmed => {
            if (confirmed) {
              this.resetProgress();
              resolve('startOver');
            } else {
              // User cancelled, show modal again
              this.showWelcomeBackModal().then(resolve);
            }
          });
        });

        // Backdrop click handler
        overlay.addEventListener('click', (e) => {
          if (e.target === overlay) {
            this.closeWelcomeBackModal(previouslyFocused);
            resolve('continue');
          }
        });

        // Escape key handler
        const handleEscape = (e) => {
          if (e.key === 'Escape') {
            document.removeEventListener('keydown', handleEscape);
            this.closeWelcomeBackModal(previouslyFocused);
            resolve('continue');
          }
        };
        document.addEventListener('keydown', handleEscape);

        // Focus trap
        this.trapFocus(overlay);

      } catch (e) {
        console.error('Error showing welcome back modal:', e);
        resolve('continue');
      }
    });
  }

  /**
   * Extract stage title from markdown content
   * @param {string} markdown - The stage markdown
   * @returns {string} The stage title
   */
  extractStageTitle(markdown) {
    const titleMatch = markdown.match(/##\s+\*\*Stage\s+\d+:\s*([^*]+)\*\*/i) ||
                       markdown.match(/##\s+([^\n]+)/);
    return titleMatch ? titleMatch[1].trim() : 'Unknown Stage';
  }

  /**
   * Close welcome back modal
   * @param {HTMLElement|null} previouslyFocused - Element to restore focus to
   */
  closeWelcomeBackModal(previouslyFocused = null) {
    const overlay = document.getElementById('welcome-back-overlay');
    if (overlay) {
      overlay.classList.add('hidden');
    }
    // Restore focus to the previously focused element
    if (previouslyFocused && previouslyFocused.focus) {
      previouslyFocused.focus();
    }
  }

  /**
   * Confirm start over action
   * @returns {Promise<boolean>} User's confirmation
   */
  confirmStartOver() {
    return new Promise((resolve) => {
      const overlay = document.createElement('div');
      overlay.className = 'overlay';

      const modal = document.createElement('div');
      modal.className = 'modal';
      modal.innerHTML = `
        <h3>⚠️ Start Over?</h3>
        <p>This will clear all your progress for this lesson and start from the beginning. This action cannot be undone.</p>
        <div class="modal-actions">
          <button class="modal-btn modal-btn-cancel" id="start-over-cancel-btn">Cancel</button>
          <button class="modal-btn modal-btn-confirm" id="start-over-confirm-btn">Start Over</button>
        </div>
      `;

      document.body.appendChild(overlay);
      document.body.appendChild(modal);

      const cancelBtn = document.getElementById('start-over-cancel-btn');
      const confirmBtn = document.getElementById('start-over-confirm-btn');

      const cleanup = () => {
        overlay.remove();
        modal.remove();
      };

      cancelBtn.addEventListener('click', () => {
        cleanup();
        resolve(false);
      });

      confirmBtn.addEventListener('click', () => {
        cleanup();
        resolve(true);
      });

      // Escape key handler
      const handleEscape = (e) => {
        if (e.key === 'Escape') {
          document.removeEventListener('keydown', handleEscape);
          cleanup();
          resolve(false);
        }
      };
      document.addEventListener('keydown', handleEscape);
    });
  }

  /**
   * Trap focus within a container element for accessibility
   * @param {HTMLElement} container - The container to trap focus within
   */
  trapFocus(container) {
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstFocusable = focusableElements[0];
    const lastFocusable = focusableElements[focusableElements.length - 1];

    // Focus first element
    setTimeout(() => firstFocusable.focus(), 100);

    const handleTabKey = (e) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstFocusable) {
          e.preventDefault();
          lastFocusable.focus();
        }
      } else {
        if (document.activeElement === lastFocusable) {
          e.preventDefault();
          firstFocusable.focus();
        }
      }
    };

    container.addEventListener('keydown', handleTabKey);
  }

  /**
   * Show a brief toast notification
   * @param {string} message - The message to display
   * @param {number} duration - Duration in milliseconds before auto-dismiss
   */
  showToastNotification(message, duration = 3000) {
    // Remove existing toast if any
    const existingToast = document.querySelector('.toast-notification');
    if (existingToast) {
      existingToast.remove();
    }

    const toast = document.createElement('div');
    toast.className = 'toast-notification';
    toast.innerHTML = `
      <svg class="toast-notification-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"/>
      </svg>
      <span class="toast-notification-text">${message}</span>
    `;

    document.body.appendChild(toast);

    // Trigger animation
    requestAnimationFrame(() => {
      toast.classList.add('show');
    });

    // Auto-dismiss
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 400);
    }, duration);
  }
}

// Initialize when DOM is ready
window.lessonViewer = new LessonViewer();
