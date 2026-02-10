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
    
    this.currentStage = 0;
    this.completedStages = new Set();
    this.revealedBlockIds = new Set();
    
    this.closeModal();
    this.loadLesson(this.currentLessonFile);
    
    this.showProgressNotification(
      `<strong>Progress Reset:</strong> Starting fresh - all progress has been cleared.`
    );
  }
}

// Initialize when DOM is ready
window.lessonViewer = new LessonViewer();
