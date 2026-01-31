class NavigationDropdown extends HTMLElement {
    constructor() {
      super();

      // Get the initial expanded state from the attribute, default to false
      const initialExpanded = this.getAttribute('expanded') === 'true';

      this.innerHTML = `
        <div>
          <button class="dropdown-button" aria-expanded="${initialExpanded}">
            <span><strong>Navigation</strong></span>
            <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M19 9l-7 7-7-7" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>

          <div class="dropdown-content${initialExpanded ? ' open' : ''}">
    <nav class="chapter-nav">
      <div class="section">
        <h3>Links</h3>
        <ul>
          <li><a href="https://rlhfbook.com">Home</a></li>
          <li><a href="https://github.com/natolambert/rlhf-book">GitHub Repository</a></li>
          <li><a href="https://rlhfbook.com/book.pdf">PDF</a> / <a href="https://arxiv.org/abs/2504.12501">Arxiv</a> / <a href="https://rlhfbook.com/book.epub">EPUB</a></li>
          <li><a href="https://hubs.la/Q03TsMBq0">Pre-order now!</a></li>
          <li><a href="https://rlhfbook.com/library">Library</a></li>
        </ul>
      </div>

      <div class="section">
        <h3>Introductions</h3>
        <ol start="1">
          <li><a href="https://rlhfbook.com/c/01-introduction">Introduction</a></li>
          <li><a href="https://rlhfbook.com/c/02-related-works">Key Related Works</a></li>
          <li><a href="https://rlhfbook.com/c/03-training-overview">Training Overview</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Core Training Pipeline</h3>
        <ol start="4">
          <li><a href="https://rlhfbook.com/c/04-instruction-tuning">Instruction Tuning</a></li>
          <li><a href="https://rlhfbook.com/c/05-reward-models">Reward Models</a></li>
          <li><a href="https://rlhfbook.com/c/06-policy-gradients">Reinforcement Learning</a></li>
          <li><a href="https://rlhfbook.com/c/07-reasoning">Reasoning</a></li>
          <li><a href="https://rlhfbook.com/c/08-direct-alignment">Direct Alignment</a></li>
          <li><a href="https://rlhfbook.com/c/09-rejection-sampling">Rejection Sampling</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Data & Preferences</h3>
        <ol start="10">
          <li><a href="https://rlhfbook.com/c/10-preferences">What are Preferences</a></li>
          <li><a href="https://rlhfbook.com/c/11-preference-data">Preference Data</a></li>
          <li><a href="https://rlhfbook.com/c/12-synthetic-data">Synthetic Data & CAI</a></li>
          <li><a href="https://rlhfbook.com/c/13-tools">Tool Use</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Practical Considerations</h3>
        <ol start="14">
          <li><a href="https://rlhfbook.com/c/14-over-optimization">Over-optimization</a></li>
          <li><a href="https://rlhfbook.com/c/15-regularization">Regularization</a></li>
          <li><a href="https://rlhfbook.com/c/16-evaluation">Evaluation</a></li>
          <li><a href="https://rlhfbook.com/c/17-product">Product & Character</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Appendices</h3>
        <ul>
          <li><a href="https://rlhfbook.com/c/appendix-a-definitions">A. Definitions</a></li>
          <li><a href="https://rlhfbook.com/c/appendix-b-style">B. Style & Information</a></li>
        </ul>
      </div>
    </nav>
  </div>
</div>
      `;

      // Set up click handler
      const button = this.querySelector('.dropdown-button');
      const content = this.querySelector('.dropdown-content');

      button.addEventListener('click', () => {
        const isExpanded = button.getAttribute('aria-expanded') === 'true';
        button.setAttribute('aria-expanded', !isExpanded);
        content.classList.toggle('open');
      });
    }

    // Add attribute change observer
    static get observedAttributes() {
      return ['expanded'];
    }

    attributeChangedCallback(name, oldValue, newValue) {
      if (name === 'expanded') {
        const button = this.querySelector('.dropdown-button');
        const content = this.querySelector('.dropdown-content');
        const isExpanded = newValue === 'true';

        if (button && content) {
          button.setAttribute('aria-expanded', isExpanded);
          content.classList.toggle('open', isExpanded);
        }
      }
    }
}

// Only define the component once
if (!customElements.get('navigation-dropdown')) {
  customElements.define('navigation-dropdown', NavigationDropdown);
}
