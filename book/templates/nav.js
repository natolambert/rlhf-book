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
          <li><a href="https://rlhfbook.com">Home</a> / <a href="https://github.com/natolambert/rlhf-book">GitHub</a> / <a href="https://discord.gg/yz5AwK4gBR">Discord</a></li>
          <li><a href="https://rlhfbook.com/book.pdf">PDF</a> / <a href="https://arxiv.org/abs/2504.12501">Arxiv</a> / <a href="https://rlhfbook.com/book.epub">EPUB</a> / <a href="https://rlhfbook.com/book.kindle.epub">Kindle</a></li>
          <li>Order: <a href="https://hubs.la/Q03TsMBq0">Manning</a>, <a href="https://amzn.to/4cwCDJQ">Amazon</a></li>
        </ul>
        <h3>Resources</h3>
        <ul>
          <li><a href="https://rlhfbook.com/rl-cheatsheet">RL Cheatsheet</a></li>
          <li><a href="https://rlhfbook.com/library">Compare Model Completions</a></li>
          <li><a href="https://rlhfbook.com/course">Accompanying Course</a></li>
        </ul>
      </div>

      <div class="section">
        <h3>Introductions</h3>
        <ol start="1">
          <li><a href="https://rlhfbook.com/c/01-introduction">Introduction</a></li>
          <li><a href="https://rlhfbook.com/c/02-related-works">A Tiny History of RLHF</a></li>
          <li><a href="https://rlhfbook.com/c/03-training-overview">Training Overview</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Core Training Pipeline</h3>
        <ol start="4">
          <li><a href="https://rlhfbook.com/c/04-instruction-tuning">Instruction Fine-Tuning</a></li>
          <li><a href="https://rlhfbook.com/c/05-reward-models">Reward Modeling</a> [<a href="https://github.com/natolambert/rlhf-book/tree/main/code/reward_models">code</a>]</li>
          <li><a href="https://rlhfbook.com/c/06-policy-gradients">Reinforcement Learning</a> [<a href="https://github.com/natolambert/rlhf-book/tree/main/code/policy_gradients">code</a>]</li>
          <li><a href="https://rlhfbook.com/c/07-reasoning">Reasoning and Inference-Time Scaling</a></li>
          <li><a href="https://rlhfbook.com/c/08-direct-alignment">Direct-Alignment Algorithms</a> [<a href="https://github.com/natolambert/rlhf-book/tree/main/code/direct_alignment">code</a>]</li>
          <li><a href="https://rlhfbook.com/c/09-rejection-sampling">Rejection Sampling</a> [<a href="https://github.com/natolambert/rlhf-book/tree/main/code/rejection_sampling">code</a>]</li>
        </ol>
      </div>

      <div class="section">
        <h3>Data & Preferences</h3>
        <ol start="10">
          <li><a href="https://rlhfbook.com/c/10-preferences">The Nature of Preferences</a></li>
          <li><a href="https://rlhfbook.com/c/11-preference-data">Preference Data</a></li>
          <li><a href="https://rlhfbook.com/c/12-synthetic-data">Synthetic Data</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Practical Considerations</h3>
        <ol start="13">
          <li><a href="https://rlhfbook.com/c/13-tools">Tool Use and Function Calling</a></li>
          <li><a href="https://rlhfbook.com/c/14-over-optimization">Over-Optimization</a></li>
          <li><a href="https://rlhfbook.com/c/15-regularization">Regularization</a></li>
          <li><a href="https://rlhfbook.com/c/16-evaluation">Evaluation</a></li>
          <li><a href="https://rlhfbook.com/c/17-product">Model Character & Products</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Appendices</h3>
        <ol type="A" style="padding-left: 0; list-style-position: inside;">
          <li><a href="https://rlhfbook.com/c/appendix-a-definitions">Definitions</a></li>
          <li><a href="https://rlhfbook.com/c/appendix-b-style">Beyond "Just Style"</a></li>
          <li><a href="https://rlhfbook.com/c/appendix-c-practical">Practical Issues</a></li>
        </ol>
      </div>
    </nav>
    <div id="search"></div>
  </div>
</div>
      `;

      // Initialize Pagefind search if available
      var searchEl = this.querySelector('#search');
      if (searchEl && typeof PagefindUI !== 'undefined') {
        new PagefindUI({ element: searchEl, showImages: false });
      }

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
