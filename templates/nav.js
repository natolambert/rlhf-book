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
          <li><a href="https://rlhfbook.com/library">Example Model Completions</a></li>
        </ul>
      </div>

      <div class="section">
        <h3>Introductions</h3>
        <ol start="1">
          <li><a href="https://rlhfbook.com/c/01-introduction">Introduction</a></li>
          <li><a href="https://rlhfbook.com/c/02-related-works">Seminal (Recent) Works</a></li>
          <li><a href="https://rlhfbook.com/c/03-setup">Definitions</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Problem Setup & Context</h3>
        <ol start="4">
          <li><a href="https://rlhfbook.com/c/04-optimization">Training Overview</a></li>
          <li><a href="https://rlhfbook.com/c/05-preferences">What are preferences?</a></li>
          <li><a href="https://rlhfbook.com/c/06-preference-data">Preference Data</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Optimization Tools</h3>
        <ol start="7">
          <li><a href="https://rlhfbook.com/c/07-reward-models">Reward Modeling</a></li>
          <li><a href="https://rlhfbook.com/c/08-regularization">Regularization</a></li>
          <li><a href="https://rlhfbook.com/c/09-instruction-tuning">Instruction Tuning (i.e. Supervised Finetuning)</a></li>
          <li><a href="https://rlhfbook.com/c/10-rejection-sampling">Rejection Sampling</a></li>
          <li><a href="https://rlhfbook.com/c/11-policy-gradients">Reinforcement Learning (i.e. Policy Gradients)</a></li>
          <li><a href="https://rlhfbook.com/c/12-direct-alignment">Direct Alignment Algorithms</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Advanced</h3>
        <ol start="13">
          <li><a href="https://rlhfbook.com/c/13-cai">Constitutional AI and AI Feedback</a></li>
          <li><a href="https://rlhfbook.com/c/14-reasoning">Reasoning and Inference-time Scaling</a></li>
          <li><a href="https://rlhfbook.com/c/14.5-tools">Tool Use and Function Calling</a></li>
          <li><a href="https://rlhfbook.com/c/15-synthetic">Synthetic Data & Distillation</a></li>
          <li><a href="https://rlhfbook.com/c/16-evaluation">Evaluation</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Open Questions</h3>
        <ol start="18">
          <li><a href="https://rlhfbook.com/c/17-over-optimization">Over-optimization</a></li>
          <li><a href="https://rlhfbook.com/c/18-style">Style & Information</a></li>
          <li><a href="https://rlhfbook.com/c/19-character">Product, UX, Character, and Post-Training</a></li>
        </ol>
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
