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
          <li><a href="https://rlhfbook.com/book.pdf">PDF</a> / <a href="https://arxiv.org/abs/2504.12501"> Arxiv </a></li>
          <li class="inactive">Order a copy (Soon)</li>
        </ul>
      </div>

      <div class="section">
        <h3>Introductions</h3>
        <ol start="1">
          <li><a href="https://rlhfbook.com/c/01-introduction.html">Introduction</a></li>
          <li><a href="https://rlhfbook.com/c/02-related-works.html">Seminal (Recent) Works</a></li>
          <li><a href="https://rlhfbook.com/c/03-setup.html">Definitions</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Problem Setup & Context</h3>
        <ol start="4">
          <li><a href="https://rlhfbook.com/c/04-optimization.html">Training Overview</a></li>
          <li><a href="https://rlhfbook.com/c/05-preferences.html">What are preferences?</a></li>
          <li><a href="https://rlhfbook.com/c/06-preference-data.html">Preference Data</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Optimization Tools</h3>
        <ol start="7">
          <li><a href="https://rlhfbook.com/c/07-reward-models.html">Reward Modeling</a></li>
          <li><a href="https://rlhfbook.com/c/08-regularization.html">Regularization</a></li>
          <li><a href="https://rlhfbook.com/c/09-instruction-tuning.html">Instruction Tuning</a></li>
          <li><a href="https://rlhfbook.com/c/10-rejection-sampling.html">Rejection Sampling</a></li>
          <li><a href="https://rlhfbook.com/c/11-policy-gradients.html">Policy Gradients</a></li>
          <li><a href="https://rlhfbook.com/c/12-direct-alignment.html">Direct Alignment Algorithms</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Advanced</h3>
        <ol start="13">
          <li><a href="https://rlhfbook.com/c/13-cai.html">Constitutional AI and AI Feedback</a></li>
          <li><a href="https://rlhfbook.com/c/14-reasoning.html">Reasoning and Reinforcement Finetuning</a></li>
          <li><a href="https://rlhfbook.com/c/15-synthetic.html">Synthetic Data</a></li>
          <li><a href="https://rlhfbook.com/c/16-evaluation.html">Evaluation</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Open Questions</h3>
        <ol start="17">
          <li><a href="https://rlhfbook.com/c/17-over-optimization.html">Over-optimization</a></li>
          <li><a href="https://rlhfbook.com/c/18-style.html">Style & Information</a></li>
          <li><a href="https://rlhfbook.com/c/19-character.html">Product, UX, Character, and Post-Training</a></li>
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
