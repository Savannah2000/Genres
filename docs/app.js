(() => {
  'use strict';
  const { examples, results } = window.GENRES_DATA;
  const relationships = {
    cs: { name: 'Communal Sharing', principle: 'Group unity and belonging', description: 'Relationships built on a shared group identity and a sense of belonging.' },
    em: { name: 'Equality Matching', principle: 'Balanced reciprocity', description: 'Relationships organized around equal contributions, turn-taking and reciprocal exchange.' },
    mp: { name: 'Market Pricing', principle: 'Proportional exchange', description: 'Relationships organized around proportional value, negotiated exchange and economic considerations.' },
    ar: { name: 'Authority Ranking', principle: 'Ordered social hierarchy', description: 'Relationships organized around differences in rank, authority and responsibility.' }
  };
  function activateTabs(group, selected) {
    group.querySelectorAll('[role="tab"]').forEach(button => {
      const active = button === selected;
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
    });
  }
  function renderExample(code, selected) {
    const sample = examples[code], relation = relationships[code];
    activateTabs(document.querySelector('.relationship-tabs'), selected);
    document.getElementById('example-panel').setAttribute('aria-labelledby', selected.id);
    const image = document.getElementById('example-image');
    image.src = ({ cs: "assets/example-cs-conversation.png", em: "assets/example-em-cohosts.png" })[code] || `assets/example-${code}.png`;
    image.alt = `Synthetic image paired with the ${relation.name} scenario: ${sample.scenario}`;
    document.getElementById('example-principle').textContent = relation.principle;
    document.getElementById('example-title').textContent = relation.name;
    document.getElementById('example-definition').textContent = relation.description;
    document.getElementById('example-names').textContent = sample.names.join(' & ');
    document.getElementById('example-age').textContent = sample.age.replace(/^at /, '').replace(/^a /, '').replace(/^an /, '');
    document.getElementById('example-scenario').textContent = sample.scenario;
    document.getElementById('example-prompt').textContent = sample.text_prompt;
    document.getElementById('example-source').href = `https://huggingface.co/datasets/Savannah-y7/Genres/blob/main/NEPs/genre_mf_${code}_query.jsonl`;
    document.querySelector('.example-copy details').open = false;
  }
  document.querySelectorAll('[data-relation]').forEach(button => button.addEventListener('click', () => renderExample(button.dataset.relation, button)));
  renderExample('cs', document.getElementById('relation-cs'));
  function renderResults(evaluator, selected) {
    activateTabs(document.querySelector('.evaluator-tabs'), selected);
    document.getElementById('results-panel').setAttribute('aria-labelledby', selected.id);
    const body = document.getElementById('results-body');
    body.replaceChildren(...results[evaluator].map(row => {
      const tr = document.createElement('tr'), th = document.createElement('th');
      th.scope = 'row'; th.textContent = row.model; tr.append(th);
      row.values.forEach(value => {
        const td = document.createElement('td');
        td.textContent = value; td.className = value.startsWith('-') ? 'negative' : 'positive'; tr.append(td);
      });
      return tr;
    }));
  }
  document.querySelectorAll('[data-evaluator]').forEach(button => button.addEventListener('click', () => renderResults(button.dataset.evaluator, button)));
  document.querySelectorAll('[role="tablist"]').forEach(group => group.addEventListener('keydown', event => {
    const tabs = [...group.querySelectorAll('[role="tab"]')];
    const index = tabs.indexOf(document.activeElement);
    if (index < 0) return;
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else return;
    event.preventDefault(); tabs[next].focus(); tabs[next].click();
  }));
  let statusTimer;
  function notify(message) {
    const status = document.getElementById('copy-status');
    clearTimeout(statusTimer); status.textContent = message; status.classList.add('visible');
    statusTimer = setTimeout(() => status.classList.remove('visible'), 2500);
  }
  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      try { await navigator.clipboard.writeText(text); return; } catch (_) { /* Try the local-file fallback. */ }
    }
    const area = document.createElement('textarea');
    area.value = text; area.style.cssText = 'position:fixed;left:-9999px;top:0';
    document.body.append(area); area.select();
    const copied = document.execCommand('copy'); area.remove();
    if (!copied) throw new Error('Clipboard is unavailable');
  }
  document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
    try {
      await copyText(document.getElementById(button.dataset.copy).textContent);
      const original = button.textContent; button.textContent = 'Copied!';
      notify(button.dataset.copy === 'bibtex' ? 'BibTeX copied to clipboard' : 'Dataset loading code copied');
      setTimeout(() => { button.textContent = original; }, 2200);
    } catch (_) { notify('Select the code and copy it with your keyboard.'); }
  }));
  const dialog = document.getElementById('figure-dialog');
  let trigger;
  document.querySelectorAll('[data-zoom]').forEach(button => button.addEventListener('click', () => {
    const image = document.getElementById('dialog-image');
    image.src = button.dataset.zoom; image.alt = button.querySelector('img').alt;
    document.getElementById('dialog-title').textContent = button.getAttribute('aria-label').replace(/^Enlarge /, '');
    trigger = button; dialog.showModal(); document.body.style.overflow = 'hidden';
  }));
  document.getElementById('close-dialog').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => { document.body.style.overflow = ''; if (trigger) trigger.focus(); });
  const links = [...document.querySelectorAll('.nav-links a')];
  const sections = links.map(link => document.querySelector(link.getAttribute('href')));
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting);
      if (!visible.length) return;
      const id = visible[visible.length - 1].target.id;
      links.forEach(link => {
        if (link.hash === '#' + id) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    }, { rootMargin: '-12% 0px -60% 0px', threshold: 0 });
    sections.forEach(section => observer.observe(section));
  }
})();
