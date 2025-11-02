(function () {
  const wrapTables = () => {
    const tables = document.querySelectorAll('table');
    tables.forEach((table) => {
      if (table.closest('.table-scroll')) {
        return;
      }
      if (
        table.classList.contains('table-wrap') ||
        table.classList.contains('wrap-table') ||
        table.dataset.tableBehavior === 'wrap' ||
        table.closest('.table-wrap')
      ) {
        return;
      }
      const wrapper = document.createElement('div');
      wrapper.className = 'table-scroll';
      const parent = table.parentNode;
      if (!parent) {
        return;
      }
      parent.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    });
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wrapTables, { once: true });
  } else {
    wrapTables();
  }
})();
