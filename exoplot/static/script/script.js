// ===========================================================
// Initialize Lucide Icons
// ===========================================================
lucide.createIcons();


// ===========================================================
// Stars Canvas Animation
// ===========================================================

const canvas = document.getElementById('stars-canvas');
const ctx = canvas.getContext('2d');
let stars = [];

// Resize canvas to full page and regenerate stars
function resizeCanvas() {
  const width = document.documentElement.scrollWidth;
  const height = document.documentElement.scrollHeight;

  canvas.width = width;
  canvas.height = height;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  generateStars();
}

// Generate random stars
function generateStars() {
  stars = [];
  const count = Math.floor(window.innerWidth / 5);
  for (let i = 0; i < count; i++) {
    stars.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      radius: Math.random() * 2.5 + 0.8,
      alpha: Math.random(),
      speed: Math.random() * 0.01 + 0.01,
    });
  }
}

// Animate stars with twinkling effect
function animateStars() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "white";

  stars.forEach((star) => {
    ctx.globalAlpha = star.alpha;
    ctx.beginPath();
    ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
    ctx.fill();

    star.alpha += star.speed;
    if (star.alpha >= 1 || star.alpha <= 0.2) star.speed *= -1;
  });

  ctx.globalAlpha = 1;
  requestAnimationFrame(animateStars);
}

// Handle window resize
window.addEventListener('resize', () => {
  resizeCanvas();
  generateStars();
});

resizeCanvas();
animateStars();


// ===========================================================
// Mission-Specific Form Handling
// ===========================================================

document.addEventListener("DOMContentLoaded", () => {
  const missionSelect = document.getElementById("mission");
  const missionSpecificFieldset = document.getElementById("mission-specific");
  const missionSpecificSections = missionSpecificFieldset.querySelectorAll("div[data-mission]");
  const form = document.querySelector(".discover-form");

  // Show/hide sections based on mission selection
  function toggleMissionSpecific() {
    const mission = missionSelect.value;
    let anyVisible = false;

    missionSpecificSections.forEach(section => {
      const applicable = section.dataset.mission.split(" ");
      if (mission && applicable.includes(mission)) {
        section.style.display = "";
        anyVisible = true;
      } else {
        section.style.display = "none";
      }
    });

    missionSpecificFieldset.style.display = anyVisible ? "" : "none";
  }

  toggleMissionSpecific();
  missionSelect.addEventListener("change", toggleMissionSpecific);

  // Clear old results & show loading on submit
  form.addEventListener("submit", () => {
    document.querySelector(".results-table")?.remove();
    document.querySelector(".results-heading")?.remove();
    const loading = document.getElementById("loading-message");
    if (loading) loading.style.display = "block";
  });
});


// ===========================================================
// Discover Form Submit (redundant listener for fallback?)
// ===========================================================

const discoverForm = document.querySelector(".discover-form");
if (discoverForm) {
  discoverForm.addEventListener("submit", () => {
    const loading = document.getElementById("loading-message");
    if (loading) loading.style.display = "block";
  });
}


// ===========================================================
// Tab Navigation Handler
// ===========================================================

function showTab(id) {
  const tabStrip = document.querySelector('.tab-strip');
  const tabs = tabStrip.querySelectorAll('.tab');
  const underline = tabStrip.querySelector('.tab-underline');

  const el = document.getElementById(id);
  if (!el || !tabs.length || !underline) return;

  // cacher tous les contenus
  document.querySelectorAll('.tab-content').forEach(content => {
    content.style.display = 'none';
  });

  // afficher le contenu actif
  el.style.display = '';

  const widthPercent = 100 / tabs.length;
  tabStrip.style.setProperty('--tab-underline-width', `${widthPercent}%`);
  underline.style.width = `${widthPercent}%`;

  tabs.forEach((tab, index) => {
    if (tab.dataset.tab === id) {
      tab.classList.add('active');
      underline.style.transform = `translateX(${index * 100}%)`;
    } else {
      tab.classList.remove('active');
    }
  });

  setTimeout(() => {
    el.querySelectorAll('.js-plotly-plot').forEach(div => {
      Plotly.Plots.resize(div);
    });
  }, 50);
}

// Initialiser au chargement
window.addEventListener('DOMContentLoaded', () => {
  const tabs = document.querySelectorAll('.tab-strip .tab');
  const tabStrip = document.querySelector('.tab-strip');
  const underline = document.querySelector('.tab-underline');
  if (!tabs.length || !tabStrip || !underline) return;

  const widthPercent = 100 / tabs.length;
  tabStrip.style.setProperty('--tab-underline-width', `${widthPercent}%`);
  underline.style.width = `${widthPercent}%`;

  const firstTab = tabs[0]?.dataset.tab;
  if (firstTab) showTab(firstTab);
});

// Initialiser la largeur de la barre au premier chargement
window.addEventListener('DOMContentLoaded', () => {
  const tabs = document.querySelectorAll('.tab-strip .tab');
  const tabStrip = document.querySelector('.tab-strip');
  const underline = document.querySelector('.tab-underline');
  if (!tabs.length || !tabStrip || !underline) return;

  const widthPercent = 100 / tabs.length;
  tabStrip.style.setProperty('--tab-underline-width', `${widthPercent}%`);
  underline.style.width = `${widthPercent}%`;

  // activer le premier onglet par défaut
  const firstTab = tabs[0]?.dataset.tab;
  if (firstTab) showTab(firstTab);
});


// ===========================================================
// Refold Form Submission
// ===========================================================

document.getElementById('refold-form').addEventListener('submit', (e) => {
  e.preventDefault();

  const form = e.target;
  const target = form.target.value;
  const row = form.row.value || 0;
  const harmonic = form.harmonic.value || 1;

  fetch(`/refold?target=${target}&row=${row}&harmonic=${harmonic}`)
    .then(res => res.text())
    .then(html => {
      const foldedFigure = document.getElementById('folded-figure');
      foldedFigure.innerHTML = html;

      // Execute scripts returned in response
      foldedFigure.querySelectorAll('script').forEach(oldScript => {
        const newScript = document.createElement('script');
        if (oldScript.src) newScript.src = oldScript.src;
        else newScript.textContent = oldScript.textContent;
        oldScript.parentNode.replaceChild(newScript, oldScript);
      });

      showTab('folded');

      setTimeout(() => {
        foldedFigure.querySelectorAll('.js-plotly-plot').forEach(div => {
          Plotly.Plots.resize(div);
        });
      }, 50);
    })
    .catch(err => {
      console.error("Refold failed:", err);
      document.getElementById('folded-figure').innerHTML =
        `<p style="color:red;">Error: ${err}</p>`;
    });
});


// ===========================================================
// MCMC Analysis
// ===========================================================

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById('mcmc-button');
  const loading = document.getElementById('mcmc-loading');
  const result = document.getElementById('mcmc-result');
  const bar = document.getElementById('mcmc-progress-bar');

  const lcDiv = document.getElementById('mcmc-lc');
  const paramsDiv = document.getElementById('mcmc-params');
  const traceDiv = document.getElementById('mcmc-trace');
  const cornerDiv = document.getElementById('mcmc-corner');

  if (!btn || !loading || !result || !bar) {
    console.warn("MCMC elements not found — probably not on the lightcurve page.");
    return;
  }

  // Helper to execute scripts in fetched HTML
  function executeScripts(container) {
    container.querySelectorAll("script").forEach(oldScript => {
      const newScript = document.createElement("script");
      if (oldScript.src) newScript.src = oldScript.src;
      else newScript.textContent = oldScript.textContent;

      for (const attr of oldScript.attributes) {
        newScript.setAttribute(attr.name, attr.value);
      }

      oldScript.parentNode.replaceChild(newScript, oldScript);
    });
  }

  // Start MCMC on button click
  btn.addEventListener("click", () => {
    btn.style.display = 'none';
    loading.style.display = '';
    bar.style.width = '0%';

    const target = document.querySelector('input[name="target"]').value;
    const row = document.querySelector('input[name="row"]')?.value || 0;
    const harmonic = document.querySelector('input[name="harmonic"]')?.value || 1;

    fetch(`/mcmc?target=${target}&row=${row}&harmonic=${harmonic}`)
      .then(res => res.json())
      .then(data => {
        if (!data.task_id) throw new Error("No task_id returned");
        pollProgress(data.task_id);
      })
      .catch(err => {
        loading.textContent = 'MCMC failed to start: ' + err;
      });
  });

  // Poll server for MCMC progress
  function pollProgress(taskId) {
    fetch(`/progress/${taskId}`)
      .then(res => res.json())
      .then(data => {
        if (data.progress !== undefined) {
          const pct = Math.min(100, Math.round(data.progress * 100));
          bar.style.width = `${pct}%`;

          if (data.progress >= 1) fetchResult(taskId);
          else setTimeout(() => pollProgress(taskId), 1000);
        } else {
          loading.textContent = 'Error: no progress info';
        }
      })
      .catch(err => {
        loading.textContent = 'Progress polling failed: ' + err;
      });
  }

  // Fetch and render final MCMC result
  function fetchResult(taskId) {
    fetch(`/result/${taskId}`)
      .then(res => {
        if (res.status === 202) {
          setTimeout(() => fetchResult(taskId), 1000);
          return;
        }
        return res.json();
      })
      .then(data => {
        if (!data) return;

        loading.style.display = 'none';
        result.style.display = '';

        lcDiv.innerHTML = data.lc_html;
        executeScripts(lcDiv);

        paramsDiv.innerHTML = data.param_html;

        traceDiv.innerHTML = data.trace_html;
        executeScripts(traceDiv);

        cornerDiv.innerHTML = data.corner_html;
        executeScripts(cornerDiv);

        showTab('mcmc');
      })
      .catch(err => {
        loading.textContent = 'Failed to fetch result: ' + err;
      });
  }
});





window.addEventListener("DOMContentLoaded", () => {
  const cards = document.querySelectorAll(".plot-type-card");
  const hiddenInput = document.getElementById("plot-type-selected");

  if (!cards.length || !hiddenInput) {
    console.warn("No plot-type cards or hidden input found.");
    return;
  }

  cards.forEach(card => {
    card.addEventListener("click", () => {
      cards.forEach(c => c.classList.remove("selected"));

      card.classList.add("selected");
      hiddenInput.value = card.dataset.plot;

      console.log("Plot type selected:", hiddenInput.value);
    });
  });
});