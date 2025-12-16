// Visualization context to keep sizes and shared config in one place
const vizCtx = {
  // Overall SVG sizes
  globeWidth: 800,
  globeHeight: 800,
  statsWidth: 800,
  statsHeight: 800,
  fightsWidth: 800,
  fightsHeight: 800,

  // Margins for stats SVG
  statsMargin: { top: 40, right: 30, bottom: 40, left: 20 },

  // Data cache
  fightersByCountry: {}, // { [countryName]: Fighter[] }
  fightsByCountry: {}, // { [countryName]: Fight[] }

  // D3 handles we may want to reuse
  globeSvg: null,
  statsSvg: null,
  fightsSvg: null,

  // Globe projection state (for zooming)
  globeProjection: null,
  globePath: null,
  globeBaseScale: null,
  globeZoom: 1,
  globePanX: 0,
  globePanY: 0,

  // Stats panel configuration
  maxFightersPerCountry: 8,
  maxFightsPerCountry: 5,
};

/**
 * Entry point called from body onload in index.html
 */
function createViz() {
  console.log("createViz updated !!!!!!!");
  setupLayout();
  loadDataAndRender();
}

/**
 * Create the page layout: two SVGs side by side inside #viz
 */
function setupLayout() {
  const container = d3.select("#viz");

  // Simple side‑by‑side layout using flexbox
  container
    .style("display", "flex")
    .style("flex-direction", "row")
    .style("align-items", "flex-start")
    .style("gap", "16px")
    .style("font-family", "system-ui, -apple-system, BlinkMacSystemFont, sans-serif");

  // Left: globe SVG + zoom slider
  const globeContainer = container
    .append("div")
    .style("flex", "0 0 auto");

  vizCtx.globeSvg = globeContainer
    .append("svg")
    .attr("id", "globe-svg")
    .attr("width", vizCtx.globeWidth)
    .attr("height", vizCtx.globeHeight);

  const zoomControls = globeContainer
    .append("div")
    .attr("class", "zoom-controls");

  zoomControls.append("label").text("Zoom:");

  const zoomValue = zoomControls
    .append("span")
    .attr("class", "zoom-value")
    .text(`${vizCtx.globeZoom.toFixed(2)}x`);

  zoomControls
    .append("input")
    .attr("type", "range")
    .attr("min", 0.6)
    .attr("max", 2.5)
    .attr("step", 0.05)
    .attr("value", vizCtx.globeZoom)
    .on("input", function () {
      vizCtx.globeZoom = +this.value;
      updateGlobeZoom();
      // Update the value display
      zoomValue.text(`${vizCtx.globeZoom.toFixed(2)}x`);
    });

  // Middle: stats SVG (fighters per country)
  vizCtx.statsSvg = container
    .append("div")
    .style("flex", "0 0 auto")
    .style("overflow", "visible")
    .append("svg")
    .attr("id", "stats-svg")
    .attr("width", vizCtx.statsWidth)
    .attr("height", vizCtx.statsHeight);

  // Right: fights SVG (title fights per country)
  vizCtx.fightsSvg = container
    .append("div")
    .style("flex", "0 0 auto")
    .style("overflow", "visible")
    .append("svg")
    .attr("id", "fights-svg")
    .attr("width", vizCtx.fightsWidth)
    .attr("height", vizCtx.fightsHeight);
}

/**
 * Load GeoJSON, fighter CSV and per-fight CSV, build the lookups once,
 * then render the visualizations.
 */
function loadDataAndRender() {
  Promise.all([
    d3.json("data/countries.geo.json"),
    d3.csv("data/pro_mma_fighters.csv", fighterRowParser),
    d3.csv("data/per_fight_data.csv", fightRowParser),
  ])
    .then(([worldGeo, fighters, fights]) => {
      buildFightersByCountry(fighters);
      buildFightsByCountry(fights);
      renderGlobe(worldGeo);
      renderStatsPanel(null); // initial empty state
      renderFightsPanel(null); // initial empty state
    })
    .catch((err) => {
      console.error("Error loading data:", err);
    });
}

/**
 * Parse a row from pro_mma_fighters.csv into a nicer typed object.
 */
function fighterRowParser(d) {
  const wins = d.wins ? +d.wins : 0;
  const wins_ko = d.wins_ko ? +d.wins_ko : 0;
  const wins_submission = d.wins_submission ? +d.wins_submission : 0;
  const wins_decision = d.wins_decision ? +d.wins_decision : 0;
  const rawWinsOther = d.wins_other ? +d.wins_other : 0;
  // Recompute wins_other so that wins = ko + sub + dec + other
  let wins_other = wins - (wins_ko + wins_submission + wins_decision);
  if (!Number.isFinite(wins_other) || wins_other < 0) {
    wins_other = rawWinsOther > 0 ? rawWinsOther : 0;
  }

  const lossess = d.lossess ? +d.lossess : 0;
  const losses_ko = d.losses_ko ? +d.losses_ko : 0;
  const losses_submission = d.losses_submission ? +d.losses_submission : 0;
  const losses_decision = d.losses_decision ? +d.losses_decision : 0;
  const rawLossesOther = d.losses_other ? +d.losses_other : 0;
  // Recompute losses_other so that losses = ko + sub + dec + other
  let losses_other = lossess - (losses_ko + losses_submission + losses_decision);
  if (!Number.isFinite(losses_other) || losses_other < 0) {
    losses_other = rawLossesOther > 0 ? rawLossesOther : 0;
  }

  return {
    url: d.url,
    fighter_name: d.fighter_name,
    nickname: d.nickname,
    birth_date: d.birth_date,
    age: d.age ? +d.age : null,
    death_date: d.death_date,
    location: d.location,
    country: d.country,
    height: d.height,
    weight: d.weight,
    association: d.association,
    weight_class: d.weight_class,
    wins,
    wins_ko,
    wins_submission,
    wins_decision,
    wins_other,
    // Note: "lossess" is misspelled in the CSV header
    lossess,
    losses_ko,
    losses_submission,
    losses_decision,
    losses_other,
  };
}

/**
 * Parse a row from per_fight_data.csv into a nicer typed object.
 * We keep this logic separate from fighter parsing for modularity.
 */
function fightRowParser(d) {
  return {
    R_fighter: d.R_fighter,
    B_fighter: d.B_fighter,
    R_odds: d.R_odds ? +d.R_odds : null,
    B_odds: d.B_odds ? +d.B_odds : null,
    date: d.date,
    location: d.location,
    country: d.country,
    Winner: d.Winner,
    title_bout: d.title_bout, // keep raw string; we'll filter later
    weight_class: d.weight_class,
    gender: d.gender,
  };
}

/**
 * Build a dictionary from country name -> list of fighters.
 * This is computed once and reused for every hover.
 */
function buildFightersByCountry(fighters) {
  const byCountry = {};

  fighters.forEach((f) => {
    const key = normalizeCountryNameForGeo(f.country);
    if (!key) return;
    if (!byCountry[key]) {
      byCountry[key] = [];
    }
    byCountry[key].push(f);
  });

  vizCtx.fightersByCountry = byCountry;
}

/**
 * Build a dictionary from country name -> list of title fights.
 * This is computed once and reused for every hover.
 */
function buildFightsByCountry(fights) {
  const byCountry = {};

  fights.forEach((f) => {
    // Only keep title fights
    const isTitle =
      typeof f.title_bout === "string" &&
      f.title_bout.trim().toLowerCase() === "true";
    if (!isTitle) return;

    const key = normalizeCountryNameForGeo(f.country);
    if (!key) return;
    if (!byCountry[key]) {
      byCountry[key] = [];
    }
    byCountry[key].push(f);
  });

  vizCtx.fightsByCountry = byCountry;
}

/**
 * Normalize country names from the fighters CSV so they match the
 * GeoJSON `properties.name` values.
 *
 * For example, the CSV uses "United States" but the map uses
 * "United States of America", so we map that explicitly.
 */
function normalizeCountryNameForGeo(rawCountry) {
  if (!rawCountry) return "";
  const c = rawCountry.trim().toLowerCase();

  // Explicit aliases where CSV and GeoJSON differ
  const aliases = {
    "united states": "United States of America",
    "usa": "United States of America",
    "u.s.a.": "United States of America",
  };

  if (aliases[c]) {
    return aliases[c];
  }

  // Default: return original trimmed country string
  return rawCountry.trim();
}

/**
 * Render the world as a 2D globe (flat map projection).
 * Hovering a country updates the stats panel on the right.
 */
function renderGlobe(worldGeo) {
  const svg = vizCtx.globeSvg;
  if (!svg) return;

  const width = vizCtx.globeWidth;
  const height = vizCtx.globeHeight;

  // Use a 2D projection (Natural Earth) instead of 3D orthographic
  const baseScale = Math.min(width, height) * 0.22;
  vizCtx.globeBaseScale = baseScale;

  const projection = d3
    .geoNaturalEarth1()
    .scale(baseScale * (vizCtx.globeZoom || 1))
    .translate([
      width / 2 + (vizCtx.globePanX || 0),
      height / 2 + (vizCtx.globePanY || 0),
    ]);

  const path = d3.geoPath().projection(projection);
  const graticule = d3.geoGraticule();

  // Save for zoom updates
  vizCtx.globeProjection = projection;
  vizCtx.globePath = path;

  svg.selectAll("*").remove();

  const defs = svg.append("defs");

  // Simple radial gradient for the globe water
  const gradient = defs
    .append("radialGradient")
    .attr("id", "globe-water-gradient")
    .attr("cx", "50%")
    .attr("cy", "50%")
    .attr("r", "50%");

  gradient
    .append("stop")
    .attr("offset", "0%")
    .attr("stop-color", "#1f3b56");
  gradient
    .append("stop")
    .attr("offset", "100%")
    .attr("stop-color", "#0b1724");

  const globeGroup = svg
    .append("g")
    .attr("class", "globe-root")
    .attr("transform", `translate(0,0)`);

  // Water background
  globeGroup
    .append("path")
    .datum({ type: "Sphere" })
    .attr("class", "globe-water")
    .attr("d", path)
    .attr("fill", "url(#globe-water-gradient)")
    .attr("stroke", "#02101f")
    .attr("stroke-width", 1);

  // Graticule
  globeGroup
    .append("path")
    .datum(graticule())
    .attr("class", "globe-graticule")
    .attr("d", path)
    .attr("fill", "none")
    .attr("stroke", "rgba(255,255,255,0.15)")
    .attr("stroke-width", 0.5);

  const countriesGroup = globeGroup.append("g").attr("class", "countries");

  const countries =
    worldGeo.features || worldGeo.features === 0 ? worldGeo.features : [];

  countriesGroup
    .selectAll("path.country")
    .data(countries)
    .join("path")
    .attr("class", "country")
    .attr("d", path)
    .attr("fill", (d) => {
      const name = d.properties && d.properties.name;
      const hasFighters = !!vizCtx.fightersByCountry[name];
      return hasFighters ? "#2d8b57" : "#3b4a5a";
    })
    .attr("stroke", "#111927")
    .attr("stroke-width", 0.6)
    .on("mouseover", function (event, d) {
      d3.select(this).attr("stroke-width", 1.5).attr("stroke", "#ffffff");

      const name = d.properties && d.properties.name;
      renderStatsPanel(name);
      renderFightsPanel(name);
    })
    .on("mouseout", function () {
      d3.select(this).attr("stroke-width", 0.6).attr("stroke", "#111927");
    });

  // Drag to pan the map (adjust projection translate)
  let dragStart = null;
  let startPanX = 0;
  let startPanY = 0;

  const drag = d3
    .drag()
    .on("start", function (event) {
      dragStart = [event.x, event.y];
      startPanX = vizCtx.globePanX || 0;
      startPanY = vizCtx.globePanY || 0;
    })
    .on("drag", function (event) {
      if (!dragStart) return;
      const dx = event.x - dragStart[0];
      const dy = event.y - dragStart[1];
      vizCtx.globePanX = startPanX + dx;
      vizCtx.globePanY = startPanY + dy;

      vizCtx.globeProjection.translate([
        width / 2 + vizCtx.globePanX,
        height / 2 + vizCtx.globePanY,
      ]);

      svg.selectAll(".globe-root path").attr("d", vizCtx.globePath);
    })
    .on("end", function () {
      dragStart = null;
    });

  svg.call(drag);
}

/**
 * Update the globe projection's scale based on the current zoom level
 * and re-render all globe paths.
 */
function updateGlobeZoom() {
  const svg = vizCtx.globeSvg;
  if (!svg || !vizCtx.globeProjection || !vizCtx.globePath || !vizCtx.globeBaseScale) {
    return;
  }

  const zoom = vizCtx.globeZoom || 1;
  vizCtx.globeProjection
    .scale(vizCtx.globeBaseScale * zoom)
    .translate([
      vizCtx.globeWidth / 2 + (vizCtx.globePanX || 0),
      vizCtx.globeHeight / 2 + (vizCtx.globePanY || 0),
    ]);

  svg.selectAll(".globe-root path").attr("d", vizCtx.globePath);
}

/**
 * Render or update the right-hand stats panel for a given country.
 * countryName === null will show an empty / instruction state.
 */
function renderStatsPanel(countryName) {
  const svg = vizCtx.statsSvg;
  if (!svg) return;

  const { statsWidth, statsHeight, statsMargin } = vizCtx;

  svg.selectAll("*").remove();

  const root = svg
    .append("g")
    .attr("transform", `translate(${statsMargin.left},${statsMargin.top})`);

  const innerWidth = statsWidth - statsMargin.left - statsMargin.right;
  const innerHeight = statsHeight - statsMargin.top - statsMargin.bottom;

  // Panel background
  root
    .append("rect")
    .attr("x", -statsMargin.left + 4)
    .attr("y", -statsMargin.top + 4)
    .attr("width", statsWidth - 8)
    .attr("height", statsHeight - 8)
    .attr("rx", 12)
    .attr("fill", "#050814")
    .attr("stroke", "#1b2636")
    .attr("stroke-width", 1);

  // Title
  const titleText =
    countryName && vizCtx.fightersByCountry[countryName]
      ? `Fighters from ${countryName}`
      : "Hover a country on the globe to see its fighters";

  root
    .append("text")
    .attr("x", 0)
    .attr("y", 0)
    .attr("dy", "0.2em")
    .attr("font-size", 22)
    .attr("font-weight", 600)
    .attr("fill", "#f5f7fa")
    .text(titleText);

  if (!countryName || !vizCtx.fightersByCountry[countryName]) {
    // No country selected or no fighters
    root
      .append("text")
      .attr("x", 0)
      .attr("y", 40)
      .attr("font-size", 14)
      .attr("fill", "#9ca3af")
      .text("Move your cursor over a highlighted country on the globe.");
    return;
  }

  const allFighters = vizCtx.fightersByCountry[countryName].slice(); // copy

  // Sort fighters by total wins desc, then by name
  allFighters.sort((a, b) => {
    if (b.wins !== a.wins) return b.wins - a.wins;
    return (a.fighter_name || "").localeCompare(b.fighter_name || "");
  });

  const maxPerCountry = vizCtx.maxFightersPerCountry || allFighters.length;
  const fighters = allFighters.slice(0, maxPerCountry);

  const headerY = 34;
  const contentStartY = headerY + 28;
  const rowHeight = 46;

  // Header text
  root
    .append("text")
    .attr("x", 0)
    .attr("y", headerY)
    .attr("font-size", 15)
    .attr("font-weight", 600)
    .attr("fill", "#d1d5db")
    .text("Top fighters by wins, with KO / Sub / Dec / Other breakdown");

  // Color palette for outcome types
  const winColors = {
    ko: "#f97373",
    sub: "#a855f7",
    dec: "#3b82f6",
    oth: "#9ca3af",
  };
  const lossColors = {
    ko: "#dc2626", // Darker red for losses
    sub: "#9333ea", // Darker purple for losses
    dec: "#2563eb", // Darker blue for losses
    oth: "#6b7280", // Darker gray for losses
  };

  // Legend
  const legend = root
    .append("g")
    .attr("transform", `translate(0, ${headerY + 8})`);

  const legendItems = [
    { label: "KO", winColor: winColors.ko, lossColor: lossColors.ko },
    { label: "Sub", winColor: winColors.sub, lossColor: lossColors.sub },
    { label: "Dec", winColor: winColors.dec, lossColor: lossColors.dec },
    { label: "Other", winColor: winColors.oth, lossColor: lossColors.oth },
  ];

  const legendItemWidth = 90;

  const legendGroup = legend
    .selectAll("g.legend-item")
    .data(legendItems)
    .join("g")
    .attr("class", "legend-item")
    .attr("transform", (_, i) => `translate(${i * legendItemWidth}, 0)`);

  legendGroup
    .append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", 14)
    .attr("height", 6)
    .attr("rx", 2)
    .attr("fill", (d) => d.winColor);

  legendGroup
    .append("rect")
    .attr("x", 0)
    .attr("y", 8)
    .attr("width", 14)
    .attr("height", 6)
    .attr("rx", 2)
    .attr("fill", (d) => d.lossColor);

  legendGroup
    .append("text")
    .attr("x", 20)
    .attr("y", 8)
    .attr("dy", "0.1em")
    .attr("font-size", 12)
    .attr("fill", "#d1d5db")
    .text((d) => d.label);

  const maxRows = Math.floor(innerHeight / rowHeight) - 1;
  const rowsToShow = fighters.slice(0, Math.max(0, maxRows));

  // Layout for label vs bar area so that we don't overflow the SVG width
  const labelWidth = 210; // space reserved for text on the left
  const rightPadding = 80; // keep some breathing room on the right edge
  const barAreaWidth = Math.max(80, innerWidth - labelWidth - rightPadding);

  // Scales for wins and losses bars
  const maxWins = d3.max(rowsToShow, (d) => d.wins) || 1;
  const maxLosses = d3.max(rowsToShow, (d) => d.lossess) || 1;

  const winScale = d3.scaleLinear().domain([0, maxWins]).range([0, barAreaWidth]);
  const lossScale = d3.scaleLinear().domain([0, maxLosses]).range([0, barAreaWidth]);

  const rowsGroup = root
    .append("g")
    .attr("transform", `translate(0, ${contentStartY})`);

  const row = rowsGroup
    .selectAll("g.fighter-row")
    .data(rowsToShow)
    .join("g")
    .attr("class", "fighter-row")
    .attr("transform", (_, i) => `translate(0, ${i * rowHeight})`);

  // Left: text labels
  row
    .append("text")
    .attr("x", 0)
    .attr("y", 0)
    .attr("dy", "0.9em")
    .attr("font-size", 15)
    .attr("font-weight", 600)
    .attr("fill", "#e5e7eb")
    .text((f) => f.fighter_name || "Unknown");

  // Nickname on its own line
  row
    .append("text")
    .attr("x", 0)
    .attr("y", 0)
    .attr("dy", "2.1em")
    .attr("font-size", 12)
    .attr("fill", "#9ca3af")
    .text((f) => {
      const nickname =
        f.nickname && f.nickname.trim() && f.nickname.trim() !== "N/A"
          ? `"${f.nickname.trim()}"`
          : "";
      return nickname;
    });

  // Age, height, weight on a new line below nickname
  row
    .append("text")
    .attr("x", 0)
    .attr("y", 0)
    .attr("dy", "3.3em")
    .attr("font-size", 12)
    .attr("fill", "#9ca3af")
    .text((f) => {
      const age = f.age != null ? `${f.age}y` : "?y";
      const height = f.height || "?";
      const weight = f.weight || "?";
      return [`age ${age}`, height, weight].filter(Boolean).join(" · ");
    });

  const barOffsetX = labelWidth;
  const barHeight = 6;
  const barGap = 4;

  // Wins bars (stacked)
  const winsGroup = row
    .append("g")
    .attr("transform", `translate(${barOffsetX}, 4)`);

  winsGroup.each(function (f) {
    const g = d3.select(this);
    let x = 0;
    const segments = [
      { key: "wins_ko", color: winColors.ko },
      { key: "wins_submission", color: winColors.sub },
      { key: "wins_decision", color: winColors.dec },
      { key: "wins_other", color: winColors.oth },
    ];
    segments.forEach((seg) => {
      const value = f[seg.key] || 0;
      if (!value) return;
      const w = winScale(value);
      if (w <= 0) return;
      g.append("rect")
        .attr("x", x)
        .attr("y", 0)
        .attr("width", w)
        .attr("height", barHeight)
        .attr("rx", 2)
        .attr("fill", seg.color);
      x += w;
    });
  });

  // Losses bars (stacked)
  const lossesGroup = row
    .append("g")
    .attr("transform", `translate(${barOffsetX}, ${barHeight + barGap + 4})`);

  lossesGroup.each(function (f) {
    const g = d3.select(this);
    let x = 0;
    const segments = [
      { key: "losses_ko", color: lossColors.ko },
      { key: "losses_submission", color: lossColors.sub },
      { key: "losses_decision", color: lossColors.dec },
      { key: "losses_other", color: lossColors.oth },
    ];
    segments.forEach((seg) => {
      const value = f[seg.key] || 0;
      if (!value) return;
      const w = lossScale(value);
      if (w <= 0) return;
      g.append("rect")
        .attr("x", x)
        .attr("y", 0)
        .attr("width", w)
        .attr("height", barHeight)
        .attr("rx", 2)
        .attr("fill", seg.color);
      x += w;
    });
  });

  // Small text labels for totals at the right end of bars
  row
    .append("text")
    .attr("x", barOffsetX + barAreaWidth + 6)
    .attr("y", 0)
    .attr("dy", "1.1em")
    .attr("font-size", 12)
    .attr("fill", "#e5e7eb")
    .text((f) => `W ${f.wins || 0}`);

  row
    .append("text")
    .attr("x", barOffsetX + barAreaWidth + 6)
    .attr("y", 0)
    .attr("dy", "2.3em")
    .attr("font-size", 12)
    .attr("fill", "#e5e7eb")
    .text((f) => `L ${f.lossess || 0}`);

  if (fighters.length > rowsToShow.length) {
    root
      .append("text")
      .attr("x", 0)
      .attr("y", statsHeight - statsMargin.bottom)
      .attr("font-size", 12)
      .attr("fill", "#9ca3af")
      .text(
        `Showing ${rowsToShow.length} of ${fighters.length} fighters for ${countryName}`
      );
  }
}

/**
 * Convert betting odds to implied probabilities and renormalize to sum to 100%.
 * Returns { rProb, bProb } or null if odds are invalid.
 */
function convertOddsToProbabilities(rOdds, bOdds) {
  if (!Number.isFinite(rOdds) || !Number.isFinite(bOdds)) {
    return null;
  }

  let rProb, bProb;

  // Convert negative odds (favorite): |odds| / (|odds| + 100) * 100
  if (rOdds < 0) {
    rProb = Math.abs(rOdds) / (Math.abs(rOdds) + 100) * 100;
  } else {
    // Convert positive odds (underdog): 100 / (odds + 100) * 100
    rProb = 100 / (rOdds + 100) * 100;
  }

  if (bOdds < 0) {
    bProb = Math.abs(bOdds) / (Math.abs(bOdds) + 100) * 100;
  } else {
    bProb = 100 / (bOdds + 100) * 100;
  }

  // Renormalize so both sum to 100%
  const total = rProb + bProb;
  if (total <= 0) return null;

  return {
    rProb: (rProb / total) * 100,
    bProb: (bProb / total) * 100,
  };
}

/**
 * Render or update the fights panel (title fights per country).
 * countryName === null will show an empty / instruction state.
 * This is intentionally kept separate from the fighters stats panel logic.
 */
function renderFightsPanel(countryName) {
  const svg = vizCtx.fightsSvg;
  if (!svg) return;

  const width = vizCtx.fightsWidth;
  const height = vizCtx.fightsHeight;
  const margin = { top: 40, right: 20, bottom: 40, left: 20 };

  svg.selectAll("*").remove();

  const root = svg
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Panel background
  root
    .append("rect")
    .attr("x", -margin.left + 4)
    .attr("y", -margin.top + 4)
    .attr("width", width - 8)
    .attr("height", height - 8)
    .attr("rx", 12)
    .attr("fill", "#050814")
    .attr("stroke", "#1b2636")
    .attr("stroke-width", 1);

  const hasCountry = countryName && vizCtx.fightsByCountry[countryName];

  const titleText = hasCountry
    ? `Title fights in ${countryName}`
    : "Hover a country to see its title fights";

  root
    .append("text")
    .attr("x", 0)
    .attr("y", 0)
    .attr("dy", "0.2em")
    .attr("font-size", 22)
    .attr("font-weight", 600)
    .attr("fill", "#f5f7fa")
    .text(titleText);

  if (!hasCountry) {
    root
      .append("text")
      .attr("x", 0)
      .attr("y", 40)
      .attr("font-size", 14)
      .attr("fill", "#9ca3af")
      .text("Only title bouts are listed here.");
    return;
  }

  const allFights = vizCtx.fightsByCountry[countryName];
  const maxPerCountry = vizCtx.maxFightsPerCountry || allFights.length;
  const fights = allFights.slice(0, maxPerCountry);


  const headerY = 34;
  const cardStartY = headerY + 28;
  const cardHeight = 85;
  const cardGap = 8;
  const cardPadding = 10;

  // Header
  root
    .append("text")
    .attr("x", 0)
    .attr("y", headerY)
    .attr("font-size", 13)
    .attr("font-weight", 600)
    .attr("fill", "#d1d5db")
    .text("Title bouts with fighter odds and details");

  const maxCards = Math.floor((innerHeight - (cardStartY - headerY)) / (cardHeight + cardGap));
  const cardsToShow = fights.slice(0, Math.max(0, maxCards));

  const cardsGroup = root
    .append("g")
    .attr("transform", `translate(0, ${cardStartY})`);

  // Color definitions
  const redColor = "#f97373";
  const blueColor = "#3b82f6";
  const winnerHighlight = "#fbbf24";

  const card = cardsGroup
    .selectAll("g.fight-card")
    .data(cardsToShow)
    .join("g")
    .attr("class", "fight-card")
    .attr("transform", (_, i) => `translate(0, ${i * (cardHeight + cardGap)})`);

  // Card background
  card
    .append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", innerWidth)
    .attr("height", cardHeight)
    .attr("rx", 8)
    .attr("fill", "#0f1724")
    .attr("stroke", "#1b2636")
    .attr("stroke-width", 1);

  // Fighter section (top part of card)
  const fighterSectionY = cardPadding + 4;
  const fighterNameY = fighterSectionY + 16;
  const vsX = innerWidth / 2;

  // R_fighter (left, red)
  const rFighterGroup = card
    .append("g")
    .attr("class", "r-fighter");

  const rFighterName = (f) => f.R_fighter || "Red Fighter";
  const rFighterWinner = (f) => {
    // Winner field is "Red" or "Blue", not the fighter name
    const isWinner = f.Winner && f.Winner.trim().toLowerCase() === "red";
    return isWinner;
  };

  rFighterGroup
    .append("text")
    .attr("x", cardPadding)
    .attr("y", fighterNameY)
    .attr("font-size", 14)
    .attr("font-weight", 600)
    .attr("fill", redColor)
    .text(rFighterName);

  // Add gold medal emoji for winner
  rFighterGroup
    .filter((f) => rFighterWinner(f))
    .append("text")
    .attr("x", (f) => {
      const textWidth = rFighterName(f).length * 8.5; // Approximate text width
      return cardPadding + textWidth + 6;
    })
    .attr("y", fighterNameY)
    .attr("font-size", 16)
    .text("🥇");

  // B_fighter (right, blue)
  const bFighterGroup = card
    .append("g")
    .attr("class", "b-fighter");

  const bFighterName = (f) => f.B_fighter || "Blue Fighter";
  const bFighterWinner = (f) => {
    // Winner field is "Red" or "Blue", not the fighter name
    const isWinner = f.Winner && f.Winner.trim().toLowerCase() === "blue";
    return isWinner;
  };

  bFighterGroup
    .append("text")
    .attr("x", innerWidth - cardPadding)
    .attr("y", fighterNameY)
    .attr("text-anchor", "end")
    .attr("font-size", 14)
    .attr("font-weight", 600)
    .attr("fill", blueColor)
    .text(bFighterName);

  // Add gold medal emoji for winner (before the name since it's right-aligned)
  bFighterGroup
    .filter((f) => bFighterWinner(f))
    .append("text")
    .attr("x", (f) => {
      const textWidth = bFighterName(f).length * 8.5; // Approximate text width
      return innerWidth - cardPadding - textWidth - 6;
    })
    .attr("y", fighterNameY)
    .attr("font-size", 16)
    .text("🥇");

  // VS separator
  card
    .append("text")
    .attr("x", vsX)
    .attr("y", fighterNameY)
    .attr("text-anchor", "middle")
    .attr("font-size", 12)
    .attr("font-weight", 600)
    .attr("fill", "#9ca3af")
    .text("VS");

  // Odds bars section
  const oddsBarY = fighterNameY + 22;
  const oddsBarHeight = 6;
  const oddsBarGap = 2;
  const oddsBarAreaWidth = innerWidth - cardPadding * 2;
  const oddsBarStartX = cardPadding;

  card.each(function (f) {
    const cardSelection = d3.select(this);
    const rOdds = f.R_odds;
    const bOdds = f.B_odds;
    const probs = convertOddsToProbabilities(rOdds, bOdds);

    if (probs) {
      const rBarWidth = (probs.rProb / 100) * oddsBarAreaWidth;
      const bBarWidth = (probs.bProb / 100) * oddsBarAreaWidth;

      // R odds bar (red)
      cardSelection
        .append("rect")
        .attr("x", oddsBarStartX)
        .attr("y", oddsBarY)
        .attr("width", rBarWidth)
        .attr("height", oddsBarHeight)
        .attr("rx", 2)
        .attr("fill", redColor);

      // B odds bar (blue)
      cardSelection
        .append("rect")
        .attr("x", oddsBarStartX + rBarWidth + oddsBarGap)
        .attr("y", oddsBarY)
        .attr("width", bBarWidth)
        .attr("height", oddsBarHeight)
        .attr("rx", 2)
        .attr("fill", blueColor);

      // Odds text above bars
      cardSelection
        .append("text")
        .attr("x", oddsBarStartX)
        .attr("y", oddsBarY - 2)
        .attr("font-size", 10)
        .attr("fill", "#d1d5db")
        .text(Number.isFinite(rOdds) ? rOdds.toString() : "N/A");

      cardSelection
        .append("text")
        .attr("x", oddsBarStartX + rBarWidth + oddsBarGap)
        .attr("y", oddsBarY - 2)
        .attr("font-size", 10)
        .attr("fill", "#d1d5db")
        .text(Number.isFinite(bOdds) ? bOdds.toString() : "N/A");
    } else {
      // No valid odds - show N/A
      cardSelection
        .append("text")
        .attr("x", oddsBarStartX)
        .attr("y", oddsBarY + oddsBarHeight / 2)
        .attr("dy", "0.35em")
        .attr("font-size", 10)
        .attr("fill", "#9ca3af")
        .text("Odds: N/A");
    }
  });

  // Metadata section (bottom)
  const metadataY = oddsBarY + oddsBarHeight + 12;

  card.each(function (f) {
    const cardSelection = d3.select(this);
    const date = f.date || "?";
    const location = f.location || "";
    const weight = f.weight_class || "";
    const gender = f.gender || "";

    // Date (left)
    cardSelection
      .append("text")
      .attr("x", cardPadding)
      .attr("y", metadataY)
      .attr("font-size", 11)
      .attr("fill", "#9ca3af")
      .text(date);

    // Location (center)
    if (location) {
      cardSelection
        .append("text")
        .attr("x", vsX)
        .attr("y", metadataY)
        .attr("text-anchor", "middle")
        .attr("font-size", 11)
        .attr("fill", "#9ca3af")
        .text(location);
    }

    // Weight class + Gender (right)
    const metaParts = [weight, gender].filter(Boolean).join(" · ");
    if (metaParts) {
      cardSelection
        .append("text")
        .attr("x", innerWidth - cardPadding)
        .attr("y", metadataY)
        .attr("text-anchor", "end")
        .attr("font-size", 11)
        .attr("fill", "#9ca3af")
        .text(metaParts);
    }
  });

  if (fights.length > cardsToShow.length) {
    root
      .append("text")
      .attr("x", 0)
      .attr("y", height - margin.bottom)
      .attr("font-size", 11)
      .attr("fill", "#9ca3af")
      .text(
        `Showing ${cardsToShow.length} of ${fights.length} title fights for ${countryName}`
      );
  }
}

/**
 * Turn a fight object into a compact text row with the main fields.
 */
function formatFightRow(f) {
  const date = f.date || "?";
  const r = f.R_fighter || "Red";
  const b = f.B_fighter || "Blue";
  const rOdds = Number.isFinite(f.R_odds) ? f.R_odds : "?";
  const bOdds = Number.isFinite(f.B_odds) ? f.B_odds : "?";
  const winner = f.Winner || "?";
  const location = f.location || "";
  const weight = f.weight_class || "";
  const gender = f.gender || "";

  const fightersPart = `${r} (${rOdds}) vs ${b} (${bOdds})`;
  const winnerPart = `winner: ${winner}`;
  const metaParts = [location, weight, gender].filter(Boolean).join(" · ");

  return `${date} — ${fightersPart} — ${winnerPart}${
    metaParts ? " — " + metaParts : ""
  }`;
}

/**
 * Turn a fighter object into a compact text row with the requested fields.
 */
function formatFighterRow(f) {
  const nickname =
    f.nickname && f.nickname.trim() && f.nickname.trim() !== "N/A"
      ? f.nickname.trim()
      : "no nickname";

  const age = f.age != null ? f.age : "?";
  const height = f.height || "?";
  const weight = f.weight || "?";

  const wins = f.wins || 0;
  const losses = f.lossess || 0;

  const winsBreakdown = `W ${wins} (KO ${f.wins_ko || 0}, Sub ${
    f.wins_submission || 0
  }, Dec ${f.wins_decision || 0}, Oth ${f.wins_other || 0})`;

  const lossesBreakdown = `L ${losses} (KO ${f.losses_ko || 0}, Sub ${
    f.losses_submission || 0
  }, Dec ${f.losses_decision || 0}, Oth ${f.losses_other || 0})`;

  return `${f.fighter_name || "Unknown"} (${nickname}), age ${age}, ${height}, ${
    weight
  } — ${winsBreakdown}; ${lossesBreakdown}`;
}