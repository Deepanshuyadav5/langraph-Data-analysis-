import { useEffect, useMemo, useRef, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import html2canvas from "html2canvas";
import {
  Activity,
  BarChart3,
  Brain,
  Database,
  Download,
  FileDown,
  Layers,
  LineChart,
  Palette,
  PieChart,
  Play,
  Plus,
  Sparkles,
  Table2,
  Trash2,
  Upload,
} from "lucide-react";

const Plot = createPlotlyComponent(Plotly);
const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
const COLORS = ["#176b63", "#d47635", "#4e6f50", "#9a4d3c", "#5271a3", "#c0a03d"];

const chartTypes = [
  { value: "bar", label: "Bar", icon: BarChart3 },
  { value: "line", label: "Line", icon: LineChart },
  { value: "scatter", label: "Scatter", icon: Activity },
  { value: "area", label: "Area", icon: LineChart },
  { value: "histogram", label: "Histogram", icon: BarChart3 },
  { value: "pie", label: "Pie", icon: PieChart },
  { value: "box", label: "Box", icon: Activity },
  { value: "heatmap", label: "Heatmap", icon: Layers },
];

const tabs = [
  { id: "overview", label: "Overview", icon: Database },
  { id: "insights", label: "Insights", icon: Brain },
  { id: "visuals", label: "Visuals", icon: BarChart3 },
  { id: "prediction", label: "Prediction", icon: Sparkles },
  { id: "quality", label: "Quality", icon: Activity },
  { id: "data", label: "Data", icon: Table2 },
];

const themes = [
  { id: "aurora", label: "Aurora" },
  { id: "ember", label: "Ember" },
  { id: "cosmic", label: "Cosmic" },
];

const plotConfig = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
  toImageButtonOptions: {
    format: "png",
    filename: "graph",
    height: 900,
    width: 1400,
    scale: 2,
  },
};

const baseLayout = {
  autosize: true,
  paper_bgcolor: "transparent",
  plot_bgcolor: "#ffffff",
  font: {
    family: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
    color: "#26312f",
    size: 12,
  },
  margin: { t: 46, r: 28, b: 58, l: 58 },
  colorway: COLORS,
  hovermode: "closest",
  legend: { orientation: "h", y: -0.22 },
};

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const number = Number(value);
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: Math.abs(number) >= 1000 ? 0 : digits,
  }).format(number);
}

function makeId(prefix = "chart") {
  if (window.crypto?.randomUUID) return `${prefix}-${window.crypto.randomUUID()}`;
  return `${prefix}-${Date.now()}-${Math.round(Math.random() * 100000)}`;
}

function titleFor(config) {
  if (config.type === "heatmap") return "Correlation heatmap";
  if (config.type === "histogram") return `${config.y || "Value"} distribution`;
  return `${config.y || "Value"} by ${config.x || "Category"}`;
}

function downloadDataUrl(dataUrl, filename) {
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = filename;
  link.click();
}

function downloadBlob(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  downloadDataUrl(url, filename);
  URL.revokeObjectURL(url);
}

async function apiFetch(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    const detail = typeof payload === "object" ? payload.detail : payload;
    throw new Error(detail || response.statusText);
  }

  return payload;
}

function numericValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function aggregateRecords(records, xColumn, yColumn, aggregation = "sum") {
  const buckets = new Map();

  records.forEach((row) => {
    const key = row[xColumn] ?? "Blank";
    const value = numericValue(row[yColumn]);
    if (value === null && aggregation !== "count") return;

    const bucket = buckets.get(key) || {
      sum: 0,
      count: 0,
      min: Number.POSITIVE_INFINITY,
      max: Number.NEGATIVE_INFINITY,
    };

    bucket.count += 1;
    if (value !== null) {
      bucket.sum += value;
      bucket.min = Math.min(bucket.min, value);
      bucket.max = Math.max(bucket.max, value);
    }
    buckets.set(key, bucket);
  });

  return Array.from(buckets.entries())
    .map(([label, bucket]) => {
      let value = bucket.sum;
      if (aggregation === "average") value = bucket.count ? bucket.sum / bucket.count : 0;
      if (aggregation === "count") value = bucket.count;
      if (aggregation === "min") value = bucket.min === Number.POSITIVE_INFINITY ? 0 : bucket.min;
      if (aggregation === "max") value = bucket.max === Number.NEGATIVE_INFINITY ? 0 : bucket.max;
      return { label: String(label), value };
    })
    .sort((a, b) => b.value - a.value)
    .slice(0, 40);
}

function buildSingleFigure(config, dataset) {
  const records = dataset?.records || [];
  const title = config.title || titleFor(config);
  const layout = {
    ...baseLayout,
    title: { text: title, x: 0.02, xanchor: "left", font: { size: 15 } },
    height: config.height || 380,
  };

  if (!dataset) return { data: [], layout };

  if (config.type === "heatmap") {
    if (!dataset.correlation) {
      return {
        data: [],
        layout: {
          ...layout,
          annotations: [
            {
              text: "Need at least two numeric columns",
              x: 0.5,
              y: 0.5,
              xref: "paper",
              yref: "paper",
              showarrow: false,
            },
          ],
        },
      };
    }

    return {
      data: [
        {
          type: "heatmap",
          z: dataset.correlation.matrix,
          x: dataset.correlation.labels,
          y: dataset.correlation.labels,
          colorscale: [
            [0, "#9a4d3c"],
            [0.5, "#f3efe8"],
            [1, "#176b63"],
          ],
          zmin: -1,
          zmax: 1,
          hovertemplate: "%{y} vs %{x}<br>r=%{z:.3f}<extra></extra>",
        },
      ],
      layout,
    };
  }

  const xColumn = config.x || dataset.columns[0];
  const yColumn = config.y || dataset.numericColumns[0] || dataset.columns[0];

  if (config.type === "histogram") {
    return {
      data: [
        {
          type: "histogram",
          x: records.map((row) => row[yColumn]).filter((value) => numericValue(value) !== null),
          marker: { color: COLORS[0] },
          name: yColumn,
        },
      ],
      layout: { ...layout, bargap: 0.08, xaxis: { title: yColumn }, yaxis: { title: "Count" } },
    };
  }

  if (config.type === "scatter") {
    return {
      data: [
        {
          type: "scatter",
          mode: "markers",
          x: records.map((row) => row[xColumn]),
          y: records.map((row) => row[yColumn]),
          marker: { color: COLORS[1], size: 8, opacity: 0.78 },
          name: yColumn,
        },
      ],
      layout: { ...layout, xaxis: { title: xColumn }, yaxis: { title: yColumn } },
    };
  }

  if (config.type === "line" || config.type === "area") {
    const aggregated = aggregateRecords(records, xColumn, yColumn, config.aggregation);
    return {
      data: [
        {
          type: "scatter",
          mode: "lines+markers",
          fill: config.type === "area" ? "tozeroy" : "none",
          x: aggregated.map((row) => row.label),
          y: aggregated.map((row) => row.value),
          line: { color: COLORS[2], width: 3 },
          marker: { color: COLORS[2], size: 7 },
          name: yColumn,
        },
      ],
      layout: { ...layout, xaxis: { title: xColumn }, yaxis: { title: yColumn } },
    };
  }

  if (config.type === "pie") {
    const aggregated = aggregateRecords(records, xColumn, yColumn, config.aggregation).slice(0, 12);
    return {
      data: [
        {
          type: "pie",
          labels: aggregated.map((row) => row.label),
          values: aggregated.map((row) => row.value),
          hole: 0.36,
          marker: { colors: COLORS },
          textinfo: "label+percent",
          name: yColumn,
        },
      ],
      layout: { ...layout, margin: { t: 46, r: 24, b: 24, l: 24 }, showlegend: false },
    };
  }

  if (config.type === "box") {
    return {
      data: [
        {
          type: "box",
          x: records.map((row) => row[xColumn]),
          y: records.map((row) => row[yColumn]),
          marker: { color: COLORS[3] },
          boxmean: true,
          name: yColumn,
        },
      ],
      layout: { ...layout, xaxis: { title: xColumn }, yaxis: { title: yColumn } },
    };
  }

  const aggregated = aggregateRecords(records, xColumn, yColumn, config.aggregation);
  return {
    data: [
      {
        type: "bar",
        x: aggregated.map((row) => row.label),
        y: aggregated.map((row) => row.value),
        marker: { color: COLORS[0] },
        name: yColumn,
      },
    ],
    layout: { ...layout, xaxis: { title: xColumn }, yaxis: { title: yColumn }, bargap: 0.18 },
  };
}

function buildComboFigure(config, dataset) {
  const components = config.components || [];
  const cols = Math.min(2, Math.max(1, components.length));
  const rows = Math.ceil(components.length / cols);
  const horizontalGap = 0.08;
  const verticalGap = 0.14;
  const cellWidth = (1 - horizontalGap * (cols - 1)) / cols;
  const cellHeight = (1 - verticalGap * (rows - 1)) / rows;
  const data = [];
  const layout = {
    ...baseLayout,
    title: { text: config.title || "Merged visual", x: 0.02, xanchor: "left", font: { size: 15 } },
    height: Math.max(440, rows * 320),
    margin: { t: 62, r: 30, b: 46, l: 56 },
    annotations: [],
    showlegend: false,
  };

  components.forEach((child, index) => {
    const col = index % cols;
    const row = Math.floor(index / cols);
    const x0 = col * (cellWidth + horizontalGap);
    const x1 = x0 + cellWidth;
    const y1 = 1 - row * (cellHeight + verticalGap);
    const y0 = y1 - cellHeight;
    const suffix = index === 0 ? "" : String(index + 1);
    const xAxisKey = `xaxis${suffix}`;
    const yAxisKey = `yaxis${suffix}`;
    const xAxisRef = `x${suffix}`;
    const yAxisRef = `y${suffix}`;
    const figure = buildSingleFigure(child, dataset);

    layout.annotations.push({
      text: child.title || titleFor(child),
      x: (x0 + x1) / 2,
      y: Math.min(1.04, y1 + 0.045),
      xref: "paper",
      yref: "paper",
      showarrow: false,
      font: { size: 12, color: "#26312f" },
    });

    figure.data.forEach((trace) => {
      if (trace.type === "pie") {
        data.push({ ...trace, domain: { x: [x0, x1], y: [y0, y1] }, textinfo: "percent" });
      } else {
        data.push({ ...trace, xaxis: xAxisRef, yaxis: yAxisRef, name: child.title || titleFor(child) });
        layout[xAxisKey] = { domain: [x0, x1], anchor: yAxisRef, title: "" };
        layout[yAxisKey] = { domain: [y0, y1], anchor: xAxisRef, title: "" };
      }
    });
  });

  return { data, layout };
}

function buildFigure(config, dataset) {
  if (config.type === "combo") return buildComboFigure(config, dataset);
  return buildSingleFigure(config, dataset);
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function DataTable({ columns, rows, empty = "No rows" }) {
  return (
    <div className="table-frame">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length ? (
            rows.map((row, index) => (
              <tr key={`${index}-${columns[0]?.key}`}>
                {columns.map((column) => (
                  <td key={column.key}>{column.render ? column.render(row) : row[column.key]}</td>
                ))}
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan={columns.length}>{empty}</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function PlotPanel({ divId, figure, className = "", height }) {
  return (
    <Plot
      divId={divId}
      data={figure.data}
      layout={{ ...figure.layout, height: height || figure.layout.height }}
      config={plotConfig}
      className={`plot ${className}`}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
    />
  );
}

function LandingHero({ onUpload, onSample, loading }) {
  const features = [
    { label: "AI insights", value: "Groq + LangGraph" },
    { label: "Charts", value: "Plotly dashboard" },
    { label: "Checks", value: "Quality + prediction" },
  ];

  return (
    <section className="landing-hero" aria-label="Langalytics landing">
      <div className="hero-copy">
        <p className="eyebrow">LangGraph powered analytics</p>
        <h2>Langalytics turns raw CSV files into automated data analysis.</h2>
        <p>
          Upload a dataset and get summaries, AI insights, visualizations, prediction workflows,
          and data quality checks in one interactive workspace.
        </p>
        <div className="hero-actions">
          <label className="button primary hero-upload">
            <Upload size={18} />
            Upload dataset
            <input type="file" accept=".csv,.xlsx,.xls" onChange={onUpload} />
          </label>
          <button className="button ghost" type="button" onClick={onSample} disabled={loading}>
            <Play size={17} />
            Try sample
          </button>
        </div>
        <div className="hero-feature-row">
          {features.map((feature) => (
            <div key={feature.label}>
              <span>{feature.label}</span>
              <strong>{feature.value}</strong>
            </div>
          ))}
        </div>
      </div>

      <div className="portal-stage" aria-hidden="true">
        <div className="orbit-ring ring-one" />
        <div className="orbit-ring ring-two" />
        <div className="portal-card upload-card">
          <Database size={22} />
          <span>sales_data.csv</span>
          <strong>42K rows scanned</strong>
        </div>
        <div className="portal-card insight-card">
          <Brain size={22} />
          <span>AI Insight</span>
          <strong>Revenue spikes every Q4</strong>
        </div>
        <div className="portal-card chart-card">
          <BarChart3 size={22} />
          <span>Auto chart</span>
          <strong>Heatmap + trendline ready</strong>
        </div>
        <div className="data-stream stream-a" />
        <div className="data-stream stream-b" />
        <div className="data-stream stream-c" />
        <div className="portal-core">
          <Sparkles size={34} />
          <span>Langalytics</span>
        </div>
      </div>
    </section>
  );
}

export default function App() {
  const [theme, setTheme] = useState("aurora");
  const [dataset, setDataset] = useState(null);
  const [activeTab, setActiveTab] = useState("overview");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [builder, setBuilder] = useState({
    type: "bar",
    x: "",
    y: "",
    aggregation: "sum",
    title: "",
  });
  const [charts, setCharts] = useState([]);
  const [selectedCharts, setSelectedCharts] = useState([]);
  const [insights, setInsights] = useState("");
  const [predictionTarget, setPredictionTarget] = useState("");
  const [predictionFeatures, setPredictionFeatures] = useState([]);
  const [basicPrediction, setBasicPrediction] = useState(null);
  const [advancedPrediction, setAdvancedPrediction] = useState(null);
  const [filterColumn, setFilterColumn] = useState("");
  const [filterValue, setFilterValue] = useState("");
  const [exporting, setExporting] = useState(false);
  const dashboardRef = useRef(null);

  const previewFigure = useMemo(() => {
    if (!dataset) return buildFigure(builder, null);
    return buildFigure({ ...builder, title: builder.title || titleFor(builder) }, dataset);
  }, [builder, dataset]);

  const filteredRecords = useMemo(() => {
    if (!dataset) return [];
    if (!filterColumn || !filterValue) return dataset.records.slice(0, 200);
    return dataset.records.filter((row) => String(row[filterColumn]) === filterValue).slice(0, 200);
  }, [dataset, filterColumn, filterValue]);

  const uniqueFilterValues = useMemo(() => {
    if (!dataset || !filterColumn) return [];
    return Array.from(new Set(dataset.records.map((row) => row[filterColumn]).filter((value) => value !== null)))
      .map(String)
      .slice(0, 100);
  }, [dataset, filterColumn]);

  useEffect(() => {
    if (!dataset) return;
    const defaultY = dataset.numericColumns[0] || dataset.columns[0] || "";
    const defaultX = dataset.categoricalColumns[0] || dataset.columns.find((column) => column !== defaultY) || defaultY;
    setBuilder({
      type: "bar",
      x: defaultX,
      y: defaultY,
      aggregation: "sum",
      title: `${defaultY} by ${defaultX}`,
    });
    setCharts([]);
    setSelectedCharts([]);
    setInsights("");
    setBasicPrediction(null);
    setAdvancedPrediction(null);
    setPredictionTarget(defaultY);
    setPredictionFeatures(dataset.numericColumns.filter((column) => column !== defaultY).slice(0, 3));
    setFilterColumn(dataset.columns[0] || "");
    setFilterValue("");
    setActiveTab("overview");
  }, [dataset]);

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setStatus("Uploading dataset...");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const payload = await apiFetch("/api/upload", { method: "POST", body: formData });
      setDataset(payload);
      setStatus("Dataset ready");
    } catch (error) {
      setStatus(error.message);
    } finally {
      setLoading(false);
      event.target.value = "";
    }
  }

  async function loadSample() {
    setLoading(true);
    setStatus("Loading sample...");
    try {
      const payload = await apiFetch("/api/sample", { method: "POST" });
      setDataset(payload);
      setStatus("Sample loaded");
    } catch (error) {
      setStatus(error.message);
    } finally {
      setLoading(false);
    }
  }

  function updateBuilder(updates) {
    setBuilder((current) => {
      const next = { ...current, ...updates };
      if (!Object.prototype.hasOwnProperty.call(updates, "title")) {
        next.title = titleFor(next);
      }
      return next;
    });
  }

  function addChart() {
    if (!dataset) return;
    const chart = {
      ...builder,
      id: makeId(),
      title: builder.title || titleFor(builder),
    };
    setCharts((items) => [chart, ...items]);
    setSelectedCharts((items) => [chart.id, ...items]);
    setActiveTab("visuals");
  }

  function removeChart(chartId) {
    setCharts((items) => items.filter((item) => item.id !== chartId));
    setSelectedCharts((items) => items.filter((item) => item !== chartId));
  }

  function toggleChart(chartId) {
    setSelectedCharts((items) =>
      items.includes(chartId) ? items.filter((item) => item !== chartId) : [...items, chartId],
    );
  }

  async function downloadPlot(divId, filename) {
    const node = document.getElementById(divId);
    if (!node) return;
    const image = await Plotly.toImage(node, { format: "png", width: 1400, height: 900, scale: 2 });
    downloadDataUrl(image, filename);
  }

  function mergeSelectedCharts() {
    const selected = charts.filter((chart) => selectedCharts.includes(chart.id));
    if (selected.length < 2) {
      setStatus("Select at least two dashboard charts");
      return;
    }

    const merged = {
      id: makeId("combo"),
      type: "combo",
      title: `Merged visual (${selected.length})`,
      components: selected,
    };
    setCharts((items) => [merged, ...items]);
    setSelectedCharts([merged.id]);
    setStatus("Merged visual added");
  }

  async function downloadDashboardPng() {
    if (!dashboardRef.current || !charts.length) return;
    setExporting(true);
    try {
      const canvas = await html2canvas(dashboardRef.current, {
        backgroundColor: "#f6f4ef",
        scale: 2,
        useCORS: true,
      });
      downloadDataUrl(canvas.toDataURL("image/png"), "visualization-dashboard.png");
    } finally {
      setExporting(false);
    }
  }

  function downloadDashboardJson() {
    if (!dataset) return;
    const payload = {
      dataset: {
        name: dataset.name,
        rows: dataset.overview.rows,
        columns: dataset.columns,
      },
      charts,
    };
    downloadBlob(JSON.stringify(payload, null, 2), "visualization-dashboard.json", "application/json");
  }

  async function downloadDashboardHtml() {
    if (!charts.length) return;
    setExporting(true);
    try {
      const imageCards = await Promise.all(
        charts.map(async (chart) => {
          const node = document.getElementById(`chart-${chart.id}`);
          const image = node ? await Plotly.toImage(node, { format: "png", width: 1200, height: 760, scale: 1 }) : "";
          return `<section><h2>${chart.title}</h2><img alt="${chart.title}" src="${image}" /></section>`;
        }),
      );

      const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>${dataset?.name || "Dashboard"}</title>
  <style>
    body { margin: 0; padding: 32px; background: #f6f4ef; color: #26312f; font-family: Arial, sans-serif; }
    header { margin-bottom: 24px; }
    h1 { font-size: 24px; margin: 0 0 8px; }
    main { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }
    section { background: white; border: 1px solid #ddd8ce; border-radius: 8px; padding: 16px; }
    h2 { font-size: 16px; margin: 0 0 12px; }
    img { width: 100%; height: auto; display: block; }
  </style>
</head>
<body>
  <header>
    <h1>${dataset?.name || "Visualization dashboard"}</h1>
    <div>${charts.length} visuals</div>
  </header>
  <main>${imageCards.join("")}</main>
</body>
</html>`;
      downloadBlob(html, "visualization-dashboard.html", "text/html");
    } finally {
      setExporting(false);
    }
  }

  async function loadInsights() {
    if (!dataset) return;
    setLoading(true);
    setStatus("Generating insights...");
    try {
      const payload = await apiFetch("/api/insights", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: dataset.datasetId }),
      });
      setInsights(payload.insights);
      setStatus("Insights ready");
    } catch (error) {
      setStatus(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function trainBasicPrediction() {
    if (!dataset) return;
    setLoading(true);
    setStatus("Training model...");
    try {
      const payload = await apiFetch("/api/predict/basic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: dataset.datasetId,
          target_column: predictionTarget,
          feature_columns: predictionFeatures,
        }),
      });
      setBasicPrediction(payload);
      setStatus("Model trained");
    } catch (error) {
      setStatus(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function compareModels() {
    if (!dataset) return;
    setLoading(true);
    setStatus("Comparing models...");
    try {
      const payload = await apiFetch("/api/predict/advanced", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: dataset.datasetId,
          target_column: predictionTarget,
        }),
      });
      setAdvancedPrediction(payload);
      setStatus("Model comparison ready");
    } catch (error) {
      setStatus(error.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell" data-theme={theme}>
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">
            <BarChart3 size={23} />
          </div>
          <div>
            <p className="eyebrow">AI automated data analysis</p>
            <h1>Langalytics</h1>
            <p className="hero-line">Upload data, generate insights, build charts, and predict outcomes.</p>
          </div>
        </div>
        <div className="topbar-actions">
          <div className="theme-switcher" aria-label="Theme colors">
            <Palette size={16} />
            {themes.map((item) => (
              <button
                type="button"
                key={item.id}
                className={theme === item.id ? "active" : ""}
                onClick={() => setTheme(item.id)}
                title={`${item.label} theme`}
                aria-label={`${item.label} theme`}
              >
                <span>{item.label}</span>
              </button>
            ))}
          </div>
          <label className="button primary">
            <Upload size={17} />
            Upload
            <input type="file" accept=".csv,.xlsx,.xls" onChange={handleUpload} />
          </label>
          <button className="button" type="button" onClick={loadSample} disabled={loading}>
            <Database size={17} />
            Sample
          </button>
        </div>
      </header>

      <main className={`workspace ${!dataset ? "workspace-landing" : ""}`}>
        {dataset && (
          <aside className="side-panel">
            <div className="dataset-block">
              <span className="panel-label">Dataset</span>
              <strong>{dataset.name}</strong>
              <p>{status || "Ready"}</p>
            </div>

            <nav className="tab-list" aria-label="Dashboard sections">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    type="button"
                    key={tab.id}
                    className={activeTab === tab.id ? "active" : ""}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    <Icon size={17} />
                    {tab.label}
                  </button>
                );
              })}
            </nav>

            <div className="side-metrics">
              <Metric label="Rows" value={formatNumber(dataset.overview.rows, 0)} />
              <Metric label="Columns" value={dataset.overview.columns} />
              <Metric label="Missing" value={formatNumber(dataset.overview.missingValues, 0)} />
              <Metric label="Numeric" value={dataset.overview.numericColumns} />
            </div>
          </aside>
        )}

        <section className="content-panel">
          {!dataset ? (
            <LandingHero onUpload={handleUpload} onSample={loadSample} loading={loading} />
          ) : (
            <>
              {activeTab === "overview" && (
                <section className="section-stack">
                  <div className="metric-grid">
                    <Metric label="Rows" value={formatNumber(dataset.overview.rows, 0)} />
                    <Metric label="Columns" value={dataset.overview.columns} />
                    <Metric label="Numeric columns" value={dataset.overview.numericColumns} />
                    <Metric label="Categorical columns" value={dataset.overview.categoricalColumns} />
                    <Metric label="Missing values" value={formatNumber(dataset.overview.missingValues, 0)} />
                    <Metric label="Duplicate rows" value={formatNumber(dataset.quality.duplicates, 0)} />
                  </div>

                  <div className="split-grid">
                    <section>
                      <div className="section-title">
                        <h2>Columns</h2>
                      </div>
                      <DataTable
                        columns={[
                          { key: "column", label: "Column" },
                          { key: "dtype", label: "Type" },
                          { key: "missing", label: "Missing" },
                          { key: "unique", label: "Unique" },
                        ]}
                        rows={dataset.columnInfo}
                      />
                    </section>
                    <section>
                      <div className="section-title">
                        <h2>Statistics</h2>
                      </div>
                      <DataTable
                        columns={[
                          { key: "column", label: "Column" },
                          { key: "mean", label: "Mean", render: (row) => formatNumber(row.mean) },
                          { key: "median", label: "Median", render: (row) => formatNumber(row.median) },
                          { key: "min", label: "Min", render: (row) => formatNumber(row.min) },
                          { key: "max", label: "Max", render: (row) => formatNumber(row.max) },
                        ]}
                        rows={dataset.stats}
                        empty="No numeric columns"
                      />
                    </section>
                  </div>
                </section>
              )}

              {activeTab === "insights" && (
                <section className="section-stack">
                  <div className="section-title">
                    <h2>AI Insights</h2>
                    <button className="button primary" type="button" onClick={loadInsights} disabled={loading}>
                      <Brain size={17} />
                      Generate
                    </button>
                  </div>
                  <pre className="insight-box">{insights || "No insights generated yet."}</pre>
                </section>
              )}

              {activeTab === "visuals" && (
                <section className="section-stack">
                  <div className="visual-builder">
                    <section className="controls-panel">
                      <div className="section-title">
                        <h2>Graph Builder</h2>
                      </div>

                      <label>
                        Type
                        <select value={builder.type} onChange={(event) => updateBuilder({ type: event.target.value })}>
                          {chartTypes.map((type) => (
                            <option key={type.value} value={type.value}>
                              {type.label}
                            </option>
                          ))}
                        </select>
                      </label>

                      <label>
                        X axis
                        <select value={builder.x} onChange={(event) => updateBuilder({ x: event.target.value })}>
                          {dataset.columns.map((column) => (
                            <option key={column} value={column}>
                              {column}
                            </option>
                          ))}
                        </select>
                      </label>

                      <label>
                        Y axis
                        <select value={builder.y} onChange={(event) => updateBuilder({ y: event.target.value })}>
                          {(dataset.numericColumns.length ? dataset.numericColumns : dataset.columns).map((column) => (
                            <option key={column} value={column}>
                              {column}
                            </option>
                          ))}
                        </select>
                      </label>

                      <label>
                        Aggregate
                        <select
                          value={builder.aggregation}
                          onChange={(event) => updateBuilder({ aggregation: event.target.value })}
                        >
                          <option value="sum">Sum</option>
                          <option value="average">Average</option>
                          <option value="count">Count</option>
                          <option value="min">Min</option>
                          <option value="max">Max</option>
                        </select>
                      </label>

                      <label>
                        Title
                        <input
                          type="text"
                          value={builder.title}
                          onChange={(event) => updateBuilder({ title: event.target.value })}
                        />
                      </label>

                      <div className="button-row">
                        <button className="button primary" type="button" onClick={addChart}>
                          <Plus size={17} />
                          Add
                        </button>
                        <button className="button" type="button" onClick={() => downloadPlot("preview-plot", "graph.png")}>
                          <Download size={17} />
                          Graph
                        </button>
                      </div>
                    </section>

                    <section className="preview-panel">
                      <PlotPanel divId="preview-plot" figure={previewFigure} />
                    </section>
                  </div>

                  <div className="dashboard-toolbar">
                    <div>
                      <h2>Visualization Dashboard</h2>
                      <span>{charts.length} visuals</span>
                    </div>
                    <div className="button-row">
                      <button className="button" type="button" onClick={mergeSelectedCharts} disabled={selectedCharts.length < 2}>
                        <Layers size={17} />
                        Merge
                      </button>
                      <button className="button" type="button" onClick={downloadDashboardPng} disabled={!charts.length || exporting}>
                        <Download size={17} />
                        PNG
                      </button>
                      <button className="button" type="button" onClick={downloadDashboardHtml} disabled={!charts.length || exporting}>
                        <FileDown size={17} />
                        HTML
                      </button>
                      <button className="button" type="button" onClick={downloadDashboardJson} disabled={!charts.length}>
                        <FileDown size={17} />
                        JSON
                      </button>
                    </div>
                  </div>

                  <div className="dashboard-grid" ref={dashboardRef}>
                    {charts.length ? (
                      charts.map((chart) => {
                        const figure = buildFigure(chart, dataset);
                        return (
                          <article className="chart-tile" key={chart.id}>
                            <div className="tile-toolbar">
                              <label className="select-check">
                                <input
                                  type="checkbox"
                                  checked={selectedCharts.includes(chart.id)}
                                  onChange={() => toggleChart(chart.id)}
                                />
                                <span>{chart.title}</span>
                              </label>
                              <div>
                                <button
                                  className="icon-button"
                                  type="button"
                                  title="Download graph"
                                  onClick={() => downloadPlot(`chart-${chart.id}`, `${chart.title || "graph"}.png`)}
                                >
                                  <Download size={16} />
                                </button>
                                <button className="icon-button danger" type="button" title="Remove" onClick={() => removeChart(chart.id)}>
                                  <Trash2 size={16} />
                                </button>
                              </div>
                            </div>
                            <PlotPanel divId={`chart-${chart.id}`} figure={figure} height={chart.type === "combo" ? figure.layout.height : 360} />
                          </article>
                        );
                      })
                    ) : (
                      <div className="dashboard-empty">No dashboard visuals yet.</div>
                    )}
                  </div>
                </section>
              )}

              {activeTab === "prediction" && (
                <section className="section-stack">
                  <div className="prediction-grid">
                    <section className="controls-panel">
                      <div className="section-title">
                        <h2>Prediction</h2>
                      </div>
                      <label>
                        Target
                        <select value={predictionTarget} onChange={(event) => setPredictionTarget(event.target.value)}>
                          {dataset.numericColumns.map((column) => (
                            <option key={column} value={column}>
                              {column}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Features
                        <select
                          multiple
                          value={predictionFeatures}
                          onChange={(event) =>
                            setPredictionFeatures(Array.from(event.target.selectedOptions).map((option) => option.value))
                          }
                        >
                          {dataset.numericColumns
                            .filter((column) => column !== predictionTarget)
                            .map((column) => (
                              <option key={column} value={column}>
                                {column}
                              </option>
                            ))}
                        </select>
                      </label>
                      <div className="button-row">
                        <button className="button primary" type="button" onClick={trainBasicPrediction} disabled={loading}>
                          <Play size={17} />
                          Train
                        </button>
                        <button className="button" type="button" onClick={compareModels} disabled={loading}>
                          <Sparkles size={17} />
                          Compare
                        </button>
                      </div>
                    </section>

                    <section className="result-panel">
                      {basicPrediction ? (
                        <>
                          <div className="metric-grid compact">
                            <Metric label="R2 score" value={formatNumber(basicPrediction.score, 4)} />
                            <Metric label="Features" value={basicPrediction.features.length} />
                            <Metric label="Intercept" value={formatNumber(basicPrediction.intercept)} />
                          </div>
                          <DataTable
                            columns={[
                              { key: "feature", label: "Feature" },
                              { key: "coefficient", label: "Coefficient", render: (row) => formatNumber(row.coefficient, 4) },
                            ]}
                            rows={Object.entries(basicPrediction.coefficients).map(([feature, coefficient]) => ({
                              feature,
                              coefficient,
                            }))}
                          />
                        </>
                      ) : (
                        <div className="dashboard-empty">No trained model yet.</div>
                      )}
                    </section>
                  </div>

                  {advancedPrediction && (
                    <section>
                      <div className="section-title">
                        <h2>Model Comparison</h2>
                        <span>{advancedPrediction.bestModel}</span>
                      </div>
                      <DataTable
                        columns={[
                          { key: "name", label: "Model" },
                          { key: "r2", label: "R2", render: (row) => formatNumber(row.r2, 4) },
                          { key: "rmse", label: "RMSE", render: (row) => formatNumber(row.rmse, 4) },
                          { key: "mae", label: "MAE", render: (row) => formatNumber(row.mae, 4) },
                        ]}
                        rows={advancedPrediction.results}
                      />
                    </section>
                  )}
                </section>
              )}

              {activeTab === "quality" && (
                <section className="section-stack">
                  <div className="metric-grid">
                    <Metric label="Rows" value={formatNumber(dataset.quality.shape.rows, 0)} />
                    <Metric label="Columns" value={dataset.quality.shape.columns} />
                    <Metric label="Missing values" value={formatNumber(dataset.quality.totalMissing, 0)} />
                    <Metric label="Duplicate rows" value={formatNumber(dataset.quality.duplicates, 0)} />
                  </div>

                  <div className="split-grid">
                    <section>
                      <div className="section-title">
                        <h2>Missing Values</h2>
                      </div>
                      <DataTable
                        columns={[
                          { key: "column", label: "Column" },
                          { key: "missing", label: "Missing" },
                        ]}
                        rows={Object.entries(dataset.quality.missing)
                          .map(([column, missing]) => ({ column, missing }))
                          .filter((row) => row.missing > 0)}
                        empty="No missing values"
                      />
                    </section>
                    <section>
                      <div className="section-title">
                        <h2>Correlations</h2>
                      </div>
                      <DataTable
                        columns={[
                          { key: "col1", label: "Column 1" },
                          { key: "col2", label: "Column 2" },
                          { key: "correlation", label: "r", render: (row) => formatNumber(row.correlation, 4) },
                        ]}
                        rows={dataset.correlation?.strongCorrelations || []}
                        empty="No strong correlations"
                      />
                    </section>
                  </div>

                  <section className="wide-plot">
                    <PlotPanel
                      divId="quality-heatmap"
                      figure={buildFigure({ id: "quality", type: "heatmap", title: "Correlation heatmap" }, dataset)}
                      height={430}
                    />
                  </section>
                </section>
              )}

              {activeTab === "data" && (
                <section className="section-stack">
                  <div className="filter-row">
                    <label>
                      Column
                      <select
                        value={filterColumn}
                        onChange={(event) => {
                          setFilterColumn(event.target.value);
                          setFilterValue("");
                        }}
                      >
                        {dataset.columns.map((column) => (
                          <option key={column} value={column}>
                            {column}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Value
                      <select value={filterValue} onChange={(event) => setFilterValue(event.target.value)}>
                        <option value="">All</option>
                        {uniqueFilterValues.map((value) => (
                          <option key={value} value={value}>
                            {value}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <DataTable
                    columns={dataset.columns.map((column) => ({ key: column, label: column }))}
                    rows={filteredRecords}
                  />
                </section>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}
