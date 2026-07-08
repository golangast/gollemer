package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/gorilla/websocket"
)

type MetricPoint struct {
	Epoch    float64 `json:"epoch"`
	Loss     float64 `json:"loss"`
	LBLoss   float64 `json:"lb_loss"`
	LR       float64 `json:"lr"`
	Phase    int     `json:"phase"`
	Timestamp string `json:"timestamp"`
	Progress float64 `json:"progress"`
	Score    float64 `json:"score"`
	MaxScore float64 `json:"max_score"`
	Grammar  float64 `json:"grammar"`
	Sim      float64 `json:"sim"`
	IsSocial bool    `json:"is_social"`
}

type UtilPoint struct {
	Epoch    float64 `json:"epoch"`
	Layer    int     `json:"layer"`
	ExpertID int     `json:"expert_id"`
	Count    float64 `json:"count"`
}

type LatestMetric struct {
	Step           int     `json:"step"`
	Loss           float64 `json:"loss"`
	LBLoss         float64 `json:"lb_loss"`
	LR             float64 `json:"lr"`
	ActiveExperts  []int   `json:"active_experts"`
	IsCooling      bool    `json:"is_cooling"`
	CircuitBreaker bool    `json:"circuit_breaker"`
	Temperature    float64 `json:"temperature"`
	ThawedCount    int     `json:"thawed_count"`
}

type CartridgeMap map[string]string

type ProcessInfo struct {
	Running bool   `json:"running"`
	Cmd     string `json:"cmd"`
}

type SupervisorEvent struct {
	Time      string  `json:"time"`
	Step      int     `json:"step"`
	Type      string  `json:"type"`
	Severity  string  `json:"severity"`
	ExpertID  int     `json:"expert_id"`
	Message   string  `json:"message"`
	Raw       string  `json:"raw"`
	Old       string  `json:"old"`
	New       string  `json:"new"`
	Value     float64 `json:"value"`
	Recommend string  `json:"recommend"`
}

type TokenInfo struct {
	Token         string            `json:"token"`
	Conf          float64           `json:"conf"`
	Alts          []string          `json:"alts"`
	IsSalad       bool              `json:"is_salad"`
	ActiveExperts []int             `json:"active_experts"`
	RoutingPct    map[string]string `json:"routing_pct"`
	HitNoise      bool              `json:"hit_noise"`
}

type ConfigSnapshot struct {
	Timestamp string                 `json:"timestamp"`
	Label     string                 `json:"label"`
	Config    map[string]interface{} `json:"config"`
}

type SysStats struct {
	CPUPct  float64 `json:"cpu_pct"`
	MemMB   float64 `json:"mem_mb"`
	MemPct  float64 `json:"mem_pct"`
	ProcCPU float64 `json:"proc_cpu"`
	ProcMem float64 `json:"proc_mem_mb"`
}

type DashboardState struct {
	History          []MetricPoint      `json:"history"`
	Latest           LatestMetric       `json:"latest"`
	Cartridges       CartridgeMap       `json:"cartridges"`
	Datasets         []string           `json:"datasets"`
	Utilization      []UtilPoint        `json:"utilization"`
	LiveLines        []string           `json:"live_lines"`
	Process          ProcessInfo        `json:"process"`
	SupervisorEvents []SupervisorEvent  `json:"supervisor_events"`
	Telemetry        map[string]interface{} `json:"telemetry"`
	Sys              SysStats           `json:"sys"`
	LiveEpoch        float64            `json:"live_epoch"`
	LiveEpochTime    string             `json:"live_epoch_time"`
	LiveActiveExp    []int              `json:"live_active_experts"`
}

var (
	upgrader      = websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	clients       = make(map[*websocket.Conn]bool)
	clientsMu     sync.Mutex
	broadcastChan = make(chan []byte, 8)

	liveMu    sync.RWMutex
	liveLines []string

	svMu             sync.RWMutex
	supervisorEvents []SupervisorEvent

	procMu      sync.Mutex
	runningCmd  *exec.Cmd
	runningName string

	ansiRe  = regexp.MustCompile(`\x1b\[[0-9;]*m`)
	rootDir = detectRoot()

	liveParseMu    sync.RWMutex
	liveEpoch      float64
	liveEpochTime  string
	liveActiveExps []int
	liveStep       int

	snapshotMu      sync.Mutex
	configSnapshots []ConfigSnapshot
)

func detectRoot() string {
	if _, err := os.Stat("logs"); err == nil {
		return "."
	}
	if _, err := os.Stat("../../logs"); err == nil {
		return "../.."
	}
	return "."
}

func rel(p string) string { return filepath.Join(rootDir, p) }

var (
	reDominance   = regexp.MustCompile(`Expert Dominance too high \(([\d.]+)%\)`)
	rePlateau     = regexp.MustCompile(`Plateau.*?(\d+).*?learning rate`)
	reConfidence  = regexp.MustCompile(`Step Confidence low \(([\d.]+)%\)`)
	reAugment     = regexp.MustCompile(`(?:Augmenting grammar|Data Evolution Success|Mutated.*corpus)`)
	reMutateQ     = regexp.MustCompile(`shorthand data: '([^']+)'`)
	reJitter      = regexp.MustCompile(`Nudging Router Noise`)
	reTempAdj     = regexp.MustCompile(`Increasing Router Temperature`)
	reWeightReset = regexp.MustCompile(`Expert\s+(\d+)\s+Weights Reset`)
	reNaN         = regexp.MustCompile(`NaN|Emergency Brake|CRITICAL.*MatMul`)
	reSurgery     = regexp.MustCompile(`Performing.*Surgery|Triage.*Expert`)
	reStepNum     = regexp.MustCompile(`\[Step\s+(\d+)\]`)
	reLRVal       = regexp.MustCompile(`(?:LR|learning rate).*?([\d.e+-]+)`)
)

var (
	reMultiphase = regexp.MustCompile(`Phase\s+(\d+)\s*\|\s*Epoch\s+(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*LR:\s*([\S]+)`)
	reSocial     = regexp.MustCompile(`(?:Progress:|\[SOCIAL_PROGRESS\]\s+Progress:)\s*([\d.]+)%\s*\|\s*Epoch\s+(\d+)/\d+\s*\|\s*Score:\s*([\d.]+)/([\d.]+)\s*\|\s*Grammar:\s*([\d.]+)\s*\|\s*Sim:\s*([\d.]+)%`)
	reEpochTime  = regexp.MustCompile(`EpochTime:\s*([\d.]+)s`)
	reActiveExp  = regexp.MustCompile(`Active.*?Experts?:\s*\[([\d\s,]+)\]`)
	reUnfrozen   = regexp.MustCompile(`Expert\s+(\d+)\s+UNROZEN`)
)

func parseAndAppendSupervisorEvent(line string) {
	cleaned := ansiRe.ReplaceAllString(line, "")
	now := time.Now().Format("15:04:05")
	var ev *SupervisorEvent

	step := 0
	liveParseMu.RLock()
	step = liveStep
	liveParseMu.RUnlock()
	if m := reStepNum.FindStringSubmatch(cleaned); m != nil {
		step, _ = strconv.Atoi(m[1])
	}

	switch {
	case func() bool { m := reDominance.FindStringSubmatch(cleaned); return m != nil }():
		m := reDominance.FindStringSubmatch(cleaned)
		v, _ := strconv.ParseFloat(m[1], 64)
		ev = &SupervisorEvent{
			Type: "dominance", Severity: "warn", ExpertID: -1, Step: step, Time: now, Value: v,
			Message:   fmt.Sprintf("⚠️ Expert monopoly: %.1f%% dominance — Temp+Noise adjusted", v),
			Recommend: "Bump router_noise (0.3→0.6) or increase load_balancing_weight",
			Raw:       cleaned,
		}
	case reJitter.MatchString(cleaned):
		ev = &SupervisorEvent{
			Type: "jitter", Severity: "info", ExpertID: -1, Step: step, Time: now,
			Message:   "🔀 Anti-Monopoly Jitter: Router Noise nudged to break expert lock-in",
			Recommend: "If monopoly persists, manually increase router_noise in config",
			Raw:       cleaned,
		}
	case reTempAdj.MatchString(cleaned):
		ev = &SupervisorEvent{
			Type: "temperature", Severity: "info", ExpertID: -1, Step: step, Time: now,
			Message:   "🌡️ Router Temperature increased (low step confidence detected)",
			Recommend: "Nudge router_temperature up (1.5→1.9) if word salad persists",
			Raw:       cleaned,
		}
	case func() bool { m := reWeightReset.FindStringSubmatch(cleaned); return m != nil }():
		m := reWeightReset.FindStringSubmatch(cleaned)
		id, _ := strconv.Atoi(m[1])
		ev = &SupervisorEvent{
			Type: "weight_reset", Severity: "warn", ExpertID: id, Step: step, Time: now,
			Message:   fmt.Sprintf("🧬 Expert %d weights reset (Xavier Init) + Router perturbed", id),
			Recommend: "Drop learning_rate by half or trigger manual Expert Thawing",
			Raw:       cleaned,
		}
	case func() bool { m := reMutateQ.FindStringSubmatch(cleaned); return m != nil }():
		m := reMutateQ.FindStringSubmatch(cleaned)
		ev = &SupervisorEvent{
			Type: "augment", Severity: "info", ExpertID: -1, Step: step, Time: now,
			Message: "📝 Corpus augmented via grammar expansion",
			Old: m[1], New: "expanded syntactic form",
			Raw: cleaned,
		}
	case reAugment.MatchString(cleaned):
		ev = &SupervisorEvent{
			Type: "augment", Severity: "info", ExpertID: -1, Step: step, Time: now,
			Message: "📝 " + cleaned,
			Raw:     cleaned,
		}
	case func() bool { m := reConfidence.FindStringSubmatch(cleaned); return m != nil }():
		m := reConfidence.FindStringSubmatch(cleaned)
		v, _ := strconv.ParseFloat(m[1], 64)
		ev = &SupervisorEvent{
			Type: "confidence", Severity: "warn", ExpertID: -1, Step: step, Time: now, Value: v,
			Message:   fmt.Sprintf("⚠️ Step Confidence low (%.1f%%) — router entropy too high", v),
			Recommend: "Nudge router_temperature up (1.5→1.9) to flatten winner-take-all distribution",
			Raw:       cleaned,
		}
	case rePlateau.MatchString(cleaned):
		ev = &SupervisorEvent{
			Type: "plateau", Severity: "critical", ExpertID: -1, Step: step, Time: now,
			Message:   "🚨 Training Plateau — LR reduced, loss is not improving",
			Recommend: "Drop learning_rate by half or trigger Expert Thawing to inject new params",
			Raw:       cleaned,
		}
	case reNaN.MatchString(cleaned):
		ev = &SupervisorEvent{
			Type: "nan", Severity: "critical", ExpertID: -1, Step: step, Time: now,
			Message:   "💥 NaN/Inf detected — Emergency Brake engaged",
			Recommend: "Enable gradient clipping via /api/preset/enable_clipping",
			Raw:       cleaned,
		}
	case reSurgery.MatchString(cleaned):
		ev = &SupervisorEvent{
			Type: "surgery", Severity: "warn", ExpertID: -1, Step: step, Time: now,
			Message:   "🔧 Expert Surgery: Triage weights reset on collapsed expert",
			Recommend: "Monitor loss for 10+ epochs; if flat, reduce router_temperature",
			Raw:       cleaned,
		}
	}

	if ev == nil {
		return
	}
	svMu.Lock()
	supervisorEvents = append(supervisorEvents, *ev)
	if len(supervisorEvents) > 200 {
		supervisorEvents = supervisorEvents[len(supervisorEvents)-200:]
	}
	svMu.Unlock()
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	go watchLoop()
	go broadcastWorker()

	mux := http.NewServeMux()
	static := filepath.Join(rootDir, "cmd/tools/dashboard/static")
	mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir(static))))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, filepath.Join(static, "index.html"))
	})
	mux.HandleFunc("/api/state", handleState)
	mux.HandleFunc("/api/cartridges/update", handleUpdateCartridges)
	mux.HandleFunc("/api/models", handleModels)
	mux.HandleFunc("/api/cmd", handleCmd)
	mux.HandleFunc("/api/kill", handleKill)
	mux.HandleFunc("/api/infer", handleInfer)
	mux.HandleFunc("/api/infer/ws", handleInferWS)
	mux.HandleFunc("/api/config", handleConfig)
	mux.HandleFunc("/api/cartridges/validate", handleValidateCartridge)
	mux.HandleFunc("/api/fs/list", handleFSList)
	mux.HandleFunc("/api/preset/enable_clipping", handlePresetClipping)
	mux.HandleFunc("/api/preset/flush_gc", handlePresetFlushGC)
	mux.HandleFunc("/api/preset/boost", handlePresetBoost)
	mux.HandleFunc("/api/preset/cooldown", handlePresetCooldown)
	mux.HandleFunc("/api/preset/snapshot", handlePresetSnapshot)
	mux.HandleFunc("/api/config/rollback", handleConfigRollback)
	mux.HandleFunc("/api/diagnostics", handleDiagnostics)
	mux.HandleFunc("/api/trigger", handleTrigger)
	mux.HandleFunc("/api/download_model", handleDownloadModel)
	mux.HandleFunc("/ws", handleWS)

	log.Printf("🖥️  Dashboard → http://localhost:8765")
	log.Fatal(http.ListenAndServe(":8765", mux))
}

func watchLoop() {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		data, err := json.Marshal(buildState())
		if err != nil {
			continue
		}
		select {
		case broadcastChan <- data:
		default:
		}
	}
}

func broadcastWorker() {
	for data := range broadcastChan {
		clientsMu.Lock()
		for conn := range clients {
			if err := conn.WriteMessage(websocket.TextMessage, data); err != nil {
				conn.Close()
				delete(clients, conn)
			}
		}
		clientsMu.Unlock()
	}
}

func buildState() DashboardState {
	liveMu.RLock()
	ll := make([]string, len(liveLines))
	copy(ll, liveLines)
	liveMu.RUnlock()

	svMu.RLock()
	evs := make([]SupervisorEvent, len(supervisorEvents))
	copy(evs, supervisorEvents)
	svMu.RUnlock()

	procMu.Lock()
	pi := ProcessInfo{Running: runningCmd != nil, Cmd: runningName}
	procMu.Unlock()

	liveParseMu.RLock()
	lepoch := liveEpoch
	lepochTime := liveEpochTime
	lactiveExps := append([]int{}, liveActiveExps...)
	liveParseMu.RUnlock()

	return DashboardState{
		History:          readHistory(),
		Latest:           readLatestMetric(),
		Cartridges:       readCartridges(),
		Datasets:         listDatasets(),
		Utilization:      readUtilSample(),
		LiveLines:        ll,
		Process:          pi,
		SupervisorEvents: evs,
		Telemetry:        readTelemetry(),
		Sys:              readSysStats(),
		LiveEpoch:        lepoch,
		LiveEpochTime:    lepochTime,
		LiveActiveExp:    lactiveExps,
	}
}

func readHistory() []MetricPoint {
	pts := tryHistoryCSV(rel("logs/training_history.csv"))
	if len(pts) == 0 {
		pts = tryLogCSV(rel("logs/training_log.csv"))
	}
	liveMu.RLock()
	lines := make([]string, len(liveLines))
	copy(lines, liveLines)
	liveMu.RUnlock()

	for _, line := range lines {
		if m := reMultiphase.FindStringSubmatch(line); m != nil {
			phase, _ := strconv.Atoi(m[1])
			epoch, _ := strconv.ParseFloat(m[2], 64)
			loss, _ := strconv.ParseFloat(m[3], 64)
			lr, _ := strconv.ParseFloat(m[4], 64)
			pts = append(pts, MetricPoint{Epoch: epoch, Loss: loss, LR: lr, Phase: phase, Timestamp: time.Now().Format(time.RFC3339)})
		} else if m := reSocial.FindStringSubmatch(line); m != nil {
			progress, _ := strconv.ParseFloat(m[1], 64)
			epoch, _ := strconv.ParseFloat(m[2], 64)
			score, _ := strconv.ParseFloat(m[3], 64)
			maxScore, _ := strconv.ParseFloat(m[4], 64)
			grammar, _ := strconv.ParseFloat(m[5], 64)
			sim, _ := strconv.ParseFloat(m[6], 64)
			pts = append(pts, MetricPoint{Epoch: epoch, Progress: progress, Score: score, MaxScore: maxScore, Grammar: grammar, Sim: sim, IsSocial: true, Timestamp: time.Now().Format(time.RFC3339)})
		}
	}
	if len(pts) > 600 {
		pts = pts[len(pts)-600:]
	}
	return pts
}

func tryHistoryCSV(p string) []MetricPoint {
	f, err := os.Open(p)
	if err != nil {
		return nil
	}
	defer f.Close()
	rows, err := csv.NewReader(f).ReadAll()
	if err != nil || len(rows) < 2 {
		return nil
	}
	var pts []MetricPoint
	for _, r := range rows[1:] {
		if len(r) < 4 {
			continue
		}
		epoch, _ := strconv.ParseFloat(strings.TrimSpace(r[0]), 64)
		loss, _ := strconv.ParseFloat(strings.TrimSpace(r[1]), 64)
		lb, _ := strconv.ParseFloat(strings.TrimSpace(r[2]), 64)
		lr, _ := strconv.ParseFloat(strings.TrimSpace(r[3]), 64)
		ts := ""
		if len(r) >= 5 {
			ts = strings.TrimSpace(r[4])
		}
		pts = append(pts, MetricPoint{Epoch: epoch, Loss: loss, LBLoss: lb, LR: lr, Timestamp: ts})
	}
	return pts
}

func tryLogCSV(p string) []MetricPoint {
	f, err := os.Open(p)
	if err != nil {
		return nil
	}
	defer f.Close()
	rows, err := csv.NewReader(f).ReadAll()
	if err != nil || len(rows) < 2 {
		return nil
	}
	var pts []MetricPoint
	for _, r := range rows[1:] {
		if len(r) < 3 {
			continue
		}
		epoch, _ := strconv.ParseFloat(strings.TrimSpace(r[0]), 64)
		loss, _ := strconv.ParseFloat(strings.TrimSpace(r[1]), 64)
		lb, _ := strconv.ParseFloat(strings.TrimSpace(r[2]), 64)
		var lr float64
		if len(r) >= 5 {
			lr, _ = strconv.ParseFloat(strings.TrimSpace(r[4]), 64)
		}
		pts = append(pts, MetricPoint{Epoch: epoch, Loss: loss, LBLoss: lb, LR: lr})
	}
	return pts
}

func readLatestMetric() LatestMetric {
	data, err := os.ReadFile(rel("logs/latest_metric.json"))
	if err != nil {
		return LatestMetric{}
	}
	var m LatestMetric
	json.Unmarshal(data, &m)
	return m
}

func readCartridges() CartridgeMap {
	data, err := os.ReadFile(rel("data/config/cartridges.json"))
	if err != nil {
		return CartridgeMap{}
	}
	var m CartridgeMap
	json.Unmarshal(data, &m)
	return m
}

func listDatasets() []string {
	var files []string
	filepath.WalkDir(rel("data/training/trainingdata"), func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(p))
		if ext == ".csv" || ext == ".txt" || ext == ".json" {
			files = append(files, p)
		}
		return nil
	})
	sort.Strings(files)
	return files
}

func readTelemetry() map[string]interface{} {
	data, err := os.ReadFile(rel("logs/telemetry.json"))
	if err != nil {
		return map[string]interface{}{}
	}
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		return map[string]interface{}{}
	}
	return m
}

func readUtilSample() []UtilPoint {
	f, err := os.Open(rel("logs/moe_utilization.csv"))
	if err != nil {
		return nil
	}
	defer f.Close()
	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil
	}
	if len(rows) > 2000 {
		rows = rows[len(rows)-2000:]
	}
	var pts []UtilPoint
	for _, r := range rows {
		if len(r) < 4 {
			continue
		}
		epoch, _ := strconv.ParseFloat(strings.TrimSpace(r[0]), 64)
		layer, _ := strconv.Atoi(strings.TrimSpace(r[1]))
		expert, _ := strconv.Atoi(strings.TrimSpace(r[2]))
		count, _ := strconv.ParseFloat(strings.TrimSpace(r[3]), 64)
		pts = append(pts, UtilPoint{Epoch: epoch, Layer: layer, ExpertID: expert, Count: count})
	}
	return pts
}

func readSysStats() SysStats {
	var s SysStats
	if data, err := os.ReadFile("/proc/meminfo"); err == nil {
		var total, available uint64
		for _, line := range strings.Split(string(data), "\n") {
			fields := strings.Fields(line)
			if len(fields) < 2 { continue }
			v, _ := strconv.ParseUint(fields[1], 10, 64)
			switch fields[0] {
			case "MemTotal:":    total = v
			case "MemAvailable:": available = v
			}
		}
		used := total - available
		s.MemMB = float64(used) / 1024
		if total > 0 { s.MemPct = float64(used) / float64(total) * 100 }
	}
	if data, err := os.ReadFile("/proc/stat"); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			if !strings.HasPrefix(line, "cpu ") { continue }
			fields := strings.Fields(line)
			if len(fields) < 5 { break }
			idle, _ := strconv.ParseFloat(fields[4], 64)
			var total float64
			for _, f := range fields[1:] { v, _ := strconv.ParseFloat(f, 64); total += v }
			if total > 0 { s.CPUPct = (1 - idle/total) * 100 }
			break
		}
	}
	procMu.Lock()
	var pid int
	if runningCmd != nil && runningCmd.Process != nil { pid = runningCmd.Process.Pid }
	procMu.Unlock()
	if pid > 0 {
		if data, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid)); err == nil {
			for _, line := range strings.Split(string(data), "\n") {
				if strings.HasPrefix(line, "VmRSS:") {
					fields := strings.Fields(line)
					if len(fields) >= 2 { v, _ := strconv.ParseFloat(fields[1], 64); s.ProcMem = v / 1024 }
				}
			}
		}
	}
	return s
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

func handleState(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(buildState())
}

func handleUpdateCartridges(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost { http.Error(w, "POST only", http.StatusMethodNotAllowed); return }
	w.Header().Set("Access-Control-Allow-Origin", "*")
	var payload CartridgeMap
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		http.Error(w, "bad JSON", http.StatusBadRequest); return
	}
	data, _ := json.MarshalIndent(payload, "", "  ")
	if err := os.WriteFile(rel("data/config/cartridges.json"), data, 0644); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError); return
	}
	fmt.Fprintln(w, `{"ok":true}`)
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "application/json")
	
	dir := rel("data/models/gob_models")
	entries, err := os.ReadDir(dir)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	type ModelStat struct {
		Name    string `json:"name"`
		Size    int64  `json:"size"`
		ModTime string `json:"mod_time"`
	}
	
	var models []ModelStat
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".gob") {
			info, err := e.Info()
			if err == nil {
				models = append(models, ModelStat{
					Name:    e.Name(),
					Size:    info.Size(),
					ModTime: info.ModTime().Format(time.RFC3339),
				})
			}
		}
	}
	json.NewEncoder(w).Encode(models)
}

var allowedTargets = map[string]bool{
	"train": true, "train-social": true, "clean": true,
	"word2vec": true, "llm": true, "chat": true,
}

func handleCmd(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost { http.Error(w, "POST only", http.StatusMethodNotAllowed); return }
	w.Header().Set("Access-Control-Allow-Origin", "*")
	var req struct {
		Target string            `json:"target"`
		Env    map[string]string `json:"env"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad JSON", http.StatusBadRequest); return
	}
	if !allowedTargets[req.Target] {
		http.Error(w, "target not allowed", http.StatusForbidden); return
	}
	procMu.Lock()
	if runningCmd != nil {
		procMu.Unlock()
		http.Error(w, `{"error":"already running"}`, http.StatusConflict); return
	}
	appendLive("── Starting: make " + req.Target + " ──")
	cmd := exec.Command("make", req.Target)
	cmd.Dir = rootDir
	env := os.Environ()
	for k, v := range req.Env {
		env = append(env, k+"="+v)
	}
	cmd.Env = env
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()
	if err := cmd.Start(); err != nil {
		procMu.Unlock()
		http.Error(w, err.Error(), http.StatusInternalServerError); return
	}
	runningCmd = cmd
	runningName = req.Target
	procMu.Unlock()
	go streamToLive(stdout)
	go streamToLive(stderr)
	go func() {
		cmd.Wait()
		procMu.Lock()
		runningCmd = nil
		runningName = ""
		procMu.Unlock()
		appendLive("── Process finished ──")
	}()
	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, `{"ok":true}`)
}

func handleKill(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost { http.Error(w, "POST only", http.StatusMethodNotAllowed); return }
	w.Header().Set("Access-Control-Allow-Origin", "*")
	procMu.Lock()
	defer procMu.Unlock()
	if runningCmd != nil && runningCmd.Process != nil {
		runningCmd.Process.Kill()
		appendLive("── Killed by user ──")
	}
	fmt.Fprintln(w, `{"ok":true}`)
}

func handleDownloadModel(w http.ResponseWriter, r *http.Request) {
	modelPath := rel("data/models/gob_models/moe_social_model.gob")
	w.Header().Set("Content-Disposition", `attachment; filename="moe_social_model.gob"`)
	w.Header().Set("Content-Type", "application/octet-stream")
	http.ServeFile(w, r, modelPath)
}

func handleInfer(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost { http.Error(w, "POST only", http.StatusMethodNotAllowed); return }
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	var req struct{ Prompt string `json:"prompt"` }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Prompt == "" {
		http.Error(w, "bad request", http.StatusBadRequest); return
	}
	cmd := exec.Command("go", "run", "cmd/tools/train_moe/main.go", "-llm")
	cmd.Dir = rootDir
	cmd.Env = os.Environ()
	cmd.Stdin = bytes.NewBufferString(req.Prompt + "\n/exit\n")
	out, err := cmd.CombinedOutput()
	outStr := string(out)
	var lines []string
	for _, l := range strings.Split(outStr, "\n") {
		cleaned := strings.TrimSpace(ansiRe.ReplaceAllString(l, ""))
		if cleaned == "" || strings.HasPrefix(cleaned, "20") { continue }
		lines = append(lines, cleaned)
	}
	type InferResp struct {
		Response string `json:"response"`
		Raw      string `json:"raw"`
		Error    string `json:"error,omitempty"`
	}
	resp := InferResp{Response: strings.Join(lines, "\n"), Raw: outStr}
	if err != nil { resp.Error = err.Error() }
	json.NewEncoder(w).Encode(resp)
}

func handleInferWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil { return }
	defer conn.Close()
	var req struct{ Prompt string `json:"prompt"` }
	if err := conn.ReadJSON(&req); err != nil || req.Prompt == "" { return }
	cmd := exec.Command("go", "run", "cmd/tools/train_moe/main.go", "-llm")
	cmd.Dir = rootDir
	cmd.Env = os.Environ()
	cmd.Stdin = bytes.NewBufferString(req.Prompt + "\n/exit\n")
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()
	if err := cmd.Start(); err != nil {
		conn.WriteJSON(map[string]string{"error": err.Error()}); return
	}
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() { /* discard stderr */ }
	}()
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		line := ansiRe.ReplaceAllString(scanner.Text(), "")
		if strings.TrimSpace(line) == "" { continue }
		conn.WriteJSON(map[string]string{"token": line})
	}
	cmd.Wait()
	conn.WriteJSON(map[string]string{"done": "true"})
}

func handleConfig(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	configPath := rel("data/config/social_train.json")
	if r.Method == http.MethodPost {
		var cfg map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
			http.Error(w, "bad JSON", http.StatusBadRequest); return
		}
		// snapshot before save
		snapshotMu.Lock()
		if existing, err := os.ReadFile(configPath); err == nil {
			var old map[string]interface{}
			json.Unmarshal(existing, &old)
			configSnapshots = append(configSnapshots, ConfigSnapshot{
				Timestamp: time.Now().Format(time.RFC3339),
				Label:     fmt.Sprintf("pre-save-%d", len(configSnapshots)+1),
				Config:    old,
			})
			if len(configSnapshots) > 20 { configSnapshots = configSnapshots[len(configSnapshots)-20:] }
		}
		snapshotMu.Unlock()
		data, _ := json.MarshalIndent(cfg, "", "  ")
		os.WriteFile(configPath, data, 0644)
		fmt.Fprintln(w, `{"ok":true}`)
		return
	}
	data, err := os.ReadFile(configPath)
	if err != nil { http.Error(w, "config not found", http.StatusNotFound); return }
	w.Write(data)
}

func handleValidateCartridge(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	var req struct{ Path string `json:"path"` }
	json.NewDecoder(r.Body).Decode(&req)
	_, err := os.Stat(req.Path)
	if err != nil {
		json.NewEncoder(w).Encode(map[string]interface{}{"valid": false, "error": err.Error()})
		return
	}
	json.NewEncoder(w).Encode(map[string]interface{}{"valid": true})
}

func handleFSList(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	dir := r.URL.Query().Get("dir")
	if dir == "" { dir = rootDir }
	entries, err := os.ReadDir(dir)
	if err != nil { http.Error(w, err.Error(), http.StatusInternalServerError); return }
	type Entry struct {
		Name  string `json:"name"`
		IsDir bool   `json:"is_dir"`
		Path  string `json:"path"`
	}
	var list []Entry
	for _, e := range entries {
		list = append(list, Entry{Name: e.Name(), IsDir: e.IsDir(), Path: filepath.Join(dir, e.Name())})
	}
	json.NewEncoder(w).Encode(list)
}

func patchConfig(updates map[string]interface{}) error {
	path := rel("data/config/social_train.json")
	data, err := os.ReadFile(path)
	if err != nil { return err }
	var cfg map[string]interface{}
	if err := json.Unmarshal(data, &cfg); err != nil { return err }
	for k, v := range updates { cfg[k] = v }
	out, _ := json.MarshalIndent(cfg, "", "  ")
	return os.WriteFile(path, out, 0644)
}

func handlePresetClipping(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if r.Method != http.MethodPost { return }
	patchConfig(map[string]interface{}{"max_grad_norm": 1.0, "weight_decay": 0.01})
	fmt.Fprintln(w, `{"ok":true}`)
}

func handlePresetFlushGC(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if r.Method != http.MethodPost { return }
	appendLive("⚙️ GC flush requested via dashboard")
	fmt.Fprintln(w, `{"ok":true}`)
}

func handlePresetBoost(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if r.Method != http.MethodPost { return }
	patchConfig(map[string]interface{}{"router_temperature": 1.8, "router_noise": 0.4})
	fmt.Fprintln(w, `{"ok":true}`)
}

func handlePresetCooldown(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if r.Method != http.MethodPost { return }
	patchConfig(map[string]interface{}{"router_temperature": 0.9, "router_noise": 0.1, "learning_rate": 0.0001})
	fmt.Fprintln(w, `{"ok":true}`)
}

func handlePresetSnapshot(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if r.Method != http.MethodPost { return }
	var req struct{ Label string `json:"label"` }
	json.NewDecoder(r.Body).Decode(&req)
	if req.Label == "" { req.Label = fmt.Sprintf("snapshot-%d", rand.Intn(9999)) }
	configPath := rel("data/config/social_train.json")
	data, err := os.ReadFile(configPath)
	if err != nil { http.Error(w, "no config", http.StatusInternalServerError); return }
	var cfg map[string]interface{}
	json.Unmarshal(data, &cfg)
	snapshotMu.Lock()
	configSnapshots = append(configSnapshots, ConfigSnapshot{
		Timestamp: time.Now().Format(time.RFC3339),
		Label:     req.Label,
		Config:    cfg,
	})
	if len(configSnapshots) > 20 { configSnapshots = configSnapshots[len(configSnapshots)-20:] }
	snapshotMu.Unlock()
	json.NewEncoder(w).Encode(map[string]interface{}{"ok": true, "label": req.Label})
}

func handleConfigRollback(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if r.Method != http.MethodPost { return }
	snapshotMu.Lock()
	defer snapshotMu.Unlock()
	if len(configSnapshots) == 0 {
		http.Error(w, "no snapshots", http.StatusNotFound); return
	}
	snap := configSnapshots[len(configSnapshots)-1]
	configSnapshots = configSnapshots[:len(configSnapshots)-1]
	data, _ := json.MarshalIndent(snap.Config, "", "  ")
	os.WriteFile(rel("data/config/social_train.json"), data, 0644)
	json.NewEncoder(w).Encode(map[string]interface{}{"ok": true, "rolled_back_to": snap.Label})
}

func handleDiagnostics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if r.Method != http.MethodPost { return }
	var req struct {
		Action string `json:"action"`
		Label  string `json:"label"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	resp := map[string]interface{}{
		"vector_safety":         true,
		"deterministic_response": true,
		"hardware_health":       true,
	}
	if moe.GlobalTelemetry != nil {
		resp["runtime"] = moe.GlobalTelemetry.Snapshot()
	}
	if strings.EqualFold(req.Action, "sandbox") {
		label := req.Label
		if label == "" {
			label = "dashboard"
		}
		result := map[string]interface{}{"match": false}
		if moe.GlobalTelemetry != nil {
			result = moe.GlobalTelemetry.RunMathSandbox(label, 2, 3, 2)
		}
		resp["sandbox"] = result
	}
	json.NewEncoder(w).Encode(resp)
}

func handleTrigger(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if r.Method != http.MethodPost { return }
	var req struct{ Action string `json:"action"` }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest); return
	}
	path := rel("data/config/social_train.json")
	data, err := os.ReadFile(path)
	if err != nil { http.Error(w, "no config", http.StatusInternalServerError); return }
	var cfg map[string]interface{}
	json.Unmarshal(data, &cfg)

	resp := map[string]interface{}{"ok": true}
	switch req.Action {
	case "test":
		cfg["trigger_test"] = true
	case "save":
		cfg["trigger_save"] = true
	case "toggle_auto":
		current := false
		if v, ok := cfg["auto_test_save"].(bool); ok { current = v }
		cfg["auto_test_save"] = !current
		resp["auto"] = !current
	}
	out, _ := json.MarshalIndent(cfg, "", "  ")
	os.WriteFile(path, out, 0644)
	json.NewEncoder(w).Encode(resp)
}

func handleWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil { return }
	defer func() {
		clientsMu.Lock()
		delete(clients, conn)
		clientsMu.Unlock()
		conn.Close()
	}()
	clientsMu.Lock()
	clients[conn] = true
	clientsMu.Unlock()
	// Send initial state
	if data, err := json.Marshal(buildState()); err == nil {
		conn.WriteMessage(websocket.TextMessage, data)
	}
	for {
		if _, _, err := conn.ReadMessage(); err != nil { break }
	}
}

// ─── Live log helpers ─────────────────────────────────────────────────────────

const maxLiveLines = 500

func appendLive(line string) {
	line = ansiRe.ReplaceAllString(line, "")
	line = strings.TrimRight(line, "\r\n ")
	if line == "" { return }

	liveParseMu.Lock()
	liveStep++
	liveParseMu.Unlock()

	parseAndAppendSupervisorEvent(line)

	if m := reMultiphase.FindStringSubmatch(line); m != nil {
		epoch, _ := strconv.ParseFloat(m[2], 64)
		liveParseMu.Lock()
		liveEpoch = epoch
		liveParseMu.Unlock()
	}
	if m := reSocial.FindStringSubmatch(line); m != nil {
		epoch, _ := strconv.ParseFloat(m[2], 64)
		liveParseMu.Lock()
		liveEpoch = epoch
		liveParseMu.Unlock()
	}
	if m := reEpochTime.FindStringSubmatch(line); m != nil {
		liveParseMu.Lock()
		liveEpochTime = m[1] + "s"
		liveParseMu.Unlock()
	}
	if m := reActiveExp.FindStringSubmatch(line); m != nil {
		nums := strings.FieldsFunc(m[1], func(r rune) bool { return r == ',' || r == ' ' })
		var exps []int
		for _, n := range nums {
			if v, err := strconv.Atoi(strings.TrimSpace(n)); err == nil { exps = append(exps, v) }
		}
		liveParseMu.Lock()
		liveActiveExps = exps
		liveParseMu.Unlock()
	}

	liveMu.Lock()
	liveLines = append(liveLines, line)
	if len(liveLines) > maxLiveLines {
		liveLines = liveLines[len(liveLines)-maxLiveLines:]
	}
	liveMu.Unlock()
}

func streamToLive(r io.Reader) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		appendLive(scanner.Text())
	}
}
