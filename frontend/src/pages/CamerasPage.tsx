import { useState, useEffect, useCallback, useRef } from "react";
import { apiBase } from "../hooks/useApi";
import type {
  AutoMeasurement,
  SelectedYoloClass,
  WeightsStatus,
  YoloClass,
} from "../types/detection";
import {
  minimumClassConfidence,
  modelClassKey,
  normalizeClassConfidence,
} from "../utils/detectionPairs";
import CameraFormModal from "../components/CameraFormModal";
import CameraGrid from "../components/CameraGrid";
import MeasurementClassModal from "../components/MeasurementClassModal";

export interface Cam {
  id: number;
  name: string;
  rtsp_url: string;
  stream_key: string;
  created_at: string;
}

interface Stat {
  active: boolean;
  readers: number;
}

const MAX_IPCAMS_FALLBACK = 16; // spec F4 — /api/config 로딩 전 기본값. 실제 cap 은 백엔드 env.
const MEASURE_STORAGE_KEY = "cam-distance:auto-measure:v1";
const EMPTY_CLASSES: SelectedYoloClass[] = [];
const WEIGHTS_STATUS_ERROR = "활성 가중치 정보를 불러오지 못했습니다.";
const CLASSES_STATUS_ERROR = "활성 YOLO 클래스 목록을 불러오지 못했습니다.";
const DROPPED_SELECTION_NOTICE = "가중치가 변경되어 측정 클래스를 다시 선택해야 합니다.";

type RequestFn = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

export function readWeightsStatus(value: unknown): WeightsStatus | null {
  if (typeof value !== "object" || value === null) return null;
  const status = value as {
    active_name?: unknown;
    default_name?: unknown;
    uploaded_at?: unknown;
    refresh_error?: unknown;
  };
  if (typeof status.active_name !== "string") return null;
  if (status.default_name !== undefined && typeof status.default_name !== "string") return null;
  if (status.uploaded_at !== null && typeof status.uploaded_at !== "string") return null;
  if (status.refresh_error !== undefined && typeof status.refresh_error !== "string") return null;
  return value as WeightsStatus;
}

export function filterMeasurementsForActiveModel(
  current: Record<string, AutoMeasurement>,
  value: unknown,
): { kept: Record<string, AutoMeasurement>; droppedStreamKeys: string[] } {
  const status = readWeightsStatus(value);
  if (status === null) return { kept: current, droppedStreamKeys: [] };
  const keptEntries: [string, AutoMeasurement][] = [];
  const droppedStreamKeys: string[] = [];
  Object.entries(current).forEach(([streamKey, measurement]) => {
    const keep = measurement.classes.length >= 1
      && measurement.classes.length <= 2
      && measurement.weights_uploaded_at === status.uploaded_at
      && measurement.classes.every((item) => item.model === status.active_name);
    if (keep) keptEntries.push([streamKey, measurement]);
    else droppedStreamKeys.push(streamKey);
  });
  return { kept: Object.fromEntries(keptEntries), droppedStreamKeys };
}

export function keepMeasurementsForActiveModel(
  current: Record<string, AutoMeasurement>,
  value: unknown,
): Record<string, AutoMeasurement> {
  return filterMeasurementsForActiveModel(current, value).kept;
}

export async function disableInferenceForStreamKeys(
  streamKeys: string[],
  request: RequestFn = fetch,
): Promise<string[]> {
  const baseUrl = typeof window === "undefined" ? "" : apiBase();
  const results = await Promise.all(streamKeys.map(async (streamKey) => {
    try {
      const response = await request(
        `${baseUrl}/api/ipcams/${encodeURIComponent(streamKey)}/inference`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: false }),
        },
      );
      return response.ok || response.status === 404 ? null : streamKey;
    } catch {
      return streamKey;
    }
  }));
  return results.filter((streamKey): streamKey is string => streamKey !== null);
}

export async function synchronizeMeasurementSelections(
  current: Record<string, AutoMeasurement>,
  next: Record<string, AutoMeasurement>,
  request: RequestFn = fetch,
): Promise<{ droppedStreamKeys: string[]; failedStreamKeys: string[] }> {
  const droppedStreamKeys = Object.keys(current).filter((streamKey) => (
    !Object.prototype.hasOwnProperty.call(next, streamKey)
  ));
  const failedStreamKeys = await disableInferenceForStreamKeys(droppedStreamKeys, request);
  return { droppedStreamKeys, failedStreamKeys };
}

export async function preserveWeightsResultWhileLoadingClasses(
  loadClasses: () => Promise<YoloClass[]>,
): Promise<YoloClass[]> {
  try {
    return await loadClasses();
  } catch {
    return [];
  }
}

export async function readAutoMeasurementResources(
  calibrationResponse: Response,
  weightsResponse: Response,
  classesResponse: Response,
): Promise<{
  canEnable: boolean;
  weights: WeightsStatus;
  classes: YoloClass[];
  classesError: string;
}> {
  if (!calibrationResponse.ok) throw new Error("기준점 정보를 불러오지 못했습니다.");
  if (!weightsResponse.ok) throw new Error(WEIGHTS_STATUS_ERROR);
  const calibration = await calibrationResponse.json() as {
    enabled?: unknown;
    homography?: unknown;
  };
  const weights = readWeightsStatus(await weightsResponse.json());
  if (weights === null) throw new Error(WEIGHTS_STATUS_ERROR);
  if (!classesResponse.ok) {
    return {
      canEnable: Boolean(calibration.enabled && calibration.homography),
      weights,
      classes: [],
      classesError: CLASSES_STATUS_ERROR,
    };
  }
  try {
    const classes = await classesResponse.json() as unknown;
    if (!Array.isArray(classes)) throw new Error(CLASSES_STATUS_ERROR);
    return {
      canEnable: Boolean(calibration.enabled && calibration.homography),
      weights,
      classes: classes as YoloClass[],
      classesError: "",
    };
  } catch {
    return {
      canEnable: Boolean(calibration.enabled && calibration.homography),
      weights,
      classes: [],
      classesError: CLASSES_STATUS_ERROR,
    };
  }
}

export function makeAutoMeasurement(
  enabled: boolean,
  classes: SelectedYoloClass[],
  weights: WeightsStatus,
): AutoMeasurement {
  return {
    enabled,
    classes,
    weights_uploaded_at: weights.uploaded_at,
  };
}

function loadStoredMeasurements(): Record<string, AutoMeasurement> {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(MEASURE_STORAGE_KEY) ?? "{}") as Record<
      string,
      Partial<AutoMeasurement>
    >;
    return Object.fromEntries(
      Object.entries(parsed).flatMap(([streamKey, value]) => {
        const classes = Array.isArray(value.classes)
          ? value.classes.flatMap((item) =>
              typeof item?.id === "number"
                && typeof item?.name === "string"
                && typeof item?.model === "string"
                ? [{
                    id: item.id,
                    name: item.name,
                    model: item.model,
                    conf: normalizeClassConfidence(item.conf),
                  }]
                : [],
            ).filter((item, index, all) => (
              all.findIndex((candidate) => (
                modelClassKey(candidate.model, candidate.id) === modelClassKey(item.model, item.id)
              )) === index
            )).slice(0, 2)
          : [];
        if (classes.length < 1 || classes.length > 2) return [];
        if (classes.some((item) => item.model !== classes[0].model)) return [];
        const weightsUploadedAt = typeof value.weights_uploaded_at === "string"
          || value.weights_uploaded_at === null
          ? value.weights_uploaded_at
          : undefined;
        return [[streamKey, {
          enabled: value.enabled === true,
          classes,
          weights_uploaded_at: weightsUploadedAt,
        }]];
      }),
    );
  } catch {
    return {};
  }
}

interface Props {
  // calibration 버튼 → App 이 풀페이지 CalibrationPage 로 전환.
  onCalibrate: (cam: Cam) => void;
}

export default function CamerasPage({ onCalibrate }: Props) {
  const [cams, setCams] = useState<Cam[]>([]);
  const [stats, setStats] = useState<Record<string, Stat>>({});
  // 실측 FPS — 그리드의 WhepPlayer 가 WebRTC getStats 로 올려주는 카메라별 디코딩 프레임레이트.
  const [fps, setFps] = useState<Record<string, number>>({});
  // 등록 cap — 백엔드 /api/config(MAX_IPCAMS env)에서 받음. 프론트 하드코딩 제거(P2-1).
  const [maxIpcams, setMaxIpcams] = useState(MAX_IPCAMS_FALLBACK);
  const [formOpen, setFormOpen] = useState(false);
  const [editCam, setEditCam] = useState<Cam | null>(null);
  const [error, setError] = useState("");
  const [autoMeasurements, setAutoMeasurements] = useState<Record<string, AutoMeasurement>>(
    loadStoredMeasurements,
  );
  const autoMeasurementsRef = useRef(autoMeasurements);
  autoMeasurementsRef.current = autoMeasurements;
  const [measureTarget, setMeasureTarget] = useState<Cam | null>(null);
  const [measureCanEnable, setMeasureCanEnable] = useState(false);
  const [yoloClasses, setYoloClasses] = useState<YoloClass[]>([]);
  const [weights, setWeights] = useState<WeightsStatus | null>(null);
  const [weightsBusy, setWeightsBusy] = useState(false);
  const [weightsError, setWeightsError] = useState("");
  const [weightsNotice, setWeightsNotice] = useState("");
  const [classesError, setClassesError] = useState("");
  const [selectionResetToken, setSelectionResetToken] = useState(0);
  const [measureBusyKey, setMeasureBusyKey] = useState<string | null>(null);
  const [measureSaving, setMeasureSaving] = useState(false);

  const replaceMeasurementSelections = useCallback(async (
    current: Record<string, AutoMeasurement>,
    next: Record<string, AutoMeasurement>,
    resetSelectionUi = false,
  ): Promise<void> => {
    if (resetSelectionUi) {
      window.localStorage.removeItem(MEASURE_STORAGE_KEY);
      setSelectionResetToken((token) => token + 1);
    }
    autoMeasurementsRef.current = next;
    setAutoMeasurements(next);
    const synchronization = synchronizeMeasurementSelections(current, next);
    const hasDroppedSelections = Object.keys(current).some((streamKey) => (
      !Object.prototype.hasOwnProperty.call(next, streamKey)
    ));
    if (resetSelectionUi || hasDroppedSelections) {
      setWeightsNotice(DROPPED_SELECTION_NOTICE);
    }
    const { failedStreamKeys } = await synchronization;
    if (failedStreamKeys.length > 0) {
      setError("일부 카메라의 자동 거리측정을 끄지 못했습니다. 다시 시도하세요.");
    }
  }, []);

  const fetchCams = useCallback(async () => {
    const resp = await fetch(`${apiBase()}/api/ipcams`);
    if (!resp.ok) return;
    setCams(await resp.json());
  }, []);

  useEffect(() => {
    fetchCams();
  }, [fetchCams]);

  useEffect(() => {
    window.localStorage.setItem(MEASURE_STORAGE_KEY, JSON.stringify(autoMeasurements));
  }, [autoMeasurements]);

  // 저장된 class_id는 가중치별 의미가 다르므로 현재 활성 모델과 같은 선택만 복원한다.
  useEffect(() => {
    let cancelled = false;
    fetch(`${apiBase()}/api/inference/weights`)
      .then((response) => (response.ok ? response.json() as Promise<unknown> : null))
      .then((payload) => {
        if (cancelled || payload == null) return;
        const status = readWeightsStatus(payload);
        if (status === null) {
          setWeightsError(WEIGHTS_STATUS_ERROR);
          setError(WEIGHTS_STATUS_ERROR);
          return;
        }
        setWeights(status);
        const current = autoMeasurementsRef.current;
        const filtered = filterMeasurementsForActiveModel(current, status);
        void replaceMeasurementSelections(current, filtered.kept);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [replaceMeasurementSelections]);

  // 등록 cap 을 백엔드에서 1회 로딩 (없으면 fallback 유지).
  useEffect(() => {
    fetch(`${apiBase()}/api/config`)
      .then((r) => (r.ok ? r.json() : null))
      .then((cfg) => {
        if (cfg?.max_ipcams) setMaxIpcams(cfg.max_ipcams);
      })
      .catch(() => {});
  }, []);

  // stats 1초 polling — 등록 카메라별 {active, readers} (mediamtx path 상태).
  useEffect(() => {
    let cancelled = false;
    async function poll() {
      const entries = await Promise.all(
        cams.map(async (c) => {
          try {
            const resp = await fetch(`${apiBase()}/api/ipcams/${c.stream_key}/stats`);
            if (!resp.ok) return [c.stream_key, { active: false, readers: 0 }] as const;
            return [c.stream_key, (await resp.json()) as Stat] as const;
          } catch {
            return [c.stream_key, { active: false, readers: 0 }] as const;
          }
        })
      );
      if (!cancelled) setStats(Object.fromEntries(entries));
    }
    poll();
    const timer = setInterval(poll, 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [cams]);

  const online = cams.filter((c) => stats[c.stream_key]?.active).length;
  const atCap = cams.length >= maxIpcams;

  const handleFps = useCallback((key: string, f: number) => {
    setFps((prev) => ({ ...prev, [key]: f }));
  }, []);

  // onSave 계약: 성공이면 null, 실패면 에러메시지(모달이 표시·열린 채 유지). POST 가
  // register-time ffprobe 로 수 초 걸릴 수 있어, 호출측(모달)이 await 하며 로딩상태를 보인다.
  async function handleSave(name: string, rtspUrl: string): Promise<string | null> {
    if (editCam) {
      const resp = await fetch(`${apiBase()}/api/ipcams/${editCam.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, rtsp_url: rtspUrl }),
      });
      if (!resp.ok) return "카메라 수정에 실패했습니다.";
    } else {
      const resp = await fetch(`${apiBase()}/api/ipcams`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, rtsp_url: rtspUrl }),
      });
      if (resp.status === 409) {
        const body = await resp.json().catch(() => ({}));
        return body.detail ?? `최대 ${maxIpcams}대까지 등록할 수 있습니다`;
      }
      if (!resp.ok) return "카메라 등록에 실패했습니다.";
    }
    await fetchCams();
    return null;
  }

  async function deleteCam(cam: Cam) {
    if (!window.confirm(`${cam.name} 삭제?`)) return;
    const resp = await fetch(`${apiBase()}/api/ipcams/${cam.id}`, { method: "DELETE" });
    if (!resp.ok) {
      setError("카메라 삭제에 실패했습니다.");
      return;
    }
    setAutoMeasurements((current) => {
      const next = { ...current };
      delete next[cam.stream_key];
      return next;
    });
    await fetchCams();
  }

  async function openAutoMeasurementSettings(cam: Cam) {
    setError("");
    setWeightsError("");
    setWeightsNotice("");
    setClassesError("");
    setYoloClasses([]);
    setMeasureBusyKey(cam.stream_key);
    try {
      const [calibrationResp, weightsResp, classesResp] = await Promise.all([
        fetch(`${apiBase()}/api/ipcams/${cam.stream_key}/calibration`),
        fetch(`${apiBase()}/api/inference/weights`),
        fetch(`${apiBase()}/api/inference/classes`),
      ]);
      const resources = await readAutoMeasurementResources(
        calibrationResp,
        weightsResp,
        classesResp,
      );
      setMeasureCanEnable(resources.canEnable);
      setWeights(resources.weights);
      setYoloClasses(resources.classes);
      setClassesError(resources.classesError);
      const current = autoMeasurementsRef.current;
      const filtered = filterMeasurementsForActiveModel(current, resources.weights);
      void replaceMeasurementSelections(current, filtered.kept);
      setMeasureTarget(cam);
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "자동 측정 설정에 실패했습니다.";
      setWeightsError(message);
      setError(message);
    } finally {
      setMeasureBusyKey(null);
    }
  }

  async function resetAllMeasurementSelections(): Promise<void> {
    await replaceMeasurementSelections(autoMeasurementsRef.current, {}, true);
  }

  async function readResponseError(response: Response, fallback: string): Promise<string> {
    const body = await response.json().catch(() => null) as { detail?: unknown } | null;
    return typeof body?.detail === "string" ? body.detail : fallback;
  }

  async function loadActiveClasses(): Promise<YoloClass[]> {
    const response = await fetch(`${apiBase()}/api/inference/classes`);
    if (!response.ok) {
      setClassesError(CLASSES_STATUS_ERROR);
      throw new Error(CLASSES_STATUS_ERROR);
    }
    try {
      const classes = await response.json() as unknown;
      if (!Array.isArray(classes)) throw new Error(CLASSES_STATUS_ERROR);
      setClassesError("");
      return classes as YoloClass[];
    } catch {
      setClassesError(CLASSES_STATUS_ERROR);
      throw new Error(CLASSES_STATUS_ERROR);
    }
  }

  async function uploadWeights(file: File) {
    setWeightsBusy(true);
    setWeightsError("");
    setWeightsNotice("");
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${apiBase()}/api/inference/weights`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        throw new Error(await readResponseError(response, "가중치 업로드에 실패했습니다."));
      }
      const nextWeights = readWeightsStatus(await response.json());
      if (nextWeights === null) throw new Error(WEIGHTS_STATUS_ERROR);
      setWeights(nextWeights);
      setWeightsError(nextWeights.refresh_error ?? "");
      await resetAllMeasurementSelections();
      setYoloClasses([]);
      setYoloClasses(await preserveWeightsResultWhileLoadingClasses(loadActiveClasses));
    } catch (reason) {
      setWeightsError(reason instanceof Error ? reason.message : "가중치 업로드에 실패했습니다.");
    } finally {
      setWeightsBusy(false);
    }
  }

  async function resetWeights() {
    setWeightsBusy(true);
    setWeightsError("");
    setWeightsNotice("");
    try {
      const response = await fetch(`${apiBase()}/api/inference/weights`, { method: "DELETE" });
      if (!response.ok) {
        throw new Error(await readResponseError(response, "기본 가중치 복귀에 실패했습니다."));
      }
      const nextWeights = readWeightsStatus(await response.json());
      if (nextWeights === null) throw new Error(WEIGHTS_STATUS_ERROR);
      setWeights(nextWeights);
      setWeightsError(nextWeights.refresh_error ?? "");
      await resetAllMeasurementSelections();
      setYoloClasses([]);
      setYoloClasses(await preserveWeightsResultWhileLoadingClasses(loadActiveClasses));
    } catch (reason) {
      setWeightsError(reason instanceof Error ? reason.message : "기본 가중치 복귀에 실패했습니다.");
    } finally {
      setWeightsBusy(false);
    }
  }

  async function applyAutoMeasurement(enabled: boolean, classes: SelectedYoloClass[]) {
    if (!measureTarget) return;
    setMeasureSaving(true);
    setError("");
    try {
      if (!weights) throw new Error("활성 가중치 정보를 불러오지 못했습니다.");
      const resp = await fetch(`${apiBase()}/api/ipcams/${measureTarget.stream_key}/inference`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          enabled
            ? {
                enabled: true,
                models: [weights.active_name],
                conf_threshold: minimumClassConfidence(classes),
              }
            : { enabled: false },
        ),
      });
      if (!resp.ok) throw new Error("자동 측정 설정을 저장하지 못했습니다.");
      setAutoMeasurements((current) => ({
        ...current,
        [measureTarget.stream_key]: makeAutoMeasurement(enabled, classes, weights),
      }));
      setMeasureTarget(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "자동 측정 설정을 저장하지 못했습니다.");
    } finally {
      setMeasureSaving(false);
    }
  }

  return (
    <>
      <header className="topbar">
        <div className="topbar-title">
          <h1>실시간 모니터</h1>
        </div>
        <div className="topbar-right">
          <div className="sysbar" role="status">
            <span className={online === cams.length && cams.length > 0 ? "sysbar-state ok" : "sysbar-state warn"}>
              <i className="sysbar-dot" />{online === cams.length && cams.length > 0 ? "시스템 정상" : "상태 확인"}
            </span>
            <span className="sysbar-sep">·</span>
            <span className="sysbar-item mono">온라인 {online}/{cams.length}</span>
          </div>
          <button
            className="btn primary"
            disabled={atCap}
            onClick={() => {
              setEditCam(null);
              setFormOpen(true);
            }}
          >
            ＋ 카메라 등록
          </button>
        </div>
      </header>

      <div className="content cameras-content">
        {error && <p className="form-error">{error}</p>}
        {weightsNotice && <p className="measure-page-notice" role="status">{weightsNotice}</p>}

        <section className="kpis" aria-label="카메라 현황">
          <div className="panel kpi">
            <div className="kpi-label">전체 카메라</div>
            <div className="kpi-value">{cams.length}<span className="kpi-sub"> 대</span></div>
            <div className="hint">등록된 스트림</div>
          </div>
          <div className="panel kpi kpi-ok">
            <div className="kpi-label">온라인</div>
            <div className="kpi-value">{online}<span className="kpi-sub"> 대</span></div>
            <div className="hint">영상 수신 가능</div>
          </div>
          <div className="panel kpi kpi-muted">
            <div className="kpi-label">오프라인</div>
            <div className="kpi-value">{cams.length - online}<span className="kpi-sub"> 대</span></div>
            <div className="hint">연결 확인 필요</div>
          </div>
          <div className="panel kpi">
            <div className="kpi-label">등록 용량</div>
            <div className="kpi-value">{cams.length}<span className="kpi-sub"> / {maxIpcams}</span></div>
            <div className="hint">최대 카메라 수</div>
          </div>
        </section>

        <section className="panel table-panel camera-table-panel">
          <div className="panel-head row-between">
            <strong>카메라 목록</strong>
          </div>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: 180 }}>카메라</th>
                  <th>RTSP URL</th>
                  <th style={{ width: 110 }}>상태</th>
                  <th style={{ width: 80 }}>FPS</th>
                  <th style={{ width: 330, textAlign: "right" }}>관리</th>
                </tr>
              </thead>
              <tbody>
                {cams.length === 0 && (
                  <tr>
                    <td colSpan={5} className="empty-cell">등록된 카메라가 없습니다.</td>
                  </tr>
                )}
                {cams.map((cam) => {
                  const st = stats[cam.stream_key];
                  const active = st?.active ?? false;
                  return (
                    <tr key={cam.id}>
                      <td>
                        <div className="cam-id">CAM-{String(cam.id).padStart(2, "0")}</div>
                        <div className="cam-name">{cam.name}</div>
                      </td>
                      <td className="url-cell" title={cam.rtsp_url}>{cam.rtsp_url}</td>
                      <td>
                        <span className={active ? "status status-on" : "status status-off"}>
                          <i className="dot" />{active ? "온라인" : "오프라인"}
                        </span>
                      </td>
                      <td className="mono">{active && fps[cam.stream_key] != null ? fps[cam.stream_key].toFixed(1) : "—"}</td>
                      <td className="table-actions">
                        <button className="btn sm" onClick={() => onCalibrate(cam)}>기준점 입력</button>
                        <button
                          className="btn sm measure-toggle"
                          disabled={measureBusyKey === cam.stream_key}
                          title="자동 거리측정 설정"
                          onClick={() => openAutoMeasurementSettings(cam)}
                        >
                          거리 측정
                        </button>
                        <button
                          className="btn sm"
                          onClick={() => {
                            setEditCam(cam);
                            setFormOpen(true);
                          }}
                        >
                          RTSP 수정
                        </button>
                        <button className="btn sm danger" onClick={() => deleteCam(cam)}>
                          삭제
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel live-grid-panel">
          <div className="panel-head row-between">
            <div>
              <h2>실시간 카메라</h2>
            </div>
            <span className="badge none">{cams.length} CH</span>
          </div>
          <div className="live-grid-body">
            <CameraGrid cams={cams} onFps={handleFps} autoMeasurements={autoMeasurements} />
          </div>
        </section>
      </div>

      <CameraFormModal
        open={formOpen}
        editCam={editCam}
        onClose={() => setFormOpen(false)}
        onSave={handleSave}
      />
      <MeasurementClassModal
        open={measureTarget != null}
        cameraName={measureTarget?.name ?? ""}
        classes={yoloClasses}
        initialEnabled={
          measureTarget ? autoMeasurements[measureTarget.stream_key]?.enabled ?? false : false
        }
        initialSelection={
          measureTarget
            ? autoMeasurements[measureTarget.stream_key]?.classes ?? EMPTY_CLASSES
            : EMPTY_CLASSES
        }
        canEnable={measureCanEnable}
        saving={measureSaving}
        weights={weights}
        weightsBusy={weightsBusy}
        weightsError={weightsError}
        weightsNotice={weightsNotice}
        classesError={classesError}
        selectionResetToken={selectionResetToken}
        onClose={() => setMeasureTarget(null)}
        onConfirm={applyAutoMeasurement}
        onUploadWeights={uploadWeights}
        onResetWeights={resetWeights}
      />
    </>
  );
}
